"""
This module implements the base functionality for MCMC-based samplers for NNs.
"""
import logging
import numpy as np
import scipy as sp
import tensorflow as tf
from sampler import Sampler, SampleStats

GRADIENT_CLIP_VALUE = 1e5


class MCMC_sampler(Sampler):
    """
    Base class for MCMC (HMC/LD) -based samplers for NNs.
    """

    def __new__(cls, **kwargs):
        """ Creates a new MCMCSampler object. """
        sampler = super().__new__(cls)

        # additional non-core parameters
        sampler._properties['noise_precision'] = 100.  # precision of the Gaussian used to model the noise
        sampler._properties['weights_precision'] = .01  # precision of the Gaussian prior on network parameters
        sampler._properties['resample_noise_precision'] = False
        sampler._properties['resample_weights_precision'] = False
        sampler._properties['seek_step_sizes'] = False
        sampler._properties['anneal_step_sizes'] = False
        sampler._properties['fade_in_velocities'] = False

        return sampler

    def __init__(self, loss_fn=None, initial_position=None, test_model=None, batch_size=None, burn_in=0,
                 step_sizes=.0001, step_probabilities=1., **kwargs):
        """
        Creates a new MCMC_sampler object.

        :param loss_fn: Target loss function without regularisaion terms
        :param initial_position: Initial network weights as a 2-d array of shape [number of chains, number of weights]
        :param test_model: The model used on the test data. Default=None
        :param batch_size: Batch size used for stochastic sampling methods. Default=None
        :param burn_in: Number of burn-in samples. Default=0
        :param step_sizes: Step size or a list of step sizes. Default=.0001
        :param step_probabilities: Probabilities to choose a step from step_sizes, must sum to 1. Default=1
        """

        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.test_model = test_model

        self.initial_position = np.asarray(initial_position, dtype=np.float32)
        self.position_shape = self.initial_position.shape
        self.position_size = self.initial_position.shape[1]  # total number of parameters of one network

        # data and parameter shapes
        self.chains_num = self.initial_position.shape[0]  # number of chains to run in parallel
        self.batch_size = batch_size if batch_size is not None else self.train_size
        self.batch_x_shape = (self.batch_size, self.input_dim)
        self.batch_y_shape = (self.batch_size, self.output_dim)

        # common parameters
        self.step_sizes = np.atleast_1d(np.asarray(step_sizes, dtype=np.float32))
        self.step_probabilities = np.atleast_1d(np.asarray(step_probabilities, dtype=np.float32))
        self.burn_in = burn_in
        self.step_multiplier = np.ones(shape=(self.chains_num,), dtype=np.float32)

        # monitor acceptance rate for reporting
        self.avg_acceptance_rate = np.ones(shape=(self.chains_num,), dtype=np.float32)
        self.avg_acceptance_rate_lambda = 0.99
        self._has_burned_in = False

    def __repr__(self):
        s = super().__repr__()
        s += f'Chains num: {self.chains_num}\n'
        s += f'Batch size: {self.batch_size}\n'
        s += f'Position size: {self.position_size}\n'
        s += f'Precisions: noise = {self.noise_precision}, weights = {self.weights_precision}\n'
        s += f'Resample precision: noise = {self.resample_noise_precision}, '
        s += f'weights = {self.resample_weights_precision}\n'
        s += f'Burn in: {self.burn_in}\n'
        s += f'Seek step sizes: {self.seek_step_sizes}\n'
        s += f'Anneal step sizes: {self.anneal_step_sizes}\n'
        s += f'Fade in velocities: {self.fade_in_velocities}\n'
        s += 'Step sizes: {}\n'.format(np.array_str(self.step_sizes).replace('\n', ''))
        s += 'Step probabilities: {}\n'.format(np.array_str(self.step_probabilities).replace('\n', ''))
        return s

    def _construct(self, **kwargs):
        """ Constructs computational graph for the model. """
        # feeds
        self._feed_dict = {}  # all values fed to TF
        self._create_feeds()

        # fetches
        self._fetch_dict = {}  # everything to be fetched from TF session
        self._debug = None

        # updated position + acceptance result, will be overridden by transition step
        self._updated_position_value = np.array(self.initial_position, dtype=np.float32)
        self._updated_position = self._position

        self._accepted_value = np.ones(shape=(self.chains_num,), dtype=np.float32)
        self._accepted = tf.ones(shape=(self.chains_num,), dtype=np.float32)

        self._construct_transition_step()
        self._construct_fetches()

        self._fetch_dict['_updated_position_value'] = self._updated_position
        self._fetch_dict['_accepted_value'] = self._accepted

        self._debug_value = None
        if self._debug is not None:
            self._fetch_dict['_debug_value'] = self._debug

    def _create_feeds(self):
        """ Creates TF placeholders for positions, training sets and common parameters. """
        # "*_value" fields contain the corresponding local values updated at every sample draw and fed to placeholders
        # position
        self._position_value = np.array(self.initial_position, dtype=np.float32)
        self._position = tf.placeholder(tf.float32, shape=self.position_shape, name='position')
        self._feed_dict[self._position] = lambda: self._position_value

        # current training batch
        self._batch_train_x_value = None
        self._batch_train_x = tf.placeholder(tf.float32, shape=self.batch_x_shape, name='train_x')
        self._feed_dict[self._batch_train_x] = lambda: self._batch_train_x_value

        self._batch_train_y_value = None
        self._batch_train_y = tf.placeholder(tf.float32, shape=self.batch_y_shape, name='train_y')
        self._feed_dict[self._batch_train_y] = lambda: self._batch_train_y_value

        self._current_step_size_value = None
        self._current_step_size = tf.placeholder(tf.float32, shape=(self.chains_num,), name='step_size')
        self._feed_dict[self._current_step_size] = lambda: self._current_step_size_value

        # precisions
        self._noise_precision_value = self.noise_precision
        self._noise_precision = tf.placeholder(tf.float32, shape=(), name='noise_precision')
        self._feed_dict[self._noise_precision] = lambda: self._noise_precision_value

        self._weights_precision_value = self.weights_precision
        self._weights_precision = tf.placeholder(tf.float32, shape=(), name='weights_precision')
        self._feed_dict[self._weights_precision] = lambda: self._weights_precision_value

        # other
        self._burn_in_ratio = tf.placeholder(tf.float32, shape=(), name='burn_in_ratio')
        self._feed_dict[self._burn_in_ratio] = lambda: self._get_burn_in_ratio(skip=.0, cut=.9)

    def _construct_fetches(self):
        """ Constructs fetches for target loss and average model weights. """
        # target loss and EMA
        self._target_loss_value = np.zeros(shape=(self.chains_num,), dtype=np.float32)
        self._target_loss = self.loss_fn(self._updated_position, self._batch_train_x, self._batch_train_y)
        self._fetch_dict['_target_loss_value'] = self._target_loss

        self._target_loss_ema = np.zeros(shape=(self.chains_num,), dtype=np.float32)

        # weight norm and EMA
        self._weight_norm_value = np.zeros(shape=(self.chains_num,), dtype=np.float32)
        self._weight_norm = self._weight_norm_fn(self._updated_position)
        self._fetch_dict['_weight_norm_value'] = self._weight_norm

        self._weight_norm_ema = np.zeros(shape=(self.chains_num,), dtype=np.float32)

    def _log_likelihood(self, position):
        """ Log-likelihood component. """
        batch_adjustment = (self.train_size / self.batch_size)
        return self._noise_precision * batch_adjustment * self.loss_fn(position, self._batch_train_x,
                                                                       self._batch_train_y)

    def _d_log_likelihood(self, position):
        """ Gradient of the log-likelihood component. """
        dL = tf.gradients(tf.reduce_sum(self._log_likelihood(position)), position)[0]
        return tf.clip_by_value(dL, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE)

    def _weight_norm_fn(self, position):
        """ Log-prior component. """
        return tf.reduce_sum(tf.square(position), reduction_indices=[1])

    def _log_prior(self, position):
        """ Log-prior component. """
        return self._weights_precision * self._weight_norm_fn(position)

    def _d_log_prior(self, position):
        """ Gradient of the log-likelihood component. """
        dW = tf.gradients(tf.reduce_sum(self._log_prior(position)), position)[0]
        return tf.clip_by_value(dW, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE)  ## TODO: no need to?

    def _energy_fn(self, position):
        """ Energy function (E = logP(data|params) + logP(params)). """
        return self._log_likelihood(position) + self._log_prior(position)

    def _d_energy_fn(self, position):
        """ Gradient of the energy function. """
        return self._d_log_likelihood(position) + self._d_log_prior(position)

    # Override in all subclasses
    def _construct_transition_step(self):
        """ Constructs computational graph for MCMC transition step. """
        pass

    def _simulate(self, session):
        """ Simulates MCMC to draw a sample. """
        # sample batch train data
        self._sample_batch()
        self._sample_step_size()

        # construct feed dictionary
        feed_dict = {k: v() for k, v in self._feed_dict.items()}

        # run the simulation
        update_dict = session.run(self._fetch_dict, feed_dict=feed_dict)

        # update with fetched values
        # TODO: should apply only position for now
        for k, v in update_dict.items():
            setattr(self, k, v)

        if self._debug is not None:
            logging.info(self._debug_value)

        # check whether to discard the sample to prevent possible future instability
        weight_deviation = self._updated_position_value.max() - self._updated_position_value.min()

        if not (weight_deviation < 10 ** 9):
            logging.info(f'Sample discarded to prevent instability: {weight_deviation:.2f}')
            return None

        # accept new position
        self._position_value = self._updated_position_value
        self._complete_simulation()

        self.avg_acceptance_rate = self.avg_acceptance_rate_lambda * self.avg_acceptance_rate + \
                                   (1. - self.avg_acceptance_rate_lambda) * self._accepted_value

        # resample precisions
        weight = .9
        self._target_loss_ema = weight * self._target_loss_ema + (1. - weight) * self._target_loss_value
        self._weight_norm_ema = weight * self._weight_norm_ema + (1. - weight) * self._weight_norm_value

        self._resample_prior_params()

        if not self._has_burned_in and self._burned_in():
            self._has_burned_in = True
            logging.info(f'Burned in. Samples = {self.sample_number}, step size = {self._current_step_size_value}.')

        return self._position_value

    def _sample_step_size(self):
        """ Selects step size (1 per chain) for the current simulation. """
        step_size = np.random.choice(self.step_sizes, size=self.chains_num, p=self.step_probabilities)
        step_size = self._adjust_step_size(step_size)

        # apply step size seek during burn in
        if self.seek_step_sizes and not self._burned_in():
            lower, upper = .90, .99
            change = max(min(10. / self.burn_in, .0001), .01)
            change *= (1 - self._get_burn_in_ratio(.35))
            inc, dec = 1. + change, 1. - change

            acr = self.avg_acceptance_rate
            self.step_multiplier *= (acr < lower).astype(np.float32) * dec + (acr >= lower).astype(np.float32)
            self.step_multiplier *= (acr > upper) * inc + (acr <= upper).astype(np.float32)

        step_size *= self.step_multiplier

        if self._burned_in() and self.anneal_step_sizes:
            t = self.sample_number - self.burn_in
            gamma = .51
            base = .01 * self.burn_in
            multiplier = base ** gamma / ((base + t) ** gamma)

            step_size *= multiplier

        self._current_step_size_value = step_size

    # Override in subclasses to adjust the scale
    def _adjust_step_size(self, step_size):
        """ Adjusts step_size. """
        return step_size

    # Override in subclasses to update them using fetched values
    def _complete_simulation(self):
        """ Updates class values with fetched values. """
        pass

    def _sample_batch(self):
        """ Samples training points for the current batch. """
        indices = np.random.choice(self.train_size, self.batch_size, replace=False)
        self._batch_train_x_value = self.train_x[indices, :]
        self._batch_train_y_value = self.train_y[indices, :]

    def _resample_prior_params(self):
        """ Resamples parameters for the prior distributions. """
        weight = .01 * self._get_burn_in_ratio(.5)
        if weight == 0:
            return

        # noise
        if self.resample_noise_precision:
            precision = self._sample_noise_precision()
            self._noise_precision_value = weight * precision + (1 - weight) * self._noise_precision_value

        # weights
        if self.resample_weights_precision:
            precision = self._sample_weights_precision()
            self._weights_precision_value = weight * precision + (1 - weight) * self._weights_precision_value

    def _sample_noise_precision(self):
        prior_observations = .1 * self.batch_size
        shape = prior_observations + self.batch_size / 2
        rate = prior_observations / self._noise_precision_value + np.mean(self._target_loss_ema) / 2
        scale = 1. / rate

        sample = np.clip(np.random.gamma(shape, scale), 10., 1000.)

        return sample

    def _sample_weights_precision(self):
        prior_observations = .1 * self.position_size
        shape = prior_observations + self.position_size / 2
        rate = prior_observations / self._weights_precision_value + np.mean(self._weight_norm_ema) / 2

        scale = 1. / rate
        sample = np.clip(np.random.gamma(shape, scale), .1, 10.)
        return sample

    def _burned_in(self):
        """ Whether burn in completed. """
        return self.sample_number >= self.burn_in

    def _get_burn_in_ratio(self, skip=.0, cut=.0):
        """ Burn in phase progress. """
        burn_in = self.burn_in * (1. - cut)

        if self.sample_number >= burn_in:
            return 1.

        skip *= self.burn_in
        sample_number = self.sample_number - skip

        if sample_number <= 0:
            return 0.

        base = burn_in - skip
        ratio = sample_number / base
        ratio = 3. * ratio ** 2 - 2. * ratio ** 3  # smooth both ends in a sine-shaped manner

        return ratio

    def _transpose_mul(self, a, b):
        """ Shortcut for multiplication with a transposed matrix. """
        return tf.transpose(tf.mul(tf.transpose(a), b))

    def _sample_posterior(self, session=None, return_stats=False, **kwargs):
        """ Returns a new sample obtained via simulation. """
        stats = None
        sample = self._simulate(session)

        if return_stats:
            stats = [self._collect_stats(i) for i in range(self.chains_num)]

        return sample, stats

    def _sample_predictive(self, session=None, return_stats=False, is_discarded=False, **kwargs):
        """ Returns a new sample obtained via simulation. """
        posterior_sample = None

        for i in range(self.draw_retries_num):
            posterior_sample, _ = self._sample_posterior(session=session, return_stats=False, **kwargs)
            if posterior_sample is not None:
                break

        if posterior_sample is None:
            return None, None

        if is_discarded:
            return self.test_x, None

        model, parameters = self.test_model

        collected_samples = list()
        collected_stats = list()

        for i in range(posterior_sample.shape[0]):
            model_params = np.reshape(posterior_sample[i], (1, posterior_sample[i].shape[0]))
            sample = session.run(model, feed_dict={parameters: model_params})

            stats = None
            if sample is not None and return_stats:
                stats = self._collect_stats(i)

            collected_samples.append(sample)
            collected_stats.append(stats)

        return collected_samples, collected_stats

    def _collect_stats(self, chain):
        stats = SampleStats(time=self._running_time(),
                            loss=self._report_loss(chain),
                            norm=self._weight_norm_value[chain] / self.position_size,
                            rate=self.avg_acceptance_rate[chain],
                            step=self._current_step_size_value[chain],
                            noise_var=self._report_noise_variance(),
                            weights_var=self._report_weights_variance())

        return stats

    def _report_loss(self, chain):
        target_loss = self._target_loss_value[chain]

        if self.output_dim > 1:
            return target_loss

        return (self.train_y_std[0] ** 2) * target_loss / self.batch_size

    def _report_noise_variance(self):
        var = 1. / self._noise_precision_value

        if self.output_dim > 1:
            return var

        return (self.train_y_std[0] ** 2) * var

    def _report_weights_variance(self):
        if self._weights_precision_value == 0:
            return 1.

        return 1. / self._weights_precision_value

    @classmethod
    def model_from_position(cls, layer_descriptions, position_tensor, input_tensor, use_softmax=False):
        """ Creates TF model from the specified position and description. """
        offset = 0
        model = input_tensor

        for i in range(1, len(layer_descriptions)):
            previous_layer = layer_descriptions[i - 1]
            current_layer = layer_descriptions[i]

            previous_layer_size = previous_layer[0]
            current_layer_size = current_layer[0]

            weights_size = previous_layer_size * current_layer_size
            biases_size = current_layer_size

            weights = tf.slice(position_tensor, [0, offset], [1, weights_size])
            weights = tf.reshape(weights, shape=[previous_layer_size, current_layer_size])
            offset += weights_size

            biases = tf.slice(position_tensor, [0, offset], [1, biases_size])
            biases = tf.reshape(biases, shape=[1, biases_size])
            offset += biases_size

            model = tf.matmul(model, weights) + biases

            if i != len(layer_descriptions) - 1:
                model = tf.nn.relu(model)
            elif use_softmax and layer_descriptions[-1][0] > 1:
                model = tf.nn.softmax(model)

        return model

    @classmethod
    def model_chain_from_position(cls, chains_num, layer_descriptions, position_tensor, input_tensor):
        """ Creates multiple-chain model from the specified position and description. """
        positions = tf.split(0, chains_num, position_tensor)

        m = []
        for i in range(chains_num):
            m.append(cls.model_from_position(layer_descriptions, positions[i], input_tensor))

        models = tf.pack(m)

        return models

    @classmethod
    def create_random_position(cls, chains_num, layers_description):
        """ Creates randomly initialised position for the specified model. """
        pos_size = cls.get_model_parameters_size(layers_description)
        # position = np.random.randn(chains_num, pos_size).astype(np.float32)
        position = sp.stats.truncnorm.rvs(-1, 1, size=(chains_num, pos_size)).astype(np.float32)
        position = np.random.randn(chains_num, pos_size).astype(np.float32)
        return position

    @classmethod
    def get_mse_loss(cls, chains_num, layers_description):
        """ Returns MSE loss for the given model. """

        def mse_loss(position, tx, ty):
            model = cls.model_chain_from_position(chains_num, layers_description, position, tx)
            loss = tf.reduce_sum((ty - model) ** 2, reduction_indices=[1, 2])
            return loss

        return mse_loss

    @classmethod
    def get_ce_loss(cls, chains_num, layers_description):
        def ce_loss(position, tx, ty):
            """ Returns cross-entropy loss for the given model. """
            model = cls.model_chain_from_position(chains_num, layers_description, position, tx)
            model = tf.reshape(model, shape=(chains_num * model.get_shape()[1].value, -1))
            ty = tf.tile(ty, [chains_num, 1])
            loss = tf.nn.softmax_cross_entropy_with_logits(model, ty)
            # l = tf.nn.sparse_softmax_cross_entropy_with_logits(m, ty)
            loss = tf.reshape(loss, shape=(chains_num, -1))
            loss = tf.reduce_sum(loss, reduction_indices=[1])
            return loss

        return ce_loss
