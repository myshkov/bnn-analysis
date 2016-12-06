"""
BBB Sampler
Based on BBVI implementation in autograd.
"""

import logging
import autograd.numpy as np
import scipy as sp
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
import numpy as numpy
from autograd.optimizers import adam
from autograd import grad

from sampler import Sampler, SampleStats

np.random.seed(2305)


def black_box_variational_inference(logprob, D, num_samples=20):
    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, log_std = params[:D], params[D:]
        return mean, log_std

    def gaussian_entropy(log_std):
        return 0.5 * D * (1.0 + np.log(2 * np.pi)) + np.sum(log_std)

    rs = npr.RandomState(0)

    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std = unpack_params(params)
        samples = rs.randn(num_samples, D) * np.exp(log_std) + mean
        lower_bound = gaussian_entropy(log_std) + np.mean(logprob(samples, t))
        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params


class BBBSampler(Sampler):
    """
    BBB sampler for NNs.
    """

    def __init__(self, model=None, n_epochs=None, step_size=None, **kwargs):
        """
        Creates a new BBBSampler object.
        """
        super().__init__(**kwargs)
        self.sampler_type = "BBB"

        self._num_weights, self._predictions, self._logprob = model[0], model[1], model[2]
        self._n_epochs = n_epochs
        self._step_size = step_size
        self._samples = None
        self._stats = None
        self._current_sample = 0

    def __repr__(self):
        s = super().__repr__()
        return s

    def _fit(self, **kwargs):
        """ Fits the model prior to sampling. """
        log_posterior = lambda weights, t: self._logprob(weights, self.train_x, self.train_y)
        objective, gradient, unpack_params = black_box_variational_inference(log_posterior, self._num_weights)

        self._samples = list()
        self._stats = list()
        self._current_sample = 0

        def callback(params, t, g):
            if t % 500 == 0:
                logging.info(f"BBVI: iteration {t} lower bound = {-objective(params, t)}")

            rs = npr.RandomState(0)
            mean, log_std = unpack_params(params)
            # rs = npr.RandomState(0)
            sample_weights = rs.randn(1, self._num_weights) * np.exp(log_std) + mean
            outputs = self._predictions(sample_weights, self.test_x)
            # outputs = outputs.reshape((outputs.shape[0] * outputs.shape[1], -1))

            for i in range(outputs.shape[0]):
                self._samples.append(outputs[i])
                stats = SampleStats(time=self._running_time(), loss=0, norm=0, rate=0)
                self._stats.append(stats)

        rs = npr.RandomState(0)
        init_mean = rs.randn(self._num_weights)
        init_log_std = -5 * np.ones(self._num_weights)
        init_var_params = np.concatenate([init_mean, init_log_std])

        adam(gradient, init_var_params, step_size=self._step_size, num_iters=self._n_epochs, callback=callback)

    def _sample_predictive(self, test_x=None, return_stats=False, **kwargs):
        """ Draws a new sample from the model. """
        sample = None
        stats = None

        if self._current_sample < len(self._samples):
            sample = self._samples[self._current_sample]
            stats = self._stats[self._current_sample]
            self._current_sample += 1
        else:
            logging.info(f"No samples: current = {self._current_sample}, length = {len(self._samples)}")

        return [sample], [stats]

    @classmethod
    def model_from_description(cls, layers_description, noise_variance, batch_size, train_size):
        relu = lambda x: np.maximum(x, 0.)
        nonlinearity = relu
        wreg = .2

        layer_sizes = list(list(zip(*layers_description))[0])
        shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        num_weights = sum((m + 1) * n for m, n in shapes)

        def unpack_layers(weights):
            num_weight_sets = len(weights)
            for m, n in shapes:
                yield weights[:, :m * n].reshape((num_weight_sets, m, n)), \
                      weights[:, m * n:m * n + n].reshape((num_weight_sets, 1, n))
                weights = weights[:, (m + 1) * n:]

        def predictions(weights, inputs):
            inputs = np.expand_dims(inputs, 0)
            for W, b in unpack_layers(weights):
                outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
                inputs = nonlinearity(outputs)

            return outputs

        def logprob(weights, inputs, targets):
            log_prior = -wreg * np.sum(weights ** 2, axis=1)
            preds = predictions(weights, inputs)
            log_lik = -np.sum((preds - targets) ** 2, axis=1)[:, 0] / noise_variance
            return log_prior + log_lik

        return [num_weights, predictions, logprob]
