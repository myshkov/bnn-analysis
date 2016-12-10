"""
This module implements Bayes By Backprop -based sampler for NNs.
http://jmlr.org/proceedings/papers/v37/blundell15.pdf
"""
import numpy as np

from keras.models import Sequential
from keras.layers.core import Activation
from keras import backend as K
from keras.engine.topology import Layer

from sampler import Sampler, SampleStats


class BBBSampler(Sampler):
    """
    BBB sampler for NNs.
    """

    def __init__(self, model=None, batch_size=None, n_epochs=None, **kwargs):
        """
        Creates a new BBBSampler object.
        """
        super().__init__(**kwargs)
        self.sampler_type = 'BBB'

        self.model = model
        self.batch_size = batch_size if batch_size is not None else self.train_set_size
        self.n_epochs = n_epochs

    def __repr__(self):
        s = super().__repr__()
        return s

    def _fit(self, n_epochs=None, verbose=0, **kwargs):
        """ Fits the model before sampling. """
        n_epochs = n_epochs if n_epochs is not None else self.n_epochs
        self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size, nb_epoch=n_epochs,
                       verbose=verbose)

    def _sample_predictive(self, test_x=None, return_stats=False, **kwargs):
        """ Draws a new sample from the model. """
        sample = self.model.predict(test_x, batch_size=self.batch_size)

        stats = None
        if return_stats:
            stats = SampleStats(time=self._running_time())

        return [sample], [stats]

    @classmethod
    def model_from_description(cls, layers, noise_std, weights_std, batch_size, train_size):
        """ Creates a BBB model from the specified parameters. """
        n_batches = int(train_size / batch_size)
        step = .01

        class BBBLayer(Layer):
            def __init__(self, output_dim, **kwargs):
                self.output_dim = output_dim
                super().__init__(**kwargs)

            def build(self, input_shape):
                input_dim = input_shape[1]
                shape = [input_dim, self.output_dim]

                eps_std = step

                # weights
                self.eps_w = K.random_normal([input_shape[0]] + shape, std=eps_std)

                self.mu_w = K.variable(np.random.normal(0., 10. * step, size=shape), name='mu_w')
                self.rho_w = K.variable(np.random.normal(0., 10. * step, size=shape), name='rho_w')
                self.W = self.mu_w + self.eps_w * K.log(1.0 + K.exp(self.rho_w))

                self.eps_b = K.random_normal([self.output_dim], std=eps_std)

                self.mu_b = K.variable(np.random.normal(0., 10. * step, size=[self.output_dim]), name='mu_b')
                self.rho_b = K.variable(np.random.normal(0., 10. * step, size=[self.output_dim]), name='rho_b')
                self.b = self.mu_b + self.eps_b * K.log(1.0 + K.exp(self.rho_b))

                self.trainable_weights = [self.mu_w, self.rho_w, self.mu_b, self.rho_b]

            def call(self, x, mask=None):
                return K.squeeze(K.batch_dot(K.expand_dims(x, dim=1), self.W), axis=1) + self.b

            def get_output_shape_for(self, input_shape):
                return (input_shape[0], self.output_dim)

        def log_gaussian(x, mean, std):
            return -K.log(std) - (x - mean) ** 2 / (2. * std ** 2)

        def sigma_from_rho(rho):
            return K.log(1. + K.exp(rho)) / step

        def variational_objective(model, noise_std, weights_std, batch_size, nb_batches):
            def loss(y, fx):

                log_pw = K.variable(0.)
                log_qw = K.variable(0.)

                for layer in model.layers:
                    if type(layer) is BBBLayer:
                        log_pw += K.sum(log_gaussian(layer.W, 0., weights_std))
                        log_pw += K.sum(log_gaussian(layer.b, 0., weights_std))

                        log_qw += K.sum(log_gaussian(layer.W, layer.mu_w, sigma_from_rho(layer.rho_w)))
                        log_qw += K.sum(log_gaussian(layer.b, layer.mu_b, sigma_from_rho(layer.rho_b)))

                log_likelihood = K.sum(log_gaussian(y, fx, noise_std))

                return K.sum((log_qw - log_pw) / nb_batches - log_likelihood) / batch_size

            return loss

        model = Sequential()

        in_shape = [batch_size, layers[0][0]]

        # input
        model.add(BBBLayer(layers[1][0], batch_input_shape=in_shape))
        model.add(Activation('relu'))

        # hidden layers
        for l in range(2, len(layers) - 1):
            model.add(BBBLayer(layers[l - 1][0]))
            model.add(Activation('relu'))

        # output layer
        model.add(BBBLayer(1))

        loss = variational_objective(model, noise_std, weights_std, batch_size, n_batches)
        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

        return model
