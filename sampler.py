""" This module implements the base functionality for all NN samplers. """
import logging
from collections import namedtuple
from time import perf_counter

import numpy as np

SampleStats = namedtuple('SampleStats', 'time loss norm rate step noise_var weights_var')
SampleStats.__new__.__defaults__ = (0,) * len(SampleStats._fields)


class Sampler:
    """
    Base class for models that draw samples from the posterior/predictive of NNs.

    Methods to work with a sampler:
        construct
        fit
        sample_posterior / sample_predictive

    Methods to override in implementations:
        _construct
        _fit
        _sample_posterior / _sample_predictive

    When implementing a model, non-core parameters can be added to _properties dictionary instead of being explicitly
    declared in the constructor.
    """

    def __new__(cls, **kwargs):
        """ Creates a new Sampler object. """
        sampler = super().__new__(cls)
        sampler.sampler_type = None

        # dictionary for non-core parameters
        sampler._properties = {
            'draw_retries_num': 100,
            'normalise_data': True,
        }

        return sampler

    def __init__(self, train_x=None, train_y=None, test_x=None, test_y=None, **kwargs):
        """
        Creates a new Sampler object.
        :param train_x: Training set - x points
        :param train_y: Training set - y labels
        :param test_x: Test set - x point
        :param test_y: Test set - y labels
        :param draw_retries_num: The number of trials to draw a sample. Default=100
        """
        # apply non-core parameters dictionary
        for prop, default in self._properties.items():
            setattr(self, prop, kwargs.get(prop, default))

        self.train_x = np.asarray(train_x, dtype=np.float32)
        self.train_y = np.asarray(train_y, dtype=np.float32)
        self.train_size = self.train_x.shape[0]  # number of points

        test_x = test_x if test_x is not None else train_x
        test_y = test_x if test_x is not None else train_y
        self.test_x = np.asarray(test_x, dtype=np.float32)
        self.test_y = np.asarray(test_y, dtype=np.float32)
        self.test_size = self.test_x.shape[0]  # number of points

        self.input_dim = self.train_x.shape[1]
        self.output_dim = self.train_y.shape[1]
        self._normalise_data()

        self.sample_number = 0  # current sample id
        self._pc = None  # performance counter

    def __repr__(self):
        s = f"Sampler: {self.sampler_type}\n"
        s += f"Train size: {self.train_size}\n"
        s += f"Test size: {self.test_size}\n"
        s += f"Normalise: {self.normalise_data}\n"
        s += f"X: mean={self.train_x_mean}, std={self.train_x_std}\n"
        s += f"Y: mean={self.train_y_mean}, std={self.train_y_std}\n"
        return s

    def _normalise_data(self):
        self.train_x_mean = np.zeros(self.input_dim)
        self.train_x_std = np.ones(self.input_dim)

        self.train_y_mean = np.zeros(self.output_dim)
        self.train_y_std = np.ones(self.output_dim)

        if self.normalise_data:
            self.train_x_mean = np.mean(self.train_x, axis=0)
            self.train_x_std = np.std(self.train_x, axis=0)
            self.train_x_std[self.train_x_std == 0] = 1.

            self.train_x = (self.train_x - np.full(self.train_x.shape, self.train_x_mean, dtype=np.float32)) / \
                           np.full(self.train_x.shape, self.train_x_std, dtype=np.float32)

            self.test_x = (self.test_x - np.full(self.test_x.shape, self.train_x_mean, dtype=np.float32)) / \
                          np.full(self.test_x.shape, self.train_x_std, dtype=np.float32)

            self.train_y_mean = np.mean(self.train_y, axis=0)
            self.train_y_std = np.std(self.train_y, axis=0)

            if self.train_y_std == 0:
                self.train_y_std[self.train_y_std == 0] = 1.

            self.train_y = (self.train_y - self.train_y_mean) / self.train_y_std

    def _denormalise_sample(self, sample):
        return self.train_y_mean + sample * self.train_y_std

    def _running_time(self):
        """ Time elapsed since the computation started. """
        return perf_counter() - self._pc

    # Override in subclasses explicitly creating computation graphs
    def _construct(self, **kwargs):
        """ Constructs computation graph for the model. """
        pass

    # Override in subclasses that fit the model prior to sampling
    def _fit(self, **kwargs):
        """ Fits the model prior to sampling. """
        pass

    # Override in subclasses supporting draws from the posterior
    def _sample_posterior(self, session=None, return_stats=False, **kwargs):
        """ Draws a new sample from the model posterior. """
        pass

    # Override in subclasses supporting draws from the predictive
    def _sample_predictive(self, session=None, test_x=None, return_stats=False, is_discarded=False, **kwargs):
        """ Draws a new sample from the model posterior predictive. """
        pass

    def construct(self, **kwargs):
        """ Constructs computation graph for the model. """
        logging.info("Constructing computation graph...")
        self._construct(**kwargs)
        logging.info("{!r}".format(self))

    def fit(self, **kwargs):
        """ Fits the model prior to drawing samples. """
        if self._pc is None:
            self._pc = perf_counter()

        self._fit(**kwargs)

    def sample_posterior(self, session=None, return_stats=False, **kwargs):
        """
        Returns a new sample from the posterior distribution of the parameters.
        :param return_stats: Whether to return sampling process statistics
        :return: the generated sample
        """

        # make a number of tries to draw a sample
        for i in range(self.draw_retries_num):
            sample, stats = self._sample_posterior(session=session, return_stats=return_stats, **kwargs)
            if sample is not None:
                break

        if sample is not None:
            self.sample_number += 1
        else:
            logging.warning("Impossible to draw a sample with the specified parameters.")

        if return_stats:
            return sample, stats

        return sample

    def sample_predictive(self, session=None, test_x=None, return_stats=False, is_discarded=False, **kwargs):
        """
        Returns a new sample from the posterior distribution of the parameters.
        :param test_x: Test set
        :param return_stats: Whether to return sampling process statistics
        :return: the generated sample and accompanying statistics
        """

        test_x = test_x if test_x is not None else self.test_x

        # make a number of tries to draw a sample
        for i in range(self.draw_retries_num):
            sample, stats = self._sample_predictive(session=session, test_x=test_x, return_stats=return_stats,
                                                    is_discarded=is_discarded, **kwargs)
            if sample is not None:
                break

        if sample is not None:
            self.sample_number += 1

            # denormalise
            sample = [self._denormalise_sample(s) for s in sample]
        else:
            logging.warning("Impossible to draw a sample with the specified parameters.")

        if return_stats:
            return sample, stats

        return sample

    @classmethod
    def get_model_parameters_size(cls, layers_description):
        """ Calculates the size of the position. """
        pos_size = 0
        for i in range(1, len(layers_description)):
            pos_size += layers_description[i][0] * layers_description[i - 1][0] + layers_description[i][0]

        return pos_size
