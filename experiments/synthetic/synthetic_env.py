"""
Tests for f(x) = (1 + x) sin(10 tanh(x)).
"""

import numpy as np
import scipy as sp

from env import Env


class SyntheticEnv(Env):
    """ Test environment for experiments with synthetic data: f(x) = (1 + x) sin(10 tanh(x)). """

    def __init__(self):
        super().__init__()

        # fixed test parameters
        self.data_size = 50
        self.data_interval_left = -4
        self.data_interval_right = 4
        self.test_data_size = 100
        self.test_data_interval_left = -6
        self.test_data_interval_right = 6
        self.view_xrange = [-6, 6]
        self.view_yrange = [-4, 3]
        self.n_splits = 1

        # setup defaults
        self.env_name = 'synthetic'
        self.layers_description = [[1, 0.0], [100, 0.0], [100, 0.0], [1, 0.0]]
        self.batch_size = 10

    def true_f(self, x):
        return 1. * (1. + x) * np.sin(10. * np.tanh(x))

    def create_training_test_sets(self):
        # training set
        scale = self.data_interval_right - self.data_interval_left
        train_x = sp.stats.truncnorm.rvs(-2, 2, scale=0.25 * scale, size=self.data_size).astype(np.float32)
        train_x = np.sort(train_x)
        train_y = self.true_f(train_x) + 0.2 * np.random.randn(self.data_size)

        self.train_x = [train_x.reshape((train_x.shape[0], 1))]
        self.train_y = [train_y.reshape((train_y.shape[0], 1))]

        # test set
        # scale = self.test_data_interval_right - self.test_data_interval_left
        # test_x = sp.stats.truncnorm.rvs(-2, 2, scale=0.25 * scale, size=self.test_data_size).astype(np.float32)
        # test_x = np.sort(test_x)
        # test_y = self.true_f(test_x)

        self.test_x = np.arange(self.view_xrange[0], self.view_xrange[1], 0.01, dtype=np.float32)
        self.test_y = self.true_f(self.test_x)

        self.test_x = [self.test_x.reshape((self.test_x.shape[0], 1))]
        self.test_y = [self.test_y.reshape((self.test_y.shape[0], 1))]
