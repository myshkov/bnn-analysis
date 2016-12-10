"""
Tests for f(x) = x^3
"""

import numpy as np

from env import Env


class X3Env(Env):
    """ Test environment for experiments with synthetic data: f(x) = x^3. """

    def __init__(self):
        super().__init__()

        # fixed test parameters
        self.data_size = 20
        self.data_interval_left = -4
        self.data_interval_right = 4
        self.test_data_size = 50
        self.test_data_interval_left = -6
        self.test_data_interval_right = 6
        self.view_xrange = [-6, 6]
        self.view_yrange = [-100, 100]
        self.n_splits = 1

        # setup defaults
        self.env_name = 'x3'
        self.layers_description = [[1, 0.0], [100, 0.0], [1, 0.0]]
        self.batch_size = 20

    def true_f(self, x):
        return 1. * x ** 3

    def create_training_test_sets(self):
        # training set
        train_x = np.random.uniform(self.data_interval_left, self.data_interval_right, size=self.data_size)
        train_x = np.sort(train_x)
        train_y = self.true_f(train_x) + 3. * np.random.randn(self.data_size)

        self.train_x = [train_x.reshape((train_x.shape[0], 1))]
        self.train_y = [train_y.reshape((train_y.shape[0], 1))]

        # test set for visualisation
        self.test_x = np.arange(self.view_xrange[0], self.view_xrange[1], 0.01, dtype=np.float32)
        self.test_x = np.reshape(self.test_x, (self.test_x.shape[0], 1))
        self.test_y = self.true_f(self.test_x)
        self.test_y = np.reshape(self.test_y, (self.test_y.shape[0], 1))

        self.test_x = [self.test_x]
        self.test_y = [self.test_y]
