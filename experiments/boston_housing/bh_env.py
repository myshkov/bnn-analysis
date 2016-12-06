"""
Boston Housing
"""

from env import Env


class BostonHousingEnv(Env):
    """ Test environment for experiments with Boston Housing data set. """

    def __init__(self):
        super().__init__()

        # setup defaults
        self.env_name = 'bh'
        self.layers_description = [[13, 0.0], [50, 0.0], [1, 0.0]]
        self.n_splits = 4
        self.batch_size = 32
