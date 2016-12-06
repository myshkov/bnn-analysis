"""
Energy Efficiency
"""

from env import Env


class EnergyEnv(Env):
    """ Test environment for experiments with Energy Efficiency data set. """

    def __init__(self):
        super().__init__()
        # setup defaults
        self.env_name = 'en'
        self.layers_description = [[8, 0.0], [50, 0.0], [1, 0.0]]
        self.batch_size = 64
        self.n_splits = 4
