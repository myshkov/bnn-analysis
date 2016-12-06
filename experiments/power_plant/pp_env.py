"""
Power Plant
"""

from env import Env


class PowerPlantEnv(Env):
    """ Test environment for experiments with Power Plant data set. """

    def __init__(self):
        super().__init__()

        # setup defaults
        self.env_name = 'pp'
        self.layers_description = [[4, 0.0], [100, 0.0], [100, 0.0], [1, 0.0]]
        self.n_splits = 4
        self.batch_size = 64
