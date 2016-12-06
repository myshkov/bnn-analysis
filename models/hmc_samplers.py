"""
This module implements Hamiltonian based MCMC samplers for NNs.
"""

import numpy as np
import tensorflow as tf
from models.mcmc_sampler import MCMC_sampler


class HMCSampler(MCMC_sampler):
    """
    Hamiltonian Monte-Carlo (HMC) sampler for NNs.
    """

    def __new__(cls, **kwargs):
        """ Creates a new HMCSampler object. """
        sampler = super().__new__(cls)

        # specific to HMC non-core parameters
        sampler._properties['persistent_momentum'] = 0.
        sampler._properties['mh_correction'] = True
        sampler._properties['seek_step_sizes'] = True

        return sampler

    def __init__(self, hmc_steps=20, **kwargs):
        """
        Creates a new HMC_sampler object.

        Specific parameters
        :param hmc_steps: The number of leapfrog steps. Default=20
        :param persistent_momentum: Persistent momentum alpha. Default=0
        :param mh_correction: Whether to perform MH correction step. Default=True
        """
        super().__init__(**kwargs)
        self.sampler_type = "HMC"
        self.hmc_steps = hmc_steps

    def __repr__(self):
        s = super().__repr__()
        s += f"HMC steps: {self.hmc_steps}\n"
        s += f"MH correction: {self.mh_correction}\n"
        s += f"Persistent momentum: {self.persistent_momentum}\n"
        return s

    def _create_feeds(self):
        super()._create_feeds()

        """ Adds momentum to the graph. """
        self._momentum_value = np.zeros(shape=self.position_shape, dtype=np.float32)
        self._momentum = tf.placeholder(tf.float32, shape=self.position_shape)
        self._feed_dict[self._momentum] = lambda: self._momentum_value

    def _construct_transition_step(self):
        """ HMC transition step. """
        # sample random velocity
        initial_position = self._position
        initial_velocity = tf.random_normal(tf.shape(self._position))

        if self.fade_in_velocities:
            initial_velocity *= self._burn_in_ratio

        # apply persistent momentum
        if self.persistent_momentum > 0:
            initial_velocity = self.persistent_momentum * self._momentum + \
                               ((1. - self.persistent_momentum ** 2) ** 0.5) * initial_velocity

        # run the simulation
        final_position, final_velocity = self._simulate_hmc_dynamics(
            initial_position=initial_position,
            initial_velocity=initial_velocity,
        )

        # apply MH acceptance/rejection scheme
        if self.mh_correction:
            final_position, final_velocity = self._add_mh_correction(initial_position, initial_velocity,
                                                                     final_position, final_velocity)

        # add to fetch dict
        self._updated_momentum = final_velocity
        self._fetch_dict['_momentum_value'] = self._updated_momentum

        self._updated_position = final_position

    def _add_mh_correction(self, initial_position, initial_velocity, final_position, final_velocity):
        """ Applies MH accept/reject correction. """
        initial_energy = self._hamiltonian(initial_position, initial_velocity)
        final_energy = self._hamiltonian(final_position, final_velocity)
        accepted = self._metropolis_hastings_accept(initial_energy, final_energy)
        accepted = tf.to_float(accepted)

        # add acceptance to fetched values
        self._accepted = accepted

        if self.seek_step_sizes or self.fade_in_velocities:
            burned_in = tf.to_float(self._burn_in_ratio == 1)
            accepted = accepted * burned_in + tf.ones(shape=tf.shape(accepted)) * (1 - burned_in)

        # apply MH decision
        final_position = self._transpose_mul(final_position, accepted) + \
                         self._transpose_mul(initial_position, tf.ones(shape=tf.shape(accepted)) - accepted)

        final_velocity = self._transpose_mul(final_velocity, accepted) + \
                         self._transpose_mul(-initial_velocity, tf.ones(shape=tf.shape(accepted)) - accepted)

        return final_position, final_velocity

    def _metropolis_hastings_accept(self, initial_energy, final_energy):
        """ Returns MH accept/reject decision. """
        accept = tf.exp(initial_energy - final_energy) >= tf.random_uniform(shape=(self.chains_num,))
        return tf.to_float(accept)

    def _hamiltonian(self, position, velocity):
        """ Calculates the Hamiltonian. """
        # assuming mass is 1
        kinetic_energy = 0.5 * tf.reduce_sum(velocity ** 2, reduction_indices=[1])

        return self._energy_fn(position) + kinetic_energy

    def _leapfrog_step(self, position, velocity, velocity_step_multiplier=1.):
        """ Makes a single leapfrog step. """
        d_energy = self._d_energy_fn(position)
        velocity = velocity - self._transpose_mul(d_energy, velocity_step_multiplier * self._current_step_size)
        position = position + self._transpose_mul(velocity, self._current_step_size)

        return position, velocity

    def _simulate_hmc_dynamics(self, initial_position, initial_velocity):
        """ Simulates leapfrog steps. """
        # velocity half step + position full step
        position, velocity = self._leapfrog_step(initial_position, initial_velocity, .5)

        for i in range(0, self.hmc_steps - 1):
            position, velocity = self._leapfrog_step(position, velocity)

        _, velocity = self._leapfrog_step(position, velocity, .5)

        return position, velocity


class SGHMCSampler(HMCSampler):
    """
    Stochastic-Gradient Hamiltonian Monte-Carlo (SG-HMC) sampler for NNs.
    """

    def __new__(cls, **kwargs):
        """ Creates a new SGHMC_sampler object. """
        sampler = super().__new__(cls)

        # specific to SG_HMC parameters
        sampler._properties['friction'] = 0.
        return sampler

    def __init__(self, **kwargs):
        """
        Creates a new SGHMC_sampler object.

        Specific parameters
        :param friction: The friction term. Default=0.
        """
        # set parameters restricted by SG-HMC
        kwargs['mh_correction'] = False
        kwargs['persistent_momentum'] = 0.
        kwargs['seek_step_sizes'] = False

        super().__init__(**kwargs)
        self.sampler_type = "SG-HMC"

    def __repr__(self):
        s = super().__repr__()
        s += f"Friction: {self.friction}\n"
        return s

    def _leapfrog_step(self, position, velocity, velocity_step_multiplier=1.):
        """ Makes a single leapfrog step with friction. """
        d_energy = self._d_energy_fn(position)

        friction = self.friction
        deceleration = -friction * self._transpose_mul(velocity, self._current_step_size)

        velocity -= self._transpose_mul(d_energy, velocity_step_multiplier * self._current_step_size)
        velocity += deceleration

        # B_hat = 0, C = friction
        noise = tf.random_normal(tf.shape(velocity))
        stddevs = (2 * friction * self._current_step_size) ** 0.5
        noise = self._transpose_mul(noise, stddevs)

        velocity += noise

        position = position + self._transpose_mul(velocity, self._current_step_size)

        return position, velocity
