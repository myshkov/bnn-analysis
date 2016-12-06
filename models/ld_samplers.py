"""
This module implements LD based samplers for NNs.
"""

import numpy as np
import tensorflow as tf
from models.mcmc_sampler import MCMC_sampler


class LDSampler(MCMC_sampler):
    """
    Langevin Dynamics (LD) sampler for NNs.
    """

    def __init__(self, **kwargs):
        """ Creates a new LDSampler object. """
        # set parameters restricted by LD
        kwargs['seek_step_sizes'] = False

        super().__init__(**kwargs)
        self.sampler_type = "LD"

    def _construct_transition_step(self):
        """ Constructs LD general transition step. """
        initial_position = self._position

        # gradients of likelihood and prior
        dL = self._d_log_likelihood(initial_position)
        dW = self._d_log_prior(initial_position)

        # compute gradient and noise steps
        gradient_step, noise_step = self._compute_ld_step_components(dL, dW)

        # update position (take the step)
        if self.fade_in_velocities:
            noise_step *= self._burn_in_ratio

        self._updated_position = initial_position - gradient_step + noise_step

    def _compute_ld_step_components(self, dL, dW):
        """ Computes gradient and noise components. """
        # generate noise
        noise_stddev = tf.sqrt(2. * self._current_step_size)
        noise = tf.random_normal(self.position_shape)
        noise_step = self._transpose_mul(noise, noise_stddev)

        # calculate gradient step
        gradient = dL + dW
        gradient_step = self._transpose_mul(gradient, self._current_step_size)

        self._debug_update = gradient

        return gradient_step, noise_step

    def _adjust_step_size(self, step_size):
        """ Brings scale to that of HMC samplers. """
        step_size = step_size ** 2 / 2
        return step_size


class SGLDSampler(LDSampler):
    """
    Stochastic Gradient Langevin Dynamics (SGLD) sampler for NNs.
    """

    def __init__(self, **kwargs):
        """ Creates a new SGLDSampler object. """
        super().__init__(**kwargs)
        self.sampler_type = "SGLD"

        # effectively LD since the likelihood part of the target function is already adjusted for the batch size


class pSGLDSampler(LDSampler):
    """
    Preconditioned Stochastic Gradient Langevin Dynamics (pSGLD) sampler for NNs.
    """

    def __init__(self, preconditioned_alpha=0.99, preconditioned_lambda=1e-05, adjust_steps=False, **kwargs):
        super().__init__(**kwargs)
        self.sampler_type = "pSGLD"

        self.preconditioned_alpha = preconditioned_alpha
        self.preconditioned_lambda = preconditioned_lambda
        self.adjust_steps = adjust_steps

    def __repr__(self):
        s = super().__repr__()
        s += f"Preconditioned alpha: {self.preconditioned_alpha}\n"
        s += f"Preconditioned lambda: {self.preconditioned_lambda}\n"
        return s

    def _create_feeds(self):
        """ Adds preconditioned values to the graph. """
        super()._create_feeds()

        self._preconditioned_v_value = np.zeros(shape=self.position_shape, dtype=np.float32)
        self._preconditioned_v = tf.placeholder(tf.float32, shape=self.position_shape)
        self._feed_dict[self._preconditioned_v] = lambda: self._preconditioned_v_value

    # def _adjust_step_size(self, step_size):
    #     """ Adjust step_size for total curvature correction effect. """
    #     step_size = super()._adjust_step_size(step_size)
    #
    #     if not self.adjust_steps:
    #         return step_size
    #
    #     # p_avg_effect = 1. / (self.preconditioned_lambda + self._preconditioned_v_value ** .5)
    #     # p_avg_effect = p_avg_effect.min() ** .5
    #     # p_avg_effect = 1. / p_avg_effect
    #     # p_avg_effect = max(1., min(p_avg_effect, 100.))
    #     # step_size *= p_avg_effect
    #
    #     return step_size

    # def _update_values(self, update_dict):
    #     """ Updates preconditioned values. """
    #     self._preconditioned_v_value = update_dict[self._updated_preconditioned_v]

    def _compute_ld_step_components(self, dL, dW):
        """ Computes gradient and noise components. """
        # update average gradient
        avg_gradient = dL / self.train_size
        avg_gradient **= 2

        # is_increase = tf.to_float(avg_gradient > self._preconditioned_v)
        # is_increase = 0.00 * is_increase + (1. - is_increase)
        # is_increase = 1.  # TODO: currently disabled
        #
        # preconditioned_v = is_increase * self.preconditioned_alpha * self._preconditioned_v + \
        #                    (1. - is_increase * self.preconditioned_alpha) * avg_gradient

        preconditioned_v = self.preconditioned_alpha * self._preconditioned_v + \
                           (1. - self.preconditioned_alpha) * avg_gradient

        self._fetch_dict['_preconditioned_v_value'] = preconditioned_v

        # calculate preconditioning matrix
        g = 1. / (self.preconditioned_lambda + tf.sqrt(preconditioned_v))

        # generate step noise
        noise_stddev = tf.sqrt(2. * self._transpose_mul(g, self._current_step_size))
        noise_step = noise_stddev * tf.random_normal(self.position_shape)

        # calculate gradient step
        gradient = dL + dW
        gradient_step = self._transpose_mul(g * gradient, self._current_step_size)

        return gradient_step, noise_step
