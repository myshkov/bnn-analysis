import logging
from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils
from sampler import Sampler
from models.mcmc_sampler import MCMC_sampler
from models.dropout_sampler import DropoutSampler
from models.bbb_sampler import BBBSampler
from models.pbp_sampler import PBPSampler
import evaluation.visualisation as vis


class Experiment:
    """
    Configures, tests and evaluates models (Sampler) for a particular environment (Env).
    Contains default setups for common types of samplers for easier configuration.
    """

    def __init__(self):
        pass

    def _setup_sampler_defaults(self, sampler_params):
        pass

    def setup_env_defaults(self, env):
        env.create_training_test_sets()

    def configure_env_mcmc(self, env, sampler_class=None, sampler_params=None, loss='mse'):
        env.model_parameters_size = Sampler.get_model_parameters_size(env.layers_description)

        loss = MCMC_sampler.get_mse_loss if loss is 'mse' else MCMC_sampler.get_ce_loss

        def sampler_factory():
            params = env.get_default_sampler_params()
            self._setup_sampler_defaults(params)

            params['loss_fn'] = loss(env.chains_num, env.layers_description)
            params['initial_position'] = MCMC_sampler.create_random_position(env.chains_num,
                                                                             env.layers_description)

            params['burn_in'] = int(.45 * env.n_chunks * env.samples_per_chunk())

            if sampler_params is not None:
                params.update(sampler_params)

            sampler = sampler_class(**params)

            pos_size = env.model_parameters_size
            model_parameters = tf.placeholder(dtype=tf.float32, shape=[1, pos_size])
            model = MCMC_sampler.model_from_position(env.layers_description, model_parameters, sampler.test_x)
            sampler.test_model = [model, model_parameters]

            sampler.construct()
            sampler.fit()

            return sampler

        env.sampler_factory = sampler_factory

    def configure_env_dropout(self, env, sampler_params=None, dropout=0.01, tau=0.15, length_scale=1e-2):
        def sampler_factory():
            params = env.get_default_sampler_params()
            params['n_epochs'] = 50

            wreg = length_scale ** 2 * (1 - dropout) / (2. * env.get_train_x().shape[0] * tau)
            model = DropoutSampler.model_from_description(env.layers_description, wreg, dropout)
            logging.info(f'Reg: {wreg}')

            if sampler_params is not None:
                params.update(sampler_params)

            sampler = DropoutSampler(model=model, **params)
            sampler.construct()
            return sampler

        env.sampler_factory = sampler_factory

    def configure_env_bbb(self, env, sampler_params=None, noise_std=0.01, weigts_std=1., n_epochs=5):
        def sampler_factory():
            params = env.get_default_sampler_params()
            params['step_size'] = .1

            if sampler_params is not None:
                params.update(sampler_params)

            params['n_epochs'] = n_epochs

            model = BBBSampler.model_from_description(env.layers_description, noise_std, weigts_std, env.batch_size,
                                                      env.get_train_x().shape[0])

            sampler = BBBSampler(model=model, **params)
            sampler.construct()
            return sampler

        env.sampler_factory = sampler_factory

    def configure_env_pbp(self, env, sampler_params=None, n_epochs=50):
        def sampler_factory():
            params = env.get_default_sampler_params()
            params['model_desc'] = env.layers_description
            params['n_epochs'] = n_epochs

            if sampler_params is not None:
                params.update(sampler_params)

            sampler = PBPSampler(**params)
            sampler.construct()
            return sampler

        env.sampler_factory = sampler_factory

    def run_queue(self, queue, skip_completed=True, cpu=False):
        if cpu:
            with tf.device('/cpu:0'):
                self._run_queue(queue, skip_completed=skip_completed)
        else:
            self._run_queue(queue, skip_completed=skip_completed)

    def is_complete(self, name):
        return utils.get_latest_data_subdir(self.__to_pattern(name)) is not None

    def plot_predictive_baseline(self, name=None, split=0, discard=.5):
        env, samples = self.__load_env_baseline(name, split, discard_left=discard)
        vis.plot_predictive_baseline(env, samples, title_name=name)

    def plot_predictive_comparison(self, baseline, target, split=0, discard_left=0., discard_right=0.,
                                   target_metrics=None):
        # baseline
        env, baseline_samples = self.__load_env_baseline(baseline, split=split, discard_left=0.5)

        # target
        target_samples, target_times = self.__load_target(env, target, split, discard_left=discard_left,
                                                          discard_right=discard_right)

        vis.plot_predictive_comparison(env, baseline_samples, target_samples, target_metrics=target_metrics,
                                       title_name=target)

    def plot_predictive_point(self, baseline, target, split=0, discard_left=0., discard_right=0., point_index=0):
        # baseline
        env, baseline_samples = self.__load_env_baseline(baseline, split=split, discard_left=0.5)

        # target
        target_samples, target_times = self.__load_target(env, target, split, discard_left=discard_left,
                                                          discard_right=discard_right)

        true_x = env.get_test_x()[point_index][0]
        true_y = env.get_test_y()[point_index][0]

        vis.plot_hist(baseline_samples[:, point_index], target_samples[:, point_index], true_x, true_y)

    def compute_metrics(self, baseline, target, split=0, discard_left=0., discard_right=0., metric_names=None):
        # baseline
        env, baseline_samples = self.__load_env_baseline(baseline, split=split, discard_left=0.5)

        # target
        target_samples, target_times = self.__load_target(env, target, split, discard_left=discard_left,
                                                          discard_right=discard_right)
        return env.compute_metrics(baseline_samples, target_samples, metric_names=metric_names)

    def plot_metrics(self, baseline, target, metric_names, split=0):
        # baseline
        env, baseline_samples = self.__load_env_baseline(baseline, split=split, discard_left=.5)

        # target
        target_samples, target_times = self.__load_target(env, target, split)

        samples_dict = OrderedDict()
        samples_dict[target] = target_samples

        times_dict = OrderedDict()
        times_dict[target] = target_times

        vis.plot_metrics(baseline_samples, samples_dict, times_dict, metric_names)

    def plot_multiple_metrics(self, baseline, targets, metric_names, split=0, max_time=60, title_name=None):
        # baseline
        env, baseline_samples = self.__load_env_baseline(baseline, split=split, discard_left=.5)

        # targets
        samples_dict = OrderedDict()
        times_dict = OrderedDict()

        for t in targets:
            samples_dict[t], times_dict[t] = self.__load_target(env, name=t, split=split)

        vis.plot_metrics(baseline_samples, samples_dict, times_dict, metric_names, max_time=max_time,
                         title_name=title_name)

    def report_metrics_table(self, queue, discard_left=.75):
        for target in queue.keys():
            metrics = []

            for split in range(4):
                target_metrics = self.compute_metrics('HMC', target, discard_left=discard_left, discard_right=.0,
                                                      metric_names=['RMSE', 'KS', 'KL', 'Precision', 'Recall', 'F1'])

                metrics.append([v for v in target_metrics.values()])

            print(self.__report_avg_metrics(target, metrics))

    def __report_metrics(self, target, scores):
        str = target
        for name, score in scores.items():
            str += f' & {score:.2f}'

        str += ' \\\\'
        return str

    def __report_avg_metrics(self, target, scores):
        scores = np.asarray(scores)
        mean = scores.mean(axis=0)
        std = scores.std(axis=0)

        str = target
        for m, s in zip(mean, std):
            str += f' & {m:.2f} $\\pm$ {s:.3f}'

        str += ' \\\\'
        return str

    def _run_queue(self, queue, skip_completed):
        for name, run_fn in queue.items():
            if not skip_completed or not self.is_complete(name):
                run_fn()

    def __to_pattern(self, name):
        return '-' + name.lower() + '-'

    def __load_env_baseline(self, name=None, split=0, discard_left=.5, discard_right=0.):
        utils.set_latest_data_subdir(pattern=self.__to_pattern(name))
        env = utils.deserialize('env')

        env.current_split = split
        samples = env.load_samples(split=split, discard_left=discard_left, discard_right=discard_right)

        return env, samples

    def __load_target(self, env, name=None, split=0, discard_left=0., discard_right=0.):
        utils.set_latest_data_subdir(pattern=self.__to_pattern(name))

        samples = env.load_samples(split=split, discard_left=discard_left, discard_right=discard_right)
        times = env.load_times(split=split, discard_left=discard_left, discard_right=discard_right)

        return samples, times
