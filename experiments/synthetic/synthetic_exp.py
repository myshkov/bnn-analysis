"""
Tests for f(x) = (1 + x) sin(10 tanh(x)).
"""

from collections import OrderedDict

from experiments.experiment import Experiment
from models.hmc_samplers import HMCSampler, SGHMCSampler
from models.ld_samplers import LDSampler, SGLDSampler, pSGLDSampler
from experiments.synthetic.synthetic_env import SyntheticEnv


class SyntheticExp(Experiment):
    """ Tests for f(x) = (1 + x) sin(10 tanh(x)). """

    def __init__(self):
        super().__init__()

    def _setup_sampler_defaults(self, sampler_params):
        sampler_params['noise_precision'] = 25.
        sampler_params['weights_precision'] = 1.

    def run_baseline_hmc(self):
        env = SyntheticEnv()
        self.setup_env_defaults(env)

        env.model_name = 'hmc'
        env.test_case_name = 'baseline'

        env.chains_num = 1
        env.n_samples = 50
        env.thinning = 1

        sampler_params = dict()
        sampler_params['step_sizes'] = .0003  # 0.00039599
        sampler_params['hmc_steps'] = 50
        sampler_params['persistent_momentum'] = .5

        sampler_params['batch_size'] = None
        sampler_params['seek_step_sizes'] = True
        env.setup_data_dir()
        self.configure_env_mcmc(env, HMCSampler, sampler_params)
        env.run()

    def run_sgld(self):
        env = SyntheticEnv()
        self.setup_env_defaults(env)

        env.model_name = 'sgld'

        env.n_samples = 100
        env.thinning = 99

        sampler_params = dict()
        sampler_params['step_sizes'] = .0005  # obtained from HMC
        sampler_params['anneal_step_sizes'] = True

        env.setup_data_dir()
        self.configure_env_mcmc(env, SGLDSampler, sampler_params)
        env.run()

    def run_psgld(self):
        env = SyntheticEnv()
        self.setup_env_defaults(env)

        env.model_name = 'psgld'

        env.chains_num = 3
        env.n_samples = 100
        env.thinning = 49

        sampler_params = dict()
        sampler_params['step_sizes'] = .001
        sampler_params['anneal_step_sizes'] = True
        sampler_params['preconditioned_alpha'] = .999
        sampler_params['preconditioned_lambda'] = 1.
        sampler_params['noise_precision'] = 10.

        env.setup_data_dir()
        self.configure_env_mcmc(env, pSGLDSampler, sampler_params)
        env.run()

    def run_sghmc(self):
        env = SyntheticEnv()
        self.setup_env_defaults(env)

        env.model_name = 'sghmc'

        env.n_samples = 100
        env.thinning = 9

        sampler_params = dict()
        sampler_params['step_sizes'] = .0005
        sampler_params['anneal_step_sizes'] = True
        sampler_params['hmc_steps'] = 10
        sampler_params['friction'] = 1.

        env.setup_data_dir()
        self.configure_env_mcmc(env, SGHMCSampler, sampler_params)
        env.run()

    def run_dropout(self):
        env = SyntheticEnv()
        self.setup_env_defaults(env)

        env.model_name = 'dropout'

        env.n_samples = 100

        sampler_params = dict()
        sampler_params['n_epochs'] = 100

        dropout = 0.02
        tau = 0.15

        env.setup_data_dir()
        self.configure_env_dropout(env, sampler_params=sampler_params, dropout=dropout, tau=tau)
        env.run()

    def run_bbb(self):
        env = SyntheticEnv()
        self.setup_env_defaults(env)

        env.model_name = 'bbb'
        n_epochs = 50
        env.n_samples = 20

        env.setup_data_dir()
        self.configure_env_bbb(env, n_epochs=n_epochs)
        env.run()

    def run_pbp(self):
        env = SyntheticEnv()
        self.setup_env_defaults(env)

        env.model_name = 'pbp'

        env.n_samples = 100

        sampler_params = dict()
        sampler_params['normalise_data'] = True

        env.setup_data_dir()
        self.configure_env_pbp(env, sampler_params=sampler_params, n_epochs=20)
        env.run()


def main():
    experiment = SyntheticExp()

    queue = OrderedDict()

    queue['HMC'] = experiment.run_baseline_hmc
    queue['SGLD'] = experiment.run_sgld
    queue['pSGLD'] = experiment.run_psgld
    queue['SGHMC'] = experiment.run_sghmc
    queue['BBB'] = experiment.run_bbb
    # queue["PBP"] = experiment.run_pbp
    queue['Dropout'] = experiment.run_dropout

    experiment.run_queue(queue, cpu=False)

    del queue['HMC']

    for target in queue.keys():
        target_metrics = experiment.compute_metrics('HMC', target, discard_left=.45, discard_right=.0,
                                                    metric_names=['RMSE', 'KS', 'KL', 'Precision', 'Recall', 'F1'])

        print(experiment.__report_metrics(target, target_metrics))

        experiment.plot_predictive_comparison('HMC', target, target_metrics=target_metrics, discard_left=.45,
                                              discard_right=.0)

    max_time = 60
    experiment.plot_multiple_metrics('HMC', queue.keys(), ['Precision'], max_time=max_time, title_name='Precision')
    experiment.plot_multiple_metrics('HMC', queue.keys(), ['Recall'], max_time=max_time, title_name='Recall')
    experiment.plot_multiple_metrics('HMC', queue.keys(), ['KS'], max_time=max_time, title_name='KS distance')

    # experiment.plot_multiple_metrics("HMC", queue.keys(), ["KL"], max_time=max_time, title_name="KL divergence")
    # experiment.plot_multiple_metrics("HMC", queue.keys(), ["F1"], max_time=max_time, title_name="F1 score")
    # experiment.plot_multiple_metrics("HMC", queue.keys(), ["IoU"], max_time=max_time)


if __name__ == '__main__':
    main()
