""" Tests on f(x) = x^3. """

from collections import OrderedDict

from experiments.experiment import Experiment
from models.hmc_samplers import HMCSampler, SGHMCSampler
from models.ld_samplers import SGLDSampler, pSGLDSampler
from experiments.x3.x3_env import X3Env


class X3Exp(Experiment):
    """ Tests on f(x) = x^3. """

    def __init__(self):
        super().__init__()

    def run_baseline_hmc(self):
        env = X3Env()
        self.setup_env_defaults(env)
        env.batch_size = None

        env.model_name = 'hmc'
        env.test_case_name = 'baseline'

        sampler_params = dict()
        sampler_params['step_sizes'] = .0005
        sampler_params['hmc_steps'] = 10
        sampler_params['persistent_momentum'] = .5

        env.setup_data_dir()
        self.configure_env_mcmc(env, HMCSampler, sampler_params)
        env.run()

    def run_sgld(self):
        env = X3Env()
        self.setup_env_defaults(env)

        env.model_name = 'sgld'
        env.thinning = 4

        sampler_params = dict()
        sampler_params['step_sizes'] = .005

        env.setup_data_dir()
        self.configure_env_mcmc(env, SGLDSampler, sampler_params)
        env.run()

    def run_sghmc(self):
        env = X3Env()
        self.setup_env_defaults(env)

        env.model_name = 'sghmc'

        sampler_params = dict()
        sampler_params['step_sizes'] = .005
        sampler_params['hmc_steps'] = 10
        sampler_params['friction'] = 1.

        env.setup_data_dir()
        self.configure_env_mcmc(env, SGHMCSampler, sampler_params)
        env.run()

    def run_psgld(self):
        env = X3Env()
        self.setup_env_defaults(env)

        env.model_name = 'psgld'
        env.thinning = 4

        sampler_params = dict()
        sampler_params['step_sizes'] = .005
        sampler_params['preconditioned_lambda'] = .1

        env.setup_data_dir()
        self.configure_env_mcmc(env, pSGLDSampler, sampler_params)
        env.run()

    def run_dropout(self):
        env = X3Env()
        self.setup_env_defaults(env)
        env.model_name = 'dropout'

        sampler_params = dict()
        sampler_params['n_epochs'] = 5

        dropout = 0.01

        env.setup_data_dir()
        self.configure_env_dropout(env, sampler_params=sampler_params, dropout=dropout)
        env.run()

    def run_bbb(self):
        env = X3Env()
        self.setup_env_defaults(env)

        env.model_name = 'bbb'

        n_epochs = 15
        env.n_samples = 20

        env.setup_data_dir()
        self.configure_env_bbb(env, n_epochs=n_epochs)
        env.run()

    def run_pbp(self):
        env = X3Env()
        self.setup_env_defaults(env)

        env.model_name = 'pbp'

        env.setup_data_dir()
        self.configure_env_pbp(env, n_epochs=4)
        env.run()


def main():
    test_space = True
    experiment = X3Exp()

    queue = OrderedDict()
    queue['HMC'] = experiment.run_baseline_hmc
    queue['SGLD'] = experiment.run_sgld
    queue['SGHMC'] = experiment.run_sghmc
    queue['pSGLD'] = experiment.run_psgld
    queue['BBB'] = experiment.run_bbb
    # queue["PBP"] = experiment.run_pbp
    queue['Dropout'] = experiment.run_dropout

    experiment.run_queue(queue, cpu=True)

    def report(baseline, target):
        if test_space:
            experiment.plot_predictive_comparison(baseline, target, discard_left=.45)

    for name in queue.keys():
        if name == 'HMC':
            experiment.plot_predictive_baseline(name)
        else:
            report('HMC', name)


if __name__ == '__main__':
    main()
