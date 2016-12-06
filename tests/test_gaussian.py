"""
Tests MCMC samplers on a multivariate Gaussian
"""
import logging
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib.patches import Ellipse

import env
from models.hmc_samplers import HMCSampler, SGHMCSampler
from models.ld_samplers import SGLDSampler, pSGLDSampler

seed = 2305
np.random.seed(seed)
tf.set_random_seed(seed)

# configure seaborn
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)
common_palette = [(.05, .1, .9, .5), (.05, .7, .7, .7), 'm', 'gray']
current_palette = sns.color_palette()


def run_samplers(plot_samples=False, plot_errors=True):
    mu = np.array([1., 1.], dtype=np.float32)
    cov = np.asarray([[1., .5], [.5, .8]], dtype=np.float32)
    prec = np.linalg.inv(cov)

    def target_loss(x, t_x, t_y):
        likelihood = 0.5 * tf.matmul((x - mu), prec) * (x - mu)
        return tf.reduce_sum(likelihood, reduction_indices=1)

    samplers = OrderedDict()

    samples_num = 1000
    burn_in = 0

    params = dict(
        loss_fn=target_loss,
        initial_position=np.random.randn(1, 2).astype(np.float32),
        train_x=np.zeros((1, 1), dtype=np.float32),
        train_y=np.zeros((1, 1), dtype=np.float32),
        noise_precision=1.,
        weights_precision=0.,
        step_sizes=0.3,
        burn_in=burn_in,
        seek_step_sizes=False,
        anneal_step_sizes=False)

    samplers['HMC'] = HMCSampler(hmc_steps=5, persistent_momentum=.3, **params)
    samplers['SGHMC'] = SGHMCSampler(hmc_steps=5, friction=1., **params)
    samplers['SGLD'] = SGLDSampler(**params)
    samplers['pSGLD'] = pSGLDSampler(**params)

    for sampler in samplers.values():
        sampler.construct()
        sampler.fit()

    samples = OrderedDict()
    with tf.Session() as session:
        tf.initialize_all_variables().run()

        for name, sampler in samplers.items():
            logging.info("Sampling {}...".format(name))
            collected_samples = list()
            for idx in range(samples_num):
                sample, stats = sampler.sample_posterior(session=session, return_stats=True)
                collected_samples.append(sample[0])
                if idx % 500 == 0:
                    logging.info(f"Collected = {idx}, rate = {stats[0].rate:.2f}, step = {stats[0].step:.8f}")

            samples[name] = np.asarray(collected_samples)

    samples['NumPy'] = np.random.multivariate_normal(mu, cov, size=samples_num)

    points = 1000
    window = 900

    # plot HMC types
    if plot_samples:
        show_num = 200
        plot_dict = OrderedDict()

        for sampler in samplers.keys():
            plot_dict[sampler + "-x"] = samples[sampler][:show_num, 0]
            plot_dict[sampler + "-y"] = samples[sampler][:show_num, 1]

        df = pd.DataFrame.from_dict(plot_dict).reset_index()

        g = sns.FacetGrid(df, size=6)

        legend = []
        for idx, sampler in enumerate(samplers.keys()):
            g.map(plt.scatter, sampler + "-x", sampler + "-y", color=current_palette[idx])
            legend.append(sampler)

        plt.legend(legend)

        # add contours
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)

        ax = g.ax
        ax.set_title("Correlated Gaussian: HMC samplers")
        ax.set(xlabel="X", ylabel="Y")
        ax.set_xlim(-2, 4)
        ax.set_ylim(-2, 4)

        e = Ellipse(xy=(mu[0], mu[1]), width=lambda_[0] * 2 * 2, height=lambda_[1] * 2 * 2,
                    angle=np.rad2deg(np.arccos(v[0, 0])))

        e.set_facecolor('none')
        e.set_ec('r')
        e.set_linewidth(1)
        g.ax.add_artist(e)
        plt.show()

    # plot errors
    if plot_errors:
        true_params = np.concatenate((mu, cov.ravel()))

        colors = ['g', 'b', 'orange', 'violet', 'gray', 'r']
        c = 0
        legend = list()
        for name, collected_samples in samples.items():
            logging.info("Plotting {}...".format(name))
            errors = []
            for run in range(points):
                # current window of samples
                offset_end = int(samples_num * (run + 1) / points)
                offset_start = max(0, int(samples_num * (run + 1 - window) / points))

                idx = collected_samples[offset_start:offset_end]
                emp_mu = idx.mean(axis=0)
                emp_cov = np.cov(idx.T)
                sample_params = np.concatenate((emp_mu, emp_cov.ravel()))
                diff = ((true_params - sample_params) ** 2).mean() ** .5
                errors.append(diff)

            errors = np.asarray(errors)
            sns.tsplot(errors, color=colors[c])
            legend.append(name)
            c += 1

        plt.title('Correlated Gaussian')
        plt.xlabel('Sample Number x10')
        plt.ylabel('Error')
        plt.legend(legend)
        plt.show()


def main():
    with tf.device('/cpu:0'):
        run_samplers()


if __name__ == "__main__":
    main()
