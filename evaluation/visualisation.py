import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import evaluation.metrics as metrics
import utils

sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1.5)
common_palette = [(.05, .1, .9, .5), (.05, .7, .7, .7), 'm', 'gray']
current_palette = list(sns.color_palette())
current_palette.extend(list(sns.color_palette('pastel')))


def plot_predictive_baseline(env, samples, stddev_mult=3., title_name=None):
    # single var regression only
    samples = samples.squeeze()

    train_x, train_y = env.get_train_x(), env.get_train_y()
    test_x, test_y = env.get_test_x(), env.get_test_y()

    pad_width = test_x.shape[0] - train_x.shape[0]
    train_x_padded = np.pad(train_x[:, 0], (0, pad_width), 'constant', constant_values=np.nan)
    train_y_padded = np.pad(train_y[:, 0], (0, pad_width), 'constant', constant_values=np.nan)

    data = samples

    df = pd.DataFrame.from_dict({
        'time': test_x[:, 0],
        'true_y': test_y[:, 0],
        'train_x': train_x_padded,
        'train_y': train_y_padded,
        'mean': data.mean(axis=0),
        'std': stddev_mult * data.std(axis=0),
        # 'stdn': 2. * (data.std(axis=0) + .5 ** .5),
    }).reset_index()

    g = sns.FacetGrid(df, size=9, aspect=1.8)

    g.map(plt.errorbar, 'time', 'mean', 'std', color=(0.7, 0.1, 0.1, 0.09))
    g.map(plt.plot, 'time', 'mean', color='b', lw=1)
    g.map(plt.plot, 'time', 'true_y', color='r', lw=1)
    g.map(plt.scatter, 'train_x', 'train_y', color='g', s=20)

    ax = g.ax
    ax.set_title('Posterior Predictive Distribution' + (': ' + title_name) if title_name is not None else '')
    ax.set(xlabel='X', ylabel='Y')
    ax.set_xlim(env.view_xrange[0], env.view_xrange[1])
    ax.set_ylim(env.view_yrange[0], env.view_yrange[1])

    legend = ['Prediction mean', 'True f(x)', 'Training data', 'StdDev']
    plt.legend(legend)

    # ax.annotate("MSE: {:.03f}".format(0), xy=(0.1, 0.9), xytext=(0.1, 0.9), xycoords='figure fraction',
    #             textcoords='figure fraction')

    name = utils.DATA_DIR.replace('/', '-')
    plt.tight_layout(pad=0.6)
    utils.save_fig('predictive-distribution-' + name)


def plot_predictive_comparison(env, baseline_samples, target_samples, stddev_mult=3., target_metrics=None,
                               title_name=None):
    # single var regression only
    baseline_samples = baseline_samples.squeeze()
    target_samples = target_samples.squeeze()

    train_x, train_y = env.get_train_x(), env.get_train_y()
    test_x, test_y = env.get_test_x(), env.get_test_y()

    pad_width = test_x.shape[0] - train_x.shape[0]
    train_x_padded = np.pad(train_x[:, 0], (0, pad_width), 'constant', constant_values=np.nan)
    train_y_padded = np.pad(train_y[:, 0], (0, pad_width), 'constant', constant_values=np.nan)

    df = pd.DataFrame.from_dict({
        'time': test_x[:, 0],
        'true_y': test_y[:, 0],
        'train_x': train_x_padded,
        'train_y': train_y_padded,
        'mean': target_samples.mean(axis=0),
        'std': stddev_mult * target_samples.std(axis=0),
        'base_mean': baseline_samples.mean(axis=0),
        'base_std': stddev_mult * baseline_samples.std(axis=0),
    }).reset_index()

    g = sns.FacetGrid(df, size=9, aspect=1.8)

    g.map(plt.errorbar, 'time', 'base_mean', 'base_std', color=(0.7, 0.1, 0.1, 0.09))
    g.map(plt.errorbar, 'time', 'mean', 'std', color=(0.1, 0.1, 0.7, 0.09))
    g.map(plt.plot, 'time', 'mean', color='b', lw=1)
    g.map(plt.plot, 'time', 'true_y', color='r', lw=1)
    g.map(plt.scatter, 'train_x', 'train_y', color='g', s=20)

    ax = g.ax
    ax.set_title('Posterior Predictive Distribution' + (': ' + title_name) if title_name is not None else '')
    ax.set(xlabel='X', ylabel='Y')
    ax.set_xlim(env.view_xrange[0], env.view_xrange[1])
    ax.set_ylim(env.view_yrange[0], env.view_yrange[1])

    legend = ['Prediction mean', 'True f(x)', 'Training data', 'True StdDev', 'Predicted StdDev']
    plt.legend(legend)

    if target_metrics is not None:
        offset = 0
        for tm, tv in target_metrics.items():
            ax.annotate(f'{tm}: {tv:.02f}', xy=(0.08, 0.92 - offset), xytext=(0.08, 0.92 - offset),
                        xycoords='figure fraction', textcoords='figure fraction')
            offset += 0.04

    name = utils.DATA_DIR.replace('/', '-')
    plt.tight_layout(pad=0.6)
    utils.save_fig('predictive-distribution-' + name)


def plot_metrics(baseline_samples, samples_dict, times_dict, metric_names, resample_base=150,
                 resample_test=150, max_start=.45, points=50, window=5, title_name=None, max_time=10):
    baseline_samples = metrics.resample_to(baseline_samples, resample_base)
    metrics._reset_cache()

    plot_dict = dict()

    for name, samples in samples_dict.items():
        logging.info(f'Testing: {name}')
        times = times_dict[name]

        if max_time is not None:
            ind_max = np.argmax(times > max_time * 60)

            if ind_max > 0:
                times = times[:ind_max]
                samples = samples[:ind_max]

        if samples.shape[0] == 0:
            logging.info(f'No samples for {name}.')
            continue

        run_metrics = None
        run_times = list()

        # run window
        for run in range(points):
            # determine current window of samples
            offset_end = int(samples.shape[0] * (run + 1) / points)
            max_start_offset = int(max_start * samples.shape[0])
            offset_start = min(max_start_offset, max(0, int(samples.shape[0] * (run + 1 - window) / points)))
            target_samples = metrics.resample_to(samples[offset_start:offset_end], resample_test)

            values = metrics.calculate_all_metrics(baseline_samples, target_samples, metric_names)
            avg_metrics = values.mean(axis=1)

            if run_metrics is not None:
                weight = .3
                ema = weight * avg_metrics + (1 - weight) * run_metrics[-1]
                run_metrics = np.vstack((run_metrics, ema))
            else:
                run_metrics = avg_metrics

            run_times.append(times[offset_end - 1])

            # logging.info(avg_metrics)

        for i in range(len(metric_names)):
            plot_dict[(name, i, 'time')] = np.asarray(run_times) / 60  # minutes
            plot_dict[(name, i, 'value')] = run_metrics[:, i]

    df = pd.DataFrame.from_dict(plot_dict).reset_index()
    g = sns.FacetGrid(df, size=8, aspect=1.6)

    legend = list()
    for name in samples_dict.keys():
        for i in range(len(metric_names)):
            if (name, i, 'time') in plot_dict:
                g.map(plt.plot, (name, i, 'time'), (name, i, 'value'), lw=4, color=current_palette[len(legend)])
                if len(metric_names) > 1:
                    legend.append(name + ': ' + metric_names[i])
                else:
                    legend.append(name)

    g.ax.set_ylim(0, 1.05)

    plt.xlabel('Time (minutes)')
    plt.ylabel('Metric Value')
    plt.title(title_name if title_name is not None else 'Metrics')
    plt.legend(legend)

    file_name = 'metrics'
    file_name += '--' + '-'.join((k.lower() for k in samples_dict.keys()))
    file_name += '--' + '-'.join((k.lower() for k in metric_names))
    plt.tight_layout(pad=0.6)
    utils.save_fig(file_name)


def plot_hist(baseline_samples, target_samples, true_x, true_y):
    baseline_samples = baseline_samples.squeeze()
    target_samples = target_samples.squeeze()

    bmin, bmax = baseline_samples.min(), baseline_samples.max()

    ax = sns.kdeplot(baseline_samples, shade=True, color=(0.6, 0.1, 0.1, 0.2))
    ax = sns.kdeplot(target_samples, shade=True, color=(0.1, 0.1, 0.6, 0.2))
    ax.set_xlim(bmin, bmax)

    y0, y1 = ax.get_ylim()

    plt.plot([true_y, true_y], [0, y1 - (y1 - y0) * 0.01], linewidth=1, color='r')
    plt.title('Predictive' + (f' at {true_x:.2f}' if true_x is not None else ''))

    fig = plt.gcf()
    fig.set_size_inches(9, 9)
    # plt.tight_layout()  # pad=0.4, w_pad=0.5, h_pad=1.0)

    name = utils.DATA_DIR.replace('/', '-')
    # plt.tight_layout(pad=0.6)
    utils.save_fig('predictive-at-point-' + name)
