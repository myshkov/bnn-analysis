import numpy as np
import scipy.stats as sc

MAX_KL = 1.
MAX_GRID_POINTS = 100

_cached_p_pdf = dict()


def kl_divergence(p_samples, q_samples):
    # estimate densities
    # p_samples = np.nan_to_num(p_samples)
    # q_samples = np.nan_to_num(q_samples)

    if isinstance(p_samples, tuple):
        idx, p_samples = p_samples

        if idx not in _cached_p_pdf:
            _cached_p_pdf[idx] = sc.gaussian_kde(p_samples)

        p_pdf = _cached_p_pdf[idx]
    else:
        p_pdf = sc.gaussian_kde(p_samples)

    q_pdf = sc.gaussian_kde(q_samples)

    # joint support
    left = min(min(p_samples), min(q_samples))
    right = max(max(p_samples), max(q_samples))

    p_samples_num = p_samples.shape[0]
    q_samples_num = q_samples.shape[0]

    # quantise
    lin = np.linspace(left, right, min(max(p_samples_num, q_samples_num), MAX_GRID_POINTS))
    p = p_pdf.pdf(lin)
    q = q_pdf.pdf(lin)

    # KL
    kl = min(sc.entropy(p, q), MAX_KL)

    return kl


def ks_distance(p_samples, q_samples):
    if isinstance(p_samples, tuple):
        idx, p_samples = p_samples

    return sc.ks_2samp(p_samples, q_samples)[0]


def _compute_ci(samples, alpha):
    samples = np.sort(samples)
    samples_num = samples.shape[0]

    alpha = .5 * (1 - alpha)
    left = samples[int(alpha * samples_num)]
    right = samples[int((1 - alpha) * samples_num)]
    # TODO: np.partition(a, 4)[e]

    return left, right


def ci_metrics(p_samples, q_samples, alpha=.95):
    if isinstance(p_samples, tuple):
        idx, p_samples = p_samples

    # compute CIs
    p_left, p_right = _compute_ci(p_samples, alpha)
    q_left, q_right = _compute_ci(q_samples, alpha)

    precision = 0
    iou = 0
    recall = 0
    f1 = 0

    if (p_right > q_left) and (q_right > p_left):
        # intersection
        int_left = max(q_left, p_left)
        int_right = min(p_right, q_right)

        union_left = min(p_left, q_left)
        union_right = max(p_right, q_right)

        iou = (int_right - int_left) / (union_right - union_left)

        precision = (int_right - int_left) / (q_right - q_left)
        recall = (int_right - int_left) / (p_right - p_left)

        f1 = 2 * precision * recall / (precision + recall)

        # estimate densities
        # p_pdf = sc.gaussian_kde(p_samples)
        # q_pdf = sc.gaussian_kde(q_samples)
        #
        # precision = q_pdf.integrate_box(int_left, int_right) / alpha
        # recall = p_pdf.integrate_box(int_left, int_right) / alpha

    return iou, precision, recall, f1


def iou(p_samples, q_samples, alpha=.90):
    return ci_metrics(p_samples, q_samples, alpha)[0]


def precision(p_samples, q_samples, alpha=.90):
    return ci_metrics(p_samples, q_samples, alpha)[1]


def recall(p_samples, q_samples, alpha=.90):
    return ci_metrics(p_samples, q_samples, alpha)[2]


def f1(p_samples, q_samples, alpha=.90):
    return ci_metrics(p_samples, q_samples, alpha)[3]


METRICS_INDEX = {
    'KS': ks_distance,
    'KL': kl_divergence,
    'IoU': iou,
    'P': precision,
    'Precision': precision,
    'R': recall,
    'Recall': recall,
    'F1': f1,
}


def _reset_cache():
    _cached_p_pdf.clear()


def calculate_all_metrics(baseline, target, metric_names):
    results = None

    for name in metric_names:
        values = [METRICS_INDEX[name]((i, baseline[:, i]), target[:, i]) for i in range(baseline.shape[1])]
        values = np.asarray(values)

        results = np.vstack((results, values)) if results is not None else values

    if len(metric_names) == 1:
        results = np.expand_dims(results, 0)

    return results


def resample_to(samples, size, sort=False):
    # single var regression only
    samples = samples.squeeze()

    if samples.shape[0] > size:
        ind = np.random.choice(samples.shape[0], size=size, replace=False)
        samples = samples[ind, :]

    if sort:
        samples = np.sort(samples, axis=1)

    return samples


def moving_average(a, n=3):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array([(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))
    return result
