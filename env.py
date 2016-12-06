import logging
from time import perf_counter as pc
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import evaluation.metrics as metrics
import utils

# add console logger
logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")


class Env:
    """
    Defines the experiment environment: data set, splits, common parameters.
    Draws and stores the samples from the predictive distribution on the test set.
    """

    def __init__(self):
        """ Creates a new Env object. """
        # set seeds
        self.seed = 2305
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # test case
        self.env_name = None  # name of the environment
        self.model_name = None  # name of the model
        self.test_case_name = "test"  # name of the test
        self.baseline_test_case_name = None  # name of the test containing 'true' posterior
        self.data_dir = None

        # data
        self.input_dim = None  # number of feature
        self.output_dim = None
        self.data_size = None  # number of rows

        self.n_splits = 10
        self.current_split = 0
        self.train_x = list()
        self.train_y = list()
        self.test_x = list()
        self.test_y = list()

        # common model/sampler parameters
        self.layers_description = None
        self.model_parameters_size = None
        self.batch_size = 10
        self.chains_num = 1  # number of models to un in parallel; parameters are for each chain
        self.n_chunks = 3  # samples are drawn and stored in chunks
        self.n_samples = 50  # samples per chunk
        self.thinning = 0  # number of samples to discard

        self.sampler = None  # sampler created for current split
        self.sampler_factory = None

        # other
        self._log_handler = None

    def get_default_sampler_params(self):
        """ Returns default parameters for a Sampler. """
        params = dict()
        params['train_x'] = self.get_train_x()
        params['train_y'] = self.get_train_y()
        params['test_x'] = self.get_test_x()
        params['test_y'] = self.get_test_y()
        params['batch_size'] = self.batch_size

        return params

    def create_training_test_sets(self):
        """ For single var regressions only. """
        # load input data
        input_data = np.asarray(np.loadtxt("input/data.txt"), dtype=np.float32)
        self.input_dim = input_data.shape[1] - 1
        self.output_dim = 1

        self.data_size = input_data.shape[0]
        print(f"Loaded input data, shape = {input_data.shape}")

        # create splits
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        print(f"Splits: {self.n_splits}")

        for idx_train, idx_test in kfold.split(input_data):
            self.train_x.append(input_data[idx_train, :-1])
            self.train_y.append(input_data[idx_train, -1:])
            self.test_x.append(input_data[idx_test, :-1])
            self.test_y.append(input_data[idx_test, -1:])

        if self.layers_description is None:
            self.layers_description = [[self.input_dim, 0.0], [100, 0.0], [100, 0.0], [self.output_dim, 0.0]]

    def samples_per_chunk(self):
        return self.n_samples * (self.thinning + 1)

    def get_train_x(self):
        """ Returns current training set - x points. """
        return self.train_x[self.current_split]

    def get_train_y(self):
        """ Returns current training set - y labels. """
        return self.train_y[self.current_split]

    def get_test_x(self):
        """ Returns current test set - x points. """
        # return self.train_x[self.current_split]
        return self.test_x[self.current_split]

    def get_test_y(self):
        """ Returns current test set - y labels. """
        return self.test_y[self.current_split]

    def setup_data_dir(self, serialise_name='env'):
        """ Creates data directories and serialises the environment. """
        self.data_dir = self._create_test_dir_name()
        utils.set_data_dir(self.data_dir)

        if serialise_name is not None:
            utils.serialize(serialise_name, self)

        # configure file logging
        self._log_handler = logging.FileHandler(filename=utils.DATA_DIR + "/env.log", mode='w')
        self._log_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(self._log_handler)

    def run(self, store_data=True):
        """ Runs the experiments. """

        for split in range(self.n_splits):
            self.current_split = split
            self._run_split(store_data)
            # break  # TODO: one split only for now

    def _run_split(self, store_data):
        """ Creates sampler for the current split and draws the samples. """
        logging.info(f"Split: {self.current_split + 1} / {self.n_splits}")

        utils.set_data_dir(self.data_dir + "/split-" + str(self.current_split))
        logging.info(f"Data directory: {utils.DATA_DIR}")

        self.sampler = self.sampler_factory()

        with open(utils.DATA_DIR + "/sampler.txt", "w") as f:
            f.write(self.sampler.__repr__())

        logging.info("Total samples to draw: {}, samples per chain: {}, total samples to store: {}"
                     .format(self.samples_per_chunk() * self.chains_num * self.n_chunks,
                             self.samples_per_chunk() * self.n_chunks,
                             self.n_samples * self.chains_num * self.n_chunks))

        samples_drawn = 0

        with tf.Session() as session:
            tf.initialize_all_variables().run()

            collected_samples = list()
            collected_stats = list()

            # sample in chunks
            elapsed_ema = None
            for chunk in range(self.n_chunks):
                start = pc()

                # fit the model
                self.sampler.fit()

                # draw samples into current chunk
                for sample in range(self.n_samples):
                    if self.thinning > 0:
                        _ = [self.sampler.sample_predictive(session=session, is_discarded=True) for _ in
                             range(self.thinning)]

                    sample, stats = self.sampler.sample_predictive(return_stats=True, session=session)

                    collected_samples.extend(sample)
                    collected_stats.extend(stats)

                samples_drawn += self.chains_num * self.n_samples

                # report stats
                elapsed = pc() - start
                elapsed_ema = .1 * elapsed + .9 * elapsed_ema if elapsed_ema is not None else elapsed
                remaining = (self.n_chunks - chunk - 1) * elapsed_ema
                remaining = (remaining // 60, int(remaining) % 60)

                lag = self.n_samples * self.chains_num
                stats = collected_stats[-lag:]
                min_loss = min(stats, key=lambda s: s.loss).loss ** .5
                max_loss = max(stats, key=lambda s: s.loss).loss ** .5

                min_norm = min(stats, key=lambda s: s.norm).norm
                max_norm = max(stats, key=lambda s: s.norm).norm

                min_rate = min(stats, key=lambda s: s.rate).rate
                max_rate = max(stats, key=lambda s: s.rate).rate

                min_step = min(stats, key=lambda s: s.step).step
                max_step = max(stats, key=lambda s: s.step).step

                min_noise = min(stats, key=lambda s: s.noise_var).noise_var
                max_noise = max(stats, key=lambda s: s.noise_var).noise_var

                samples = collected_samples[-lag:]
                test_rmse = self.compute_rmse(np.asarray(samples))

                logging.info(f"Chunk = {chunk + 1}/{self.n_chunks}, elapsed = {elapsed:.1f}s, " +
                             f"remain = {remaining[0]:02.0f}:{remaining[1]:02.0f}, test RMSE: {test_rmse:.2f}, " +
                             f"rate = {min_rate:.2f}-{max_rate:.2f}, loss = {min_loss:.2f}-{max_loss:.2f}, " +
                             f"norm = {min_norm:.2f}-{max_norm:.2f}, step = {min_step:.12f}-{max_step:.12f}, " +
                             f"noise var = {min_noise:.2f}-{max_noise:.2f}")

                # store collected data
                if store_data and (((chunk + 1) % 10 == 0) or ((chunk + 1) == self.n_chunks)):
                    start = pc()
                    utils.serialize('samples', np.asarray(collected_samples))
                    utils.serialize('stats', collected_stats)

                    logging.info("---> Collections serialized in {:.0f} seconds.".format(pc() - start))

        logging.info("Sampling complete")
        logging.getLogger().removeHandler(self._log_handler)

    def _deserialise_from_split(self, name, split):
        data_dir = utils.DATA_DIR
        split_dir = utils.DATA_DIR + "/split-" + str(split)
        utils.DATA_DIR = split_dir
        data = utils.deserialize(name)
        utils.DATA_DIR = data_dir
        return data, split_dir

    def load_samples(self, split=0, discard_left=0., discard_right=0.):
        samples, split_dir = self._deserialise_from_split('samples', split)

        if discard_right > 0:
            samples = samples[int(discard_left * samples.shape[0]):-int(discard_right * samples.shape[0])]
        else:
            samples = samples[int(discard_left * samples.shape[0]):]

        return samples

    def load_stats(self, split=0, discard_left=0., discard_right=0., key=None):
        stats, split_dir = self._deserialise_from_split('stats', split)

        if discard_right > 0:
            stats = stats[int(discard_left * len(stats)):-int(discard_right * len(stats))]
        else:
            stats = stats[int(discard_left * len(stats)):]

        if key is not None:
            stats = np.asarray(list(map(key, stats)))

        return stats

    def load_times(self, split=0, discard_left=0., discard_right=0.):
        return self.load_stats(split=split, discard_left=discard_left, discard_right=discard_right,
                               key=lambda stat: stat.time)

    def compute_rmse(self, samples, test_y=None):
        samples = samples.squeeze()
        test_y = test_y if test_y is not None else self.get_test_y()
        test_y = test_y.squeeze()

        mean_prediction = samples.mean(axis=0)
        rmse = (np.mean((test_y - mean_prediction) ** 2)) ** .5

        return rmse

    def compute_metrics(self, baseline_samples, target_samples, discard_target=0.,
                        resample_baseline=1000, resample_target=1000, metric_names=None):
        baseline_samples = metrics.resample_to(baseline_samples, resample_baseline)

        target_samples = target_samples[int(discard_target * target_samples.shape[0]):]
        target_samples = metrics.resample_to(target_samples, resample_target)

        test_y = self.get_test_y()

        results = dict()
        if metric_names is None:
            metric_names = ["KS", "KL", "Precision", "Recall"]

        if "RMSE" in metric_names:
            results["RMSE"] = self.compute_rmse(target_samples, test_y=test_y)
            metric_names.remove("RMSE")

        for metric_name in metric_names:
            metric_fn = metrics.METRICS_INDEX[metric_name]
            values = list()
            for test_point in range(baseline_samples.shape[1]):
                values.append(metric_fn(baseline_samples[:, test_point], target_samples[:, test_point]))

            results[metric_name] = np.mean(values)

        return results

    def _create_test_dir_name(self):
        timestamp = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
        sample_num = self.n_samples * (self.thinning + 1) * self.chains_num * self.n_chunks

        name = ""
        name += self.env_name
        name += "-" + self.model_name
        name += "-" + self.test_case_name

        name += "--chains-" + str(self.chains_num)
        name += "--samples-" + str(sample_num)

        name += "--timestamp-" + timestamp

        return name
