""" PBP Sampler:
Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks
https://arxiv.org/abs/1502.05336

Uses implementation by the authors:
https://github.com/HIPS/Probabilistic-Backpropagation
"""
import numpy as np

from sampler import Sampler, SampleStats
# from models.PBP_net import PBP_net  # put PBP_net in this directory and uncomment to use the sampler


class PBPSampler(Sampler):
    """
    PBP sampler for NNs.
    """

    def __init__(self, model_desc=None, n_epochs=None, **kwargs):
        """
        Creates a new PBPSampler object.
        """
        super().__init__(**kwargs)
        self.sampler_type = 'PBP'

        self.models_desc = model_desc
        self._n_epochs = n_epochs

        self._model = None
        self._prediction = None

        shapes = list(list(zip(*self.models_desc))[0])
        shapes = shapes[1:-1]

    def __repr__(self):
        s = super().__repr__()
        return s

    def _fit(self, **kwargs):
        """ Fits the model before sampling. """
        if self._model is None:
            shapes = list(list(zip(*self.models_desc))[0])
            shapes = shapes[1:-1]

            self._model = PBP_net.PBP_net(self.train_x, self.train_y.squeeze(), shapes, normalize=True,
                                          n_epochs=self._n_epochs)
        else:
            self._model.re_train(self.train_x, self.train_y.squeeze(), self._n_epochs)

        *self._prediction, v_noise = self._model.predict(self.test_x)

    def _sample_predictive(self, test_x=None, return_stats=False, **kwargs):
        """ Draws a new sample from the model. """

        m, v = self._prediction
        std = v ** .5

        sample = np.random.randn(test_x.shape[0])
        sample = m + std * sample
        sample = np.expand_dims(sample, axis=-1)
        stats = None

        if sample is not None and return_stats:
            stats = SampleStats(time=self._running_time())

        return [sample], [stats]
