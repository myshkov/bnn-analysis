"""
This module implements MC Dropout based sampler.
https://arxiv.org/pdf/1506.02142.pdf
Following https://github.com/yaringal/DropoutUncertaintyExps
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2
from keras import backend as K

from sampler import Sampler, SampleStats


class DropoutSampler(Sampler):
    """
    Dropout sampler for NNs.
    """

    def __init__(self, model=None, batch_size=None, n_epochs=None, **kwargs):
        """
        Creates a new DropoutSampler object.
        """
        super().__init__(**kwargs)
        self.sampler_type = 'Dropout'

        self.model = model
        self.batch_size = batch_size if batch_size is not None else self.train_set_size
        self.n_epochs = n_epochs

        self._sample_predictive_fn = None

    def __repr__(self):
        s = super().__repr__()
        s += f'Batch size: {self.batch_size}\n'
        return s

    def _fit(self, n_epochs=None, verbose=0, **kwargs):
        """ Fits the model before sampling. """
        n_epochs = n_epochs if n_epochs is not None else self.n_epochs
        self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size, nb_epoch=n_epochs,
                       verbose=verbose)

    def _sample_predictive(self, test_x=None, return_stats=False, **kwargs):
        """ Draws a new sample from the model. """
        if self._sample_predictive_fn is None:
            self._sample_predictive_fn = K.function([self.model.layers[0].input, K.learning_phase()],
                                                    [self.model.layers[-1].output])

        sample = self._sample_predictive_fn([test_x, 1])

        stats = None
        if return_stats:
            stats = SampleStats(time=self._running_time())

        return sample, [stats]

    @classmethod
    def model_from_description(cls, layers_description, w_reg, dropout=None, add_softmax=False):
        model = Sequential()

        def dropout_at(idx_layer):
            return dropout if dropout is not None else layers_description[idx_layer][1]

        # input layer
        model.add(Dropout(dropout_at(0), input_shape=(layers_description[0][0],)))

        # hidden layers
        for l in range(1, len(layers_description) - 1):
            model.add(Dense(input_dim=layers_description[l - 1][0], output_dim=layers_description[l][0],
                            W_regularizer=l2(w_reg),
                            activation='relu'))

            if dropout_at(l) > 0:
                model.add(Dropout(dropout_at(l)))

        model.add(
            Dense(input_dim=layers_description[-2][0], output_dim=layers_description[-1][0], W_regularizer=l2(w_reg)))

        if add_softmax:
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam')
        else:
            model.compile(loss='mse', optimizer='adam')

        return model
