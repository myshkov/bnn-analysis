"""
MC Dropout.
Dropout implementation and config following https://github.com/yaringal/DropoutUncertaintyExps
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
        self.sampler_type = "Dropout"

        self.model = model
        self.batch_size = batch_size if batch_size is not None else self.train_set_size
        self.n_epochs = n_epochs

        self._sample_predictive_fn = None

    def __repr__(self):
        s = super().__repr__()
        s += f"Batch size: {self.batch_size}\n"
        return s

    def _fit(self, n_epochs=None, verbose=0, **kwargs):
        """ Fits the model prior to sampling. """
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
        # model.add(Dropout(0, input_shape=(layers_description[0][0],)))

        # hidden layers
        for l in range(1, len(layers_description) - 1):
            # model.add(Dense(input_dim=layers[l - 1][0], output_dim=layers[l][0], W_regularizer=l2(w_reg),
            #                 b_regularizer=l2(w_reg), activation='relu'))

            model.add(Dense(input_dim=layers_description[l - 1][0], output_dim=layers_description[l][0],
                            W_regularizer=l2(w_reg),
                            activation='relu'))

            if dropout_at(l) > 0:
                model.add(Dropout(dropout_at(l)))

        # output layer
        # model.add(Dense(input_dim=layers[-2][0], output_dim=layers[-1][0], W_regularizer=l2(w_reg),
        #                 b_regularizer=l2(w_reg)))

        model.add(
            Dense(input_dim=layers_description[-2][0], output_dim=layers_description[-1][0], W_regularizer=l2(w_reg)))

        if add_softmax:
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam')
        else:
            model.compile(loss='mse', optimizer='adam')

        return model

#
# class net:
#     def __init__(self, X_train, y_train, n_hidden, n_epochs=40,
#                  normalize=False, X_test=None, y_test=None):
#
#         """
#             Constructor for the class implementing a Bayesian neural network
#             trained with the probabilistic back propagation method.
#
#             @param X_train      Matrix with the features for the training data.
#             @param y_train      Vector with the target variables for the
#                                 training data.
#             @param n_hidden     Vector with the number of neurons for each
#                                 hidden layer.
#             @param n_epochs     Numer of epochs for which to train the
#                                 network. The recommended value 40 should be
#                                 enough.
#             @param normalize    Whether to normalize the input features. This
#                                 is recommended unles the input vector is for
#                                 example formed by binary features (a
#                                 fingerlogging.info). In that case we do not recommend
#                                 to normalize the features.
#         """
#
#         # We normalize the training data to have zero mean and unit standard
#         # deviation in the training set if necessary
#
#         if normalize:
#             self.std_X_train = np.std(X_train, 0)
#             self.std_X_train[self.std_X_train == 0] = 1
#             self.mean_X_train = np.mean(X_train, 0)
#         else:
#             self.std_X_train = np.ones(X_train.shape[1])
#             self.mean_X_train = np.zeros(X_train.shape[1])
#
#         X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
#                   np.full(X_train.shape, self.std_X_train)
#
#         self.mean_y_train = np.mean(y_train)
#         self.std_y_train = np.std(y_train)
#
#         y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
#         y_train_normalized = np.array(y_train_normalized, ndmin=2).T
#
#         # We construct the network
#         N = X_train.shape[0]
#         dropout = 0.05
#         batch_size = 128
#         tau = 0.159707652696  # obtained from BO
#         lengthscale = 1e-2
#         reg = lengthscale ** 2 * (1 - dropout) / (2. * N * tau)
#
#         model = Sequential()
#         model.add(Dropout(dropout, input_shape=(X_train.shape[1],)))
#         model.add(Dense(n_hidden[0], activation='relu', W_regularizer=l2(reg)))
#         for i in range(len(n_hidden) - 1):
#             model.add(Dropout(dropout))
#             model.add(Dense(n_hidden[i + 1], activation='relu', W_regularizer=l2(reg)))
#         model.add(Dropout(dropout))
#         model.add(Dense(y_train_normalized.shape[1], W_regularizer=l2(reg)))
#
#         model.compile(loss='mean_squared_error', optimizer='adam')
#
#         # We iterate the learning process
#         start_time = time.time()
#         model.fit(X_train, y_train_normalized, batch_size=batch_size, nb_epoch=n_epochs, verbose=0)
#         self.model = model
#         self.tau = tau
#         self.running_time = time.time() - start_time
#
#         # We are done!
#
#     def predict(self, X_test, y_test):
#
#         """
#             Function for making predictions with the Bayesian neural network.
#
#             @param X_test   The matrix of features for the test data
#
#
#             @return m       The predictive mean for the test target variables.
#             @return v       The predictive variance for the test target
#                             variables.
#             @return v_noise The estimated variance for the additive noise.
#
#         """
#
#         X_test = np.array(X_test, ndmin=2)
#         y_test = np.array(y_test, ndmin=2).T
#
#         # We normalize the test set
#
#         X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
#                  np.full(X_test.shape, self.std_X_train)
#
#         # We compute the predictive mean and variance for the target variables
#         # of the test data
#
#         model = self.model
#         standard_pred = model.predict(X_test, batch_size=500, verbose=1)
#         standard_pred = standard_pred * self.std_y_train + self.mean_y_train
#         rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze()) ** 2.) ** 0.5
#
#         T = 10000
#         predict_stochastic = K.function([model.layers[0].input, K.learning_phase()], model.layers[-1].output)
#
#         Yt_hat = np.array([predict_stochastic([X_test, 1]) for _ in range(T)])
#         Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
#         MC_pred = np.mean(Yt_hat, 0)
#         rmse = np.mean((y_test.squeeze() - MC_pred.squeeze()) ** 2.) ** 0.5
#
#         # We compute the test log-likelihood
#         ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat) ** 2., 0) - np.log(T)
#               - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(self.tau))
#         test_ll = np.mean(ll)
#
#         logging.info
#         'Standard rmse %f' % (rmse_standard_pred)
#         logging.info
#         'MC rmse %f' % (rmse)
#         logging.info
#         'test_ll %f' % (test_ll)
#
#         # We are done!
#         return rmse_standard_pred, rmse, test_ll
#
# #
# def test_dropout():
#     # construct model
#     learning_rate = 0.0005
#     dropout_prob = 0.99
#
#     graph = tf.Graph()
#
#     with graph.as_default():
#         tensor_x = tf.placeholder(tf.float32, shape=(None, 784), name='X')
#         tensor_y = tf.placeholder(tf.float32, shape=(None, 10), name='Y')
#
#         keep_prob = tf.placeholder(tf.float32)
#
#         previous_layer = tensor_x
#
#         # hidden layers
#         for l in range(1, len(layers_description) - 1):
#             weights = tf.Variable(tf.truncated_normal([layers_description[l - 1][0], layers_description[l][0]]))
#             biases = tf.Variable(tf.zeros([layers_description[l][0]]))
#
#             layer = tf.matmul(previous_layer, weights) + biases
#             logits = tf.nn.relu(layer)
#             dl = tf.nn.dropout(logits, keep_prob=keep_prob)
#
#             if l == 1:
#                 w1 = weights
#                 b1 = biases
#
#             if l == 2:
#                 w2 = weights
#                 b2 = biases
#
#             previous_layer = dl
#
#         # output layer
#         weights = tf.Variable(tf.truncated_normal([layers_description[-2][0], layers_description[-1][0]]))
#         biases = tf.Variable(tf.zeros([layers_description[-1][0]]))
#         prediction_t = tf.matmul(previous_layer, weights) + biases
#         prediction = tf.nn.softmax(prediction_t)
#
#         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction_t, tensor_y))
#
#                # +               0.01 * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(
#                #     weights) + tf.nn.l2_loss(biases))
#
#         optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#
#     with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False)) as session:
#         tf.initialize_all_variables().run()
#
#         for step in range(10000):
#             feed_dict = {tensor_x: train_x, tensor_y: train_y, keep_prob: dropout_prob}
#             _, l, predictions = session.run([optimizer, loss, prediction], feed_dict=feed_dict)
#
#             if step % 1000 == 0:
#                 logging.info("Loss at step {}: {:.2f}".format(step, l))
#
#                 # logging.info("x = {0}, y = {1}, prediction = {2:.2f}, loss = {3:.2f}".format(batch_data[0, 0],
#                 #                                                                       batch_labels[0, 0],
#                 #                                                                       predictions[0, 0], l))
#
#         # feed_dict = {tensor_x: test_x_line,
#         #              tensor_y: test_y_line,
#         #              keep_prob: dropout_prob}
#
#         feed_dict = {tensor_x: test_x,
#                      tensor_y: test_y,
#                      keep_prob: dropout_prob}
#
#         pred_y = []
#
#         for n in range(500):
#             p, l = session.run([prediction, loss], feed_dict=feed_dict)
#             pred_y.append(p)
#
#         pred_y = np.asarray(pred_y)
#         # data = pred_y[:, :, 0]
#
#         # data = utils.deserialize('do-samples')
#         utils.serialize('do-samples', pred_y)
#         return
#
#         pad_width = test_x_line.shape[0] - train_x.shape[0]
#         train_x_padded = np.pad(train_x[:, 0], (0, pad_width), 'constant', constant_values=np.nan)
#         train_y_padded = np.pad(train_y[:, 0], (0, pad_width), 'constant', constant_values=np.nan)
#
#         utils.serialize('dropout-data', data)
#
#         df = pd.DataFrame.from_dict({
#             "time": test_x_line[:, 0],
#             "true_y": test_y_line[:, 0],
#             "tx": train_x_padded,
#             "ty": train_y_padded,
#             "mean": data.mean(axis=0),
#             "std": 2. * data.std(axis=0),
#         }).reset_index()
#
#         g = sns.FacetGrid(df, size=9, aspect=1.8)
#         g.map(plt.errorbar, "time", "mean", "std", color=(0.1, 0.1, 0.6, 0.1))
#         g.map(plt.scatter, "tx", "ty", color="g")
#         g.map(plt.plot, "time", "true_y", color="r", lw=1)
#
#         g.ax.set(xlabel="X", ylabel="Y")
#         plt.xlim(view_xrange[0], view_xrange[1])
#         plt.ylim(view_yrange[0], view_yrange[1])
#         plt.title("Dropout")
#
#         plt.legend(['Prediction mean', 'True f(x)', 'Train data', 'Prediction StdDev'])
#
#         plt.show()
