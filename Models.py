from abc import ABC
from typing import Tuple, Dict, Optional

import tensorflow as tf

from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer as lso
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Flatten, SimpleRNN, GRU


class BaseModel(Model):
    LayerIndex = int  # index of the layer in the model.weights sequence
    LayerName = str   # desired name of the layer
    ParamType = str   # either "w" for weight or "b" for bias
    LayerInfo = Tuple[LayerName, ParamType]
    LayerIndexNameDict = Optional[Dict[LayerIndex, LayerInfo]]  # either Dict or None

    def __init__(self, log_weights_period: int, log_gradients_period: int, layer_index_name_dict: LayerIndexNameDict):
        """ Base model class for a clear network

        :param log_weights_period (int): Period (in batches) at which to log weight values to files (0 == no logging)
        :param log_gradients_period (int): Period (in batches) at which to log grad values to files (0 == no logging)
        :param layer_index_name_dict (Opt[Dict[int: Tuple[str, str]]]): Dictionary of layers for logging
        """
        super(BaseModel, self).__init__()

        self.log_weights_period = log_weights_period
        self.log_gradients_period = log_gradients_period
        self.layer_index_name_dict = layer_index_name_dict

        # self.gradients = tf.Variable(tf.zeros(shape=(32, 48)), shape=(32, 48), dtype=tf.float32)
        if self.log_weights_period:
            self.weights_summary = tf.summary.create_file_writer("./log/weights/")
        if self.log_gradients_period:
            self.gradients_summary = tf.summary.create_file_writer("./log/gradients/")

    def train_step(self, data):
        """The logic for one training step.
        This method can be overridden to support custom training logic.
        This method is called by `Model.make_train_function`.
        This method should contain the mathemetical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.
        Arguments:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided. These utilities will be exposed
        # publicly.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        step = self.optimizer.iterations

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)

        # TODO: automate shape inference for all layers to make structures for gradients
        # TODO: maybe use metrics for the gradients instead?
        if self.log_weights_period:
            tf.cond(tf.cast(step % self.log_weights_period, tf.bool),
                    true_fn=lambda: None,
                    false_fn=lambda: _log_weights(self.weights_summary, self.weights, step, self.layer_index_name_dict))
        if self.log_gradients_period:
            tf.cond(tf.cast(step % self.log_gradients_period, tf.bool),
                    true_fn=lambda: None,
                    false_fn=lambda: _log_weights(self.gradients_summary, gradients, step, self.layer_index_name_dict))

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        # The _minimize call does a few extra steps unnecessary in most cases,
        # such as loss scaling and gradient clipping.
        # _minimize(self.distribute_strategy, tape, self.optimizer, loss, self.trainable_variables)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}


class DenseClassifier(BaseModel):

    def __init__(self, input_shape: Tuple[int, int], hidden_size: int, num_classes: int,
                 log_weights_period=0, log_gradients_period=0, layer_index_name_dict=None):
        """ Simple Dense Neural Network classifier with an input Flatten layer, hidden Dense layer
        and output Dense layer with Softmax output.

        :param input_shape (Tuple[int, int]): Shape of the input to the Flatten layer
        :param hidden_size (int): Number of units in hidden Dense layer
        :param num_classes (int): Number of units in output Dense layer (classes to classify)
        :param log_weights_period (int): Period (in batches) at which to log weight values to files (0 == no logging)
        :param log_gradients_period (int): Period (in batches) at which to log grad values to files (0 == no logging)
        :param layer_index_name_dict (Opt[Dict[int: Tuple[str, str]]]): Dictionary of layers for logging

        layer weight and bias shapes:
            l1w:  (flat_input_shape, hidden_size) = Input to hidden - weights
            l1b:  (hidden_size,)                  = Input to hidden - bias
            l2w:  (hidden_size, num_classes)      = Hidden to output - weights
            l2b:  (num_classes,)                  = Hidden to output - bias
        """
        if not layer_index_name_dict:
            layer_index_name_dict = {0: ("L1_Inp2Hid", "w"),
                                     1: ("L1_Inp2Hid", "b"),
                                     2: ("L2_Hid2Out", "w"),
                                     3: ("L2_Hid2Out", "b")}
        super(DenseClassifier, self).__init__(log_weights_period=log_weights_period,
                                              log_gradients_period=log_gradients_period,
                                              layer_index_name_dict=layer_index_name_dict)

        self.input_layer = Flatten(input_shape=input_shape)
        self.hidden_layer = Dense(hidden_size)
        self.output_layer = Dense(num_classes, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.hidden_layer(x)
        return self.output_layer(x)


class RNNClassifier(BaseModel):
    EmbeddingSize = int
    RNNSize = int
    DenseSize = int
    Sizes = Tuple[EmbeddingSize, RNNSize, DenseSize]

    def __init__(self, vocab_size: int, rnn_type: str, sizes: Sizes,
                 log_weights_period=0, log_gradients_period=0, layer_index_name_dict=None):
        """ Simple RNN/GRU binary classifier with Embedding input and single sigmoid output

        :param vocab_size (int): Number of unique values in inputs (vocabulary size for embedding)
        :param rnn_type (str): RNN cell type (so far can be "SimpleRNN", "CuDNNGRU" or "GRU")
        :param sizes (Tuple[int, int, int]): Number of units in [Embedding, RNN, Dense_output]
        :param log_weights_period (int): Period (in batches) at which to log weight values to files (0 == no logging)
        :param log_gradients_period (int): Period (in batches) at which to log grad values to files (0 == no logging)
        :param layer_index_name_dict (Opt[Dict[int: Tuple[str, str]]]): Dictionary of layers for logging

        embedding_size == sizes[0]
        rnn_size       == sizes[1]
        dense_size     == sizes[2]

        layer weight and bias shapes:
            l1w:   (vocab_size, embedding_size)                        = Input to Embedding - weights
            l2Whx: (embedding_size, rnn_size)                 for RNN
                   (embedding_size, embedding_size+rnn_size)  for GRU  = Input to RNN/GRU hidden - weights
            l2Whh: (rnn_size, rnn_size)                 for RNN
                   (rnn_size, embedding_size+rnn_size)  for GRU        = RNN hidden to RNN hidden/output - weights
            l2b:   (rnn_size, )                  for RNN
                   (embedding_size+rnn_size, )   for GRU
                   (2, embedding_size+rnn_size)  for CuDNNGRU          = RNN hidden to hidden/output - bias
            l3w:   (rnn_size, dense_size)                              = RNN output to dense1 - weight
            l3b:   (dense_size, )                                      = RNN output to dense1 - bias
            l4w:   (dense_size, 1)                                     = dense1 to output - weights
            l4b:   (1, )                                               = dense1 to output - bias
        """
        if not layer_index_name_dict:
            layer_index_name_dict = {0: ("L0_Embedding", "w"),
                                     4: ("L2_RNN2Hid", "w"),
                                     5: ("L2_RNN2Hid", "b"),
                                     6: ("L3_Hid2Out", "w"),
                                     7: ("L3_Hid2Out", "b")}
            if "simplernn" in rnn_type.lower():
                layer_index_name_dict[1] = ("L1_RNN_Whx", "w")
                layer_index_name_dict[2] = ("L1_RNN_Whh", "w")
                layer_index_name_dict[3] = ("L1_RNN_bh", "b")
            elif "cudnngru" in rnn_type.lower():
                layer_index_name_dict[1] = ("L1_GRU_WUx", "w")
                layer_index_name_dict[2] = ("L1_GRU_WUz", "w")
                layer_index_name_dict[3] = ("L1_GRU_bz", "w")
            elif "gru" in rnn_type.lower():
                layer_index_name_dict[1] = ("L1_GRU_WUx", "w")
                layer_index_name_dict[2] = ("L1_GRU_WUz", "w")
                layer_index_name_dict[3] = ("L1_GRU_bz", "b")
            else:
                raise NotImplementedError("RNN types other than 'SimpleRNN', 'CuDNNGRU' and 'GRU' are not supported.")
        super(RNNClassifier, self).__init__(log_weights_period=log_weights_period,
                                            log_gradients_period=log_gradients_period,
                                            layer_index_name_dict=layer_index_name_dict)

        embedding_size, rnn_size, dense_size = sizes

        if not embedding_size:
            embedding_size = 64
        if not rnn_size:
            rnn_size = 64
        if not dense_size:
            dense_size = 64

        self.embedding = Embedding(vocab_size, embedding_size, mask_zero=True)
        if "simplernn" in rnn_type.lower():
            self.rnn = SimpleRNN(rnn_size)
        elif "cudnngru" in rnn_type.lower():
            self.rnn = GRU(rnn_size)
        elif "gru" in rnn_type.lower():
            self.rnn = GRU(rnn_size, reset_after=False)
        else:
            raise NotImplementedError("RNN types other than 'SimpleRNN', 'CuDNNGRU' and 'GRU' are not supported yet.")
        self.dense = Dense(dense_size)
        self.class_out = Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        mask = self.embedding.compute_mask(inputs)
        x = self.rnn(x, mask=mask)
        x = self.dense(x)
        return self.class_out(x)


def _log_weights(summary_writer, weights_list, step, layer_index_name_dict):
    for i, (name, ptype) in layer_index_name_dict.items():
        w = weights_list[i]
        # w = w - tf.reduce_min(w) / (tf.reduce_max(w) - tf.reduce_min(w))  # normalize to (0., 1.)
        if ptype == "w":
            w = tf.transpose(w, (1, 0))
            w = tf.expand_dims(tf.expand_dims(w, 0), -1)
            with summary_writer.as_default():
                tf.summary.image(name+"_weight", w, step=step)
        elif ptype == "b":
            w = tf.expand_dims(tf.expand_dims(tf.expand_dims(w, 0), -1), -1)
            with summary_writer.as_default():
                tf.summary.image(name+"_bias", w, step=step)
        else:
            raise NotImplemented("Parameter type must be either 'w' (weight) or 'b' (bias)")


def _minimize(strategy, tape, optimizer, loss, trainable_variables):
    """Minimizes loss for one step by updating `trainable_variables`.
    This is roughly equivalent to
    ```python
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    ```
    However, this function also applies gradient clipping and loss scaling if the
    optimizer is a LossScaleOptimizer.
    Args:
    strategy: `tf.distribute.Strategy`.
    tape: A gradient tape. The loss must have been computed under this tape.
    optimizer: The optimizer used to minimize the loss.
    loss: The loss tensor.
    trainable_variables: The variables that will be updated in order to minimize
      the loss.
    """

    with tape:
        if isinstance(optimizer, lso.LossScaleOptimizer):
            loss = optimizer.get_scaled_loss(loss)

    gradients = tape.gradient(loss, trainable_variables)

    # Whether to aggregate gradients outside of optimizer. This requires support
    # of the optimizer and doesn't work with ParameterServerStrategy and
    # CentralStroageStrategy.
    aggregate_grads_outside_optimizer = (
            optimizer._HAS_AGGREGATE_GRAD and  # pylint: disable=protected-access
            not isinstance(strategy.extended,
                           parameter_server_strategy.ParameterServerStrategyExtended))

    if aggregate_grads_outside_optimizer:
        # We aggregate gradients before unscaling them, in case a subclass of
        # LossScaleOptimizer all-reduces in fp16. All-reducing in fp16 can only be
        # done on scaled gradients, not unscaled gradients, for numeric stability.
        gradients = optimizer._aggregate_gradients(zip(gradients,  # pylint: disable=protected-access
                                                       trainable_variables))
    if isinstance(optimizer, lso.LossScaleOptimizer):
        gradients = optimizer.get_unscaled_gradients(gradients)
    gradients = optimizer._clip_gradients(gradients)  # pylint: disable=protected-access
    if trainable_variables:
        if aggregate_grads_outside_optimizer:
            optimizer.apply_gradients(
                zip(gradients, trainable_variables),
                experimental_aggregate_gradients=False)
        else:
            optimizer.apply_gradients(zip(gradients, trainable_variables))
