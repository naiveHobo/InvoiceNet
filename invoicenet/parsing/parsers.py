import os
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib import layers
from invoicenet.acp.data import RealData

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Parser:
    def parse(self, x, context, is_training):
        raise NotImplementedError()

    def restore(self):
        """
        Must return a tuple of (scope, restore_file_path).
        """
        raise NotImplementedError()


class NoOpParser(Parser):
    def restore(self):
        return None

    def parse(self, x, context, is_training):
        return x


class OptionalParser(Parser):
    def __init__(self, delegate: Parser, bs, seq_out, n_out, eos_idx):
        self.eos_idx = eos_idx
        self.n_out = n_out
        self.seq_out = seq_out
        self.bs = bs

        self.delegate = delegate

    def restore(self):
        return self.delegate.restore()

    def parse(self, x, context, is_training):
        parsed = self.delegate.parse(x, context, is_training)

        empty_answer = tf.constant(self.eos_idx, tf.int32, shape=(self.bs, self.seq_out))
        empty_answer = tf.one_hot(empty_answer, self.n_out)  # (bs, seq_out, n_out)

        logit_empty = layers.fully_connected(context, 1, activation_fn=None)  # (bs, 1)

        return parsed + tf.reshape(logit_empty, (self.bs, 1, 1)) * empty_answer


class AmountParser(Parser):
    """
    You should pre-train this parser to parse amount otherwise it's hard to learn jointly.
    """
    seq_in = RealData.seq_in
    seq_out = RealData.seq_amount
    n_out = len(RealData.chars)
    scope = 'parse/amount'

    def __init__(self, bs):
        os.makedirs(r"./models/parsers/amount", exist_ok=True)
        self.bs = bs

    def restore(self):
        return self.scope, r"./models/parsers/amount/best"

    def parse(self, x, context, is_training):
        with tf.variable_scope(self.scope):
            with tf.variable_scope("encoder"):
                lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(128, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(128, forget_bias=1.0, state_is_tuple=True)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                            cell_bw=lstm_bw_cell,
                                                                            inputs=x,
                                                                            dtype=tf.float32,
                                                                            scope="BiLSTM")

                h_in = tf.concat([output_fw, output_bw], axis=2)
                h_in = tf.reshape(h_in, (self.bs, self.seq_in, 1, 256))  # (bs, seq_in, 1, 128)

            with tf.variable_scope("decoder"):
                out_input = tf.zeros((self.bs, self.seq_out, 1))
                out_input = tf.unstack(out_input, self.seq_out, 1)
                cell = tf.nn.rnn_cell.LSTMCell(128, forget_bias=1.0, state_is_tuple=True)
                h_out, _ = rnn.static_rnn(cell, out_input, dtype=tf.float32)
                h_out = tf.reshape(tf.concat(h_out, axis=-1), [self.bs, 1, self.seq_out, 128])

            # Bahdanau attention
            att = tf.nn.tanh(layers.fully_connected(h_out, 128, activation_fn=None) + layers.fully_connected(h_in, 128,
                                                                                                             activation_fn=None))
            att = layers.fully_connected(att, 1, activation_fn=None)  # (bs, seq_in, seq_out, 1)
            att = tf.nn.softmax(att, axis=1)  # (bs, seq_in, seq_out, 1)

            attended_h = tf.reduce_sum(att * h_in, axis=1)  # (bs, seq_out, 128)

            p_gen = layers.fully_connected(attended_h, 1, activation_fn=tf.nn.sigmoid)  # (bs, seq_out, 1)
            p_copy = (1 - p_gen)

            # Generate
            gen = layers.fully_connected(attended_h, self.n_out, activation_fn=None)  # (bs, seq_out, n_out)
            gen = tf.reshape(gen, (self.bs, self.seq_out, self.n_out))

            # Copy
            copy = tf.log(tf.reduce_sum(att * tf.reshape(x, (self.bs, self.seq_in, 1, self.n_out)),
                                        axis=1) + 1e-8)  # (bs, seq_out, n_out)

            output_logits = p_copy * copy + p_gen * gen
            return output_logits


class DateParser(Parser):
    """
    You should pre-train this parser to parse dates otherwise it's hard to learn jointly.
    """
    seq_out = RealData.seq_date
    n_out = len(RealData.chars)
    scope = 'parse/date'

    def __init__(self, bs):
        os.makedirs(r"./models/parsers/date", exist_ok=True)
        self.bs = bs

    def restore(self):
        return self.scope, r"./models/parsers/date/best"

    def parse(self, x, context, is_training):
        with tf.variable_scope(self.scope):
            for i in range(4):
                x = tf.layers.conv1d(x, 128, 3, padding="same", activation=tf.nn.relu)  # (bs, 128, 128)
                x = tf.layers.max_pooling1d(x, 2, 2)  # (bs, 64-32-16-8, 128)
            x = tf.reduce_sum(x, axis=1)  # (bs, 128)

            x = tf.concat([x, context], axis=1)  # (bs, 256)
            for i in range(3):
                x = layers.fully_connected(x, 256)

            x = layers.dropout(x, is_training=is_training)
            x = layers.fully_connected(x, self.seq_out * self.n_out, activation_fn=None)
            return tf.reshape(x, (self.bs, self.seq_out, self.n_out))
