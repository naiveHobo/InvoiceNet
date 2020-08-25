# Copyright (c) 2019 Tradeshift
# Copyright (c) 2020 Sarthak Mittal
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import tensorflow as tf

from ..acp.data import InvoiceData


class Parser(tf.keras.Model):
    def __init__(self):
        super(Parser, self).__init__()

    def restore(self):
        """
        Must return a tuple of (scope, restore_file_path).
        """
        raise NotImplementedError()


class NoOpParser(Parser):
    def __init__(self):
        super(NoOpParser, self).__init__()

    def restore(self):
        return None

    def call(self, inputs, training=None, mask=None):
        x, context = inputs
        return x


class OptionalParser(Parser):

    def __init__(self, delegate: Parser, seq_out):
        super(OptionalParser, self).__init__()
        self.seq_out = seq_out
        self.delegate = delegate
        self.dense_1 = tf.keras.layers.Dense(1)

    def restore(self):
        return self.delegate.restore()

    def call(self, inputs, training=None, mask=None):
        x, context = inputs
        parsed = self.delegate(inputs, training, mask)
        empty_answer = tf.fill([tf.shape(x)[0], self.seq_out], InvoiceData.eos_idx)
        empty_answer = tf.one_hot(empty_answer, InvoiceData.n_output)  # (bs, seq_out, n_out)
        logit_empty = self.dense_1(context)  # (bs, 1)
        return parsed + tf.expand_dims(logit_empty, axis=2) * empty_answer


class AmountParser(Parser):
    """
    You should pre-train this parser to parse amount otherwise it's hard to learn jointly.
    """
    def __init__(self):
        super(AmountParser, self).__init__()
        os.makedirs(r"./models/parsers/amount", exist_ok=True)

        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, unit_forget_bias=True, return_sequences=True),
            name="encoder")
        self.decoder = tf.keras.layers.LSTM(
            128, unit_forget_bias=True, return_sequences=True, name="decoder")

        self.encoder_dense = tf.keras.layers.Dense(128)
        self.decoder_dense = tf.keras.layers.Dense(128)
        self.attention_dense = tf.keras.layers.Dense(1)
        self.prob_dense = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
        self.gen_dense = tf.keras.layers.Dense(InvoiceData.n_output)

    def restore(self):
        return r"./models/parsers/amount/best"

    def call(self, inputs, training=None, mask=None):
        x, context = inputs

        # encoder
        h_in = self.encoder(x)
        h_in = tf.expand_dims(h_in, axis=2)  # (bs, seq_in, 1, 256)

        # decoder
        out_input = tf.zeros((tf.shape(x)[0], InvoiceData.seq_amount, 1))
        h_out = self.decoder(out_input)
        h_out = tf.expand_dims(h_out, axis=1)  # (bs, 1, seq_out, 128)

        # Bahdanau attention
        att = tf.math.tanh(self.decoder_dense(h_out) + self.encoder_dense(h_in))  # (bs, seq_in, seq_out, 128)
        att = self.attention_dense(att)  # (bs, seq_in, seq_out, 1)
        att = tf.math.softmax(att, axis=1)  # (bs, seq_in, seq_out, 1)

        attended_h = tf.reduce_sum(att * h_in, axis=1)  # (bs, seq_out, 128)

        p_gen = self.gen_dense(attended_h)  # (bs, seq_out, 1)
        p_copy = (1 - p_gen)

        # Generate
        gen = self.gen_dense(attended_h)  # (bs, seq_out, n_out)

        # Copy
        copy = tf.math.log(tf.reduce_sum(att * tf.expand_dims(x, axis=2), axis=1) + 1e-8)  # (bs, seq_out, n_out)

        output_logits = p_copy * copy + p_gen * gen
        return output_logits


class DateParser(Parser):
    """
    You should pre-train this parser to parse dates otherwise it's hard to learn jointly.
    """
    def __init__(self):
        super(DateParser, self).__init__()
        os.makedirs(r"./models/parsers/date", exist_ok=True)

        self.conv_block = tf.keras.Sequential()
        for _ in range(4):
            self.conv_block.add(tf.keras.layers.Conv1D(128, 3, padding='same', activation=tf.keras.activations.relu))
            self.conv_block.add(tf.keras.layers.MaxPool1D(2, 2))

        self.dense_block = tf.keras.Sequential()
        for _ in range(3):
            self.dense_block.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.relu))

        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense_out = tf.keras.layers.Dense(InvoiceData.seq_date * InvoiceData.n_output)

    def restore(self):
        return r"./models/parsers/date/best"

    def call(self, inputs, training=None, mask=None):
        x, context = inputs
        x = self.conv_block(x)  # (bs, 8, 128)
        x = tf.reduce_sum(x, axis=1)  # (bs, 128)
        x = tf.concat([x, context], axis=1)  # (bs, 256)
        x = self.dense_block(x, 256)  # (bs, 256)
        x = self.dropout(x, training=training)  # (bs, 256)
        x = self.dense_out(x)  # (bs, seq_out * n_out)
        return tf.reshape(x, (-1, InvoiceData.seq_date, InvoiceData.n_output))
