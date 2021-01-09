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

import tensorflow as tf

from .data import InvoiceData


class DilatedConvBlock(tf.keras.layers.Layer):

    def __init__(self, filters):
        super(DilatedConvBlock, self).__init__()
        self.conv_layers = [
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=3,
                padding='same',
                dilation_rate=rate,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )
            for rate in [1, 2, 4, 8]
        ]

    def call(self, inputs, training=None, mask=None):
        return tf.concat([conv(inputs) for conv in self.conv_layers], axis=3)


class AttendBlock(tf.keras.layers.Layer):

    def __init__(self, embed_size=32, frac_ce_loss=0.0001):
        super(AttendBlock, self).__init__()

        self.frac_ce_loss = frac_ce_loss
        self.embed_size = embed_size
        self.word_embed = tf.keras.layers.Embedding(
            input_dim=InvoiceData.word_hash_size,
            output_dim=embed_size,
            input_length=InvoiceData.im_size[0] * InvoiceData.im_size[1],
            name="word_embeddings")
        self.pattern_embed = tf.keras.layers.Embedding(
            input_dim=InvoiceData.pattern_hash_size,
            output_dim=embed_size,
            input_length=InvoiceData.im_size[0] * InvoiceData.im_size[1],
            name="pattern_embeddings")
        self.char_embed = tf.keras.layers.Embedding(
            input_dim=InvoiceData.n_output,
            output_dim=embed_size,
            input_length=InvoiceData.im_size[0] * InvoiceData.im_size[1],
            name="char_embeddings")

        self.conv_block = tf.keras.Sequential()
        for _ in range(4):
            self.conv_block.add(DilatedConvBlock(embed_size))

        self.dropout = tf.keras.layers.Dropout(0.5)
        self.conv_att = tf.keras.layers.Conv2D(
            filters=InvoiceData.n_memories,
            kernel_size=3,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )

    def call(self, inputs, training=None, mask=None):
        pixels, word_indices, pattern_indices, char_indices, memory_mask, parses = inputs

        # pixels: (bs, h, w)
        # word_indices: (bs, h, w)
        # pattern_indices: (bs, h, w)
        # char_indices: (bs, h, w)
        # memory_mask: (bs, h, w, m, l, d)
        # parses: (bs, h, w, 4, 2)

        bs = tf.shape(pixels)[0]
        h, w = InvoiceData.im_size[0], InvoiceData.im_size[1]

        X, Y = tf.meshgrid(tf.linspace(0.0, 1.0, InvoiceData.im_size[1]), tf.linspace(0.0, 1.0, InvoiceData.im_size[0]))
        X = tf.tile(X[None, ..., None], (bs, 1, 1, 1))
        Y = tf.tile(Y[None, ..., None], (bs, 1, 1, 1))

        word_embeddings = tf.reshape(
            self.word_embed(tf.reshape(word_indices, (bs, -1))),
            (bs, h, w, self.embed_size)
        )

        pattern_embeddings = tf.reshape(
            self.pattern_embed(tf.reshape(pattern_indices, (bs, -1))),
            (bs, h, w, self.embed_size)
        )

        char_embeddings = tf.reshape(
            self.char_embed(tf.reshape(char_indices, (bs, -1))),
            (bs, h, w, self.embed_size)
        )

        pixels = tf.reshape(pixels, (bs, h, w, 3))
        parses = tf.reshape(parses, (bs, h, w, InvoiceData.n_memories * 2))
        memory_mask = tf.reshape(memory_mask, (bs, h, w, 1))
        x = tf.concat([pixels, word_embeddings, pattern_embeddings, char_embeddings, parses, X, Y, memory_mask],
                      axis=3)

        x = self.conv_block(x)
        x = self.dropout(x, training=training)

        pre_att_logits = x
        att_logits = self.conv_att(x)  # (bs, h, w, n_memories)
        att_logits = memory_mask * att_logits - (
                1.0 - memory_mask) * 1000  # TODO only sum the memory_mask idx, in the softmax

        logits = tf.reshape(att_logits, (bs, -1))  # (bs, h * w * n_memories)
        logits -= tf.reduce_max(logits, axis=1, keepdims=True)
        lp = tf.math.log_softmax(logits, axis=1)  # (bs, h * w * n_memories)
        p = tf.math.softmax(logits, axis=1)  # (bs, h * w * n_memories)

        spatial_attention = tf.reshape(p, (bs, h * w * InvoiceData.n_memories, 1, 1))  # (bs, h * w * n_memories, 1, 1)

        p_uniform = memory_mask / tf.reduce_sum(memory_mask, axis=(1, 2, 3), keepdims=True)
        cross_entropy_uniform = -tf.reduce_sum(p_uniform * tf.reshape(lp, (bs, h, w, InvoiceData.n_memories)),
                                               axis=(1, 2, 3))  # (bs, 1)

        cp = tf.reduce_sum(tf.reshape(p, (bs, h, w, InvoiceData.n_memories)), axis=3, keepdims=True)

        context = tf.reduce_sum(cp * pre_att_logits, axis=(1, 2))  # (bs, 4*n_hidden)

        self.add_loss(self.frac_ce_loss * tf.reduce_mean(cross_entropy_uniform))

        return spatial_attention, context


class AttendCopyParseModel(tf.keras.Model):
    """
    You should pre-train this parser to parse dates otherwise it's hard to learn jointly.
    """
    def __init__(self, parser):
        super(AttendCopyParseModel, self).__init__()
        self.parser = parser
        self.attend = AttendBlock(32)

    def call(self, inputs, training=None, mask=None):
        memories, pixels, word_indices, pattern_indices, char_indices, memory_mask, parses = inputs

        spatial_attention, context = self.attend(inputs=(pixels,
                                                         word_indices,
                                                         pattern_indices,
                                                         char_indices,
                                                         memory_mask,
                                                         parses),
                                                 training=training)

        # Copy
        memories = tf.sparse.reshape(memories,
                                     (-1, InvoiceData.im_size[0] * InvoiceData.im_size[1] * InvoiceData.n_memories,
                                      InvoiceData.seq_in,
                                      InvoiceData.n_output))
        x = tf.reshape(tf.sparse.reduce_sum(spatial_attention * memories, axis=1),
                       (-1, InvoiceData.seq_in, InvoiceData.n_output))  # (bs, seq_in, n_out)

        # Parse
        parsed = self.parser(inputs=(x, context), training=training)
        return parsed
