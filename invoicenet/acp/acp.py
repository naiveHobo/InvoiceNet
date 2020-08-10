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

from ..common.model import Model
from .data import InvoiceData
from .model import AttendCopyParseModel
from .. import FIELD_TYPES, FIELDS
from ..parsing.parsers import DateParser, AmountParser, NoOpParser, OptionalParser


class AttendCopyParse(Model):

    def __init__(self, field, restore=False):
        self.field = field

        self.restore_all_path = './models/invoicenet/{}/best'.format(self.field) if restore else None
        os.makedirs("./models/invoicenet", exist_ok=True)

        if FIELDS[field] == FIELD_TYPES["optional"]:
            noop_parser = NoOpParser()
            parser = OptionalParser(noop_parser, 128)
        elif FIELDS[field] == FIELD_TYPES["amount"]:
            parser = AmountParser()
        elif FIELDS[field] == FIELD_TYPES["date"]:
            parser = DateParser()
        else:
            parser = NoOpParser()

        restore = parser.restore()
        if restore is not None:
            print("Restoring %s parser %s..." % (self.field, restore))
            tf.train.Checkpoint(model=parser).read(restore).expect_partial()

        self.model = AttendCopyParseModel(parser=parser)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)

        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=3e-4)

        self.model.compile(self.optimizer)

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

        if self.restore_all_path:
            if not os.path.exists('./models/invoicenet/{}'.format(self.field)):
                raise Exception("No trained model available for the field '{}'".format(self.field))
            print("Restoring all " + self.restore_all_path + "...")
            self.checkpoint.read(self.restore_all_path).expect_partial()

    def loss_func(self, y_true, y_pred):
        mask = tf.cast(tf.logical_not(tf.equal(y_true, InvoiceData.pad_idx)), dtype=tf.float32)  # (bs, seq)
        label_cross_entropy = tf.reduce_sum(
            self.loss_object(y_true, y_pred) * mask, axis=1) / tf.reduce_sum(mask, axis=1)
        field_loss = tf.reduce_mean(label_cross_entropy)
        loss = field_loss + sum(self.model.losses)
        return loss

    @tf.function
    def train_step(self, inputs):
        inputs, targets = inputs[:-1], inputs[-1]
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_func(targets, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    @tf.function
    def val_step(self, inputs):
        inputs, targets = inputs[:-1], inputs[-1]
        predictions = self.model(inputs, training=False)
        loss = self.loss_func(targets, predictions)
        return loss

    def predict(self, paths):
        data = InvoiceData(field=self.field)
        shapes, types = data.shapes()[:-1], data.types()[:-1]

        def _transform(i, v, s, *args):
            return (tf.SparseTensor(i, v, s),) + args

        dataset = tf.data.Dataset.from_generator(
            data.generate_test_data(paths),
            types,
            shapes
        ).map(_transform) \
            .batch(batch_size=1, drop_remainder=False)

        predictions = []
        for sample in dataset:
            try:
                logits = self.model(sample, training=False)
                chars = tf.argmax(logits, axis=2, output_type=tf.int32).numpy()
                predictions.extend(data.array_to_str(chars))
            except tf.errors.OutOfRangeError:
                break

        return predictions

    def save(self, name):
        self.checkpoint.write(file_prefix="./models/invoicenet/%s/%s" % (self.field, name))

    def load(self, name):
        self.checkpoint.read(name)
