import util

from datetime import datetime

from tensorflow.python.ops.losses.losses_impl import Reduction

from model import Model
from invoicenet.parsing.parsers import *
from invoicenet.parsing.data import *


class Parser(Model):
    devices = util.get_devices()
    batch_size = 128 * len(devices)
    context_size = 128
    experiment = datetime.now()
    type = 'amounts'  # valid are ['dates', 'amounts']
    output_length = {"dates": RealData.seq_date, "amounts": RealData.seq_amount}[type]
    continue_from = None

    def __init__(self):
        self.train = train = TabSeparated('invoicenet/parsing/data/%s/train.tsv' % self.type, self.output_length)
        self.train_iterator = self.iterator(train)

        valid = TabSeparated('invoicenet/parsing/data/%s/valid.tsv' % self.type, self.output_length)
        self.valid_iterator = self.iterator(valid)

        parser = {'amounts': AmountParser(self.batch_size), 'dates': DateParser(self.batch_size)}[self.type]

        print("Building graph...")
        config = tf.ConfigProto(allow_soft_placement=False)
        self.session = tf.Session(config=config)
        self.is_training_ph = tf.placeholder(tf.bool)

        source, self.targets = tf.cond(
            self.is_training_ph,
            true_fn=lambda: self.train_iterator.get_next(),
            false_fn=lambda: self.valid_iterator.get_next()
        )
        self.sources = source

        oh_inputs = tf.one_hot(source, train.n_output)  # (bs, seq, n_out)

        context = tf.zeros(
            (self.batch_size, self.context_size),
            dtype=tf.float32,
            name=None
        )

        output_logits = parser.parse(oh_inputs, context, self.is_training_ph)

        with tf.variable_scope('loss'):
            mask = tf.logical_not(tf.equal(self.targets, train.pad_idx))
            label_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.targets, output_logits, reduction=Reduction.NONE) * tf.to_float(mask)) / tf.log(2.)

            chars = tf.argmax(output_logits, axis=2, output_type=tf.int32)
            equal = tf.equal(self.targets, chars)
            acc = tf.reduce_mean(tf.to_float(tf.reduce_all(tf.logical_or(equal, tf.logical_not(mask)), axis=1)))

        self.actual = chars
        self.loss = label_cross_entropy

        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

        self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step, colocate_gradients_with_ops=True)

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        util.print_vars(tf.trainable_variables())

        if self.continue_from:
            print("Restoring " + self.continue_from + "...")
            self.saver.restore(self.session, self.continue_from)

    def iterator(self, data):
        return tf.data.Dataset.from_generator(
            data.sample_generator,
            data.types(),
            data.shapes()
        ).repeat(-1).batch(self.batch_size).prefetch(16).make_one_shot_iterator()

    def train_batch(self):
        step = self.session.run(self.global_step)

        if step % 1000 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, loss, sources, actual, targets, step = self.session.run([self.train_step, self.loss, self.sources, self.actual, self.targets, self.global_step], {self.is_training_ph: True}, options=run_options, run_metadata=run_metadata)
        elif step % 100 == 0:
            _, loss, sources, actual, targets, step = self.session.run([self.train_step, self.loss, self.sources, self.actual, self.targets, self.global_step], {self.is_training_ph: True})
        else:
            _, loss = self.session.run([self.train_step, self.loss], {self.is_training_ph: True})

        return loss

    def val_batch(self):
        loss, sources, actual, targets, step = self.session.run([self.loss, self.sources, self.actual, self.targets, self.global_step], {self.is_training_ph: False})
        return loss

    def save(self, name):
        self.saver.save(self.session, "./models/%s/%s" % (self.type, name))

    def load(self, name):
        self.saver.restore(self.session, name)


if __name__ == '__main__':
    m = Parser()
    m.train_batch()
    m.val_batch()
