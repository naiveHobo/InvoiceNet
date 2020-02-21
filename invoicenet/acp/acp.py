from tensorflow.contrib import layers
from tensorflow.python.ops.losses.losses_impl import Reduction

from invoicenet.common import util
from invoicenet.common.model import Model
from invoicenet.acp.data import *
from invoicenet.parsing.parsers import DateParser, AmountParser, NoOpParser, OptionalParser


class AttendCopyParse(Model):
    devices = util.get_devices()
    n_hid = 32
    frac_ce_loss = 0.0001
    lr = 3e-4
    keep_prob = 0.5

    def __init__(self, data_dir, field, batch_size=8, restore=False):
        self.data_dir = data_dir
        self.field = field
        self.batch_size = batch_size * len(self.devices)

        self.noop_parser = NoOpParser()
        self.opt_noop_parser = OptionalParser(self.noop_parser, self.batch_size, 128, 103, 1)
        self.date_parser = DateParser(self.batch_size)
        self.amount_parser = AmountParser(self.batch_size)

        self.field_parsers = {
            "vendorname": self.noop_parser,
            "invoicedate": self.date_parser,
            "invoicenumber": self.noop_parser,
            "amountnet": self.amount_parser,
            "amounttax": self.amount_parser,
            "amounttotal": self.amount_parser,
            "vatrate": self.noop_parser,
            "vatid": self.opt_noop_parser,
            "taxid": self.opt_noop_parser,
            "iban": self.opt_noop_parser,
            "bic": self.opt_noop_parser
        }

        self.restore_all_path = './models/invoicenet/{}/best'.format(self.field) if restore else None
        os.makedirs("./models/invoicenet", exist_ok=True)
        self.train = train = RealData(os.path.join(self.data_dir, 'train/'))
        self.train_iterator = self.iterator(train)
        self.next_train_batch = self.train_iterator.get_next()

        valid = RealData(os.path.join(self.data_dir, "val/"))
        self.valid_iterator = self.iterator(valid)
        self.next_valid_batch = self.valid_iterator.get_next()

        self.test = test = RealData(os.path.join(self.data_dir, "predict/"))
        self.test_iterator = self.iterator(test, n_repeat=1)
        self.next_test_batch = self.test_iterator.get_next()

        self.regularizer = layers.l2_regularizer(1e-4)

        print("Building graph...")
        config = tf.ConfigProto(allow_soft_placement=False)
        self.session = tf.Session(config=config)

        # Placeholders
        self.is_training_ph = tf.placeholder(tf.bool)
        self.memories_ph = tf.sparse_placeholder(tf.float32, name="memories")
        self.pixels_ph = tf.placeholder(tf.float32, name='pixels')
        self.word_indices_ph = tf.placeholder(tf.int32, name="word_indices")
        self.pattern_indices_ph = tf.placeholder(tf.int32, name="pattern_indices")
        self.char_indices_ph = tf.placeholder(tf.int32, name="char_indices")
        self.memory_mask_ph = tf.placeholder(tf.float32, name="memory_mask")
        self.parses_ph = tf.placeholder(tf.float32, name="parses")
        self.found_ph = tf.placeholder(tf.float32, name="found")
        # targets
        self.vendorname_ph = tf.placeholder(tf.int32, name="vendorname")
        self.invoicedate_ph = tf.placeholder(tf.int32, name="invoicedate")
        self.invoicenumber_ph = tf.placeholder(tf.int32, name="invoicenumber")
        self.amountnet_ph = tf.placeholder(tf.int32, name="amountnet")
        self.amounttax_ph = tf.placeholder(tf.int32, name="amounttax")
        self.amounttotal_ph = tf.placeholder(tf.int32, name="amounttotal")
        self.vatrate_ph = tf.placeholder(tf.int32, name="vatrate")
        self.vatid_ph = tf.placeholder(tf.int32, name="vatid")
        self.taxid_ph = tf.placeholder(tf.int32, name="taxid")
        self.iban_ph = tf.placeholder(tf.int32, name="iban")
        self.bic_ph = tf.placeholder(tf.int32, name="bic")

        self.targets = {
            "vendorname": self.vendorname_ph,
            "invoicedate": self.invoicedate_ph,
            "invoicenumber": self.invoicenumber_ph,
            "amountnet": self.amountnet_ph,
            "amounttax": self.amounttax_ph,
            "amounttotal": self.amounttotal_ph,
            "vatrate": self.vatrate_ph,
            "vatid": self.vatid_ph,
            "taxid": self.taxid_ph,
            "iban": self.iban_ph,
            "bic": self.bic_ph
        }

        h, w = train.im_size
        bs = self.batch_size
        seq_in = train.seq_in
        n_out = train.n_output

        field_idx = ["vendorname", "invoicedate", "invoicenumber", "amountnet", "amounttax", "amounttotal", "vatrate",
                     "vatid", "taxid", "iban", "bic"].index(self.field)

        def dilated_block(x):
            return tf.concat(
                [layers.conv2d(x, self.n_hid, 3, rate=rate, activation_fn=None, weights_regularizer=self.regularizer)
                 for rate in [1, 2, 4, 8]], axis=3)

        def attend(pixels, word_indices, pattern_indices, char_indices, memory_mask, parses):
            """
            :param pixels: (bs, h, w)
            :param word_indices: (bs, h, w)
            :param pattern_indices: (bs, h, w)
            :param char_indices: (bs, h, w)
            :param parses: (bs, h, w, 4, 2)
            """
            bs = tf.shape(pixels)[0]

            X, Y = tf.meshgrid(tf.linspace(0.0, 1.0, RealData.im_size[0]), tf.linspace(0.0, 1.0, RealData.im_size[0]))
            X = tf.tile(X[None, ..., None], (bs, 1, 1, 1))
            Y = tf.tile(Y[None, ..., None], (bs, 1, 1, 1))

            word_embeddings = tf.reshape(
                layers.embed_sequence(tf.reshape(word_indices, (bs, -1)), vocab_size=train.word_hash_size,
                                      embed_dim=self.n_hid, unique=False, scope="word-embeddings"),
                (bs, h, w, self.n_hid))
            pattern_embeddings = tf.reshape(
                layers.embed_sequence(tf.reshape(pattern_indices, (bs, -1)), vocab_size=train.pattern_hash_size,
                                      embed_dim=self.n_hid, unique=False, scope="pattern-embeddings"),
                (bs, h, w, self.n_hid))
            char_embeddings = tf.reshape(
                layers.embed_sequence(tf.reshape(char_indices, (bs, -1)), vocab_size=train.n_output,
                                      embed_dim=self.n_hid, unique=False, scope="char-embeddings"),
                (bs, h, w, self.n_hid))

            pixels = tf.reshape(pixels, (bs, h, w, 3))
            parses = tf.reshape(parses, (bs, h, w, 8))
            memory_mask = tf.reshape(memory_mask, (bs, h, w, 1))
            x = tf.concat([pixels, word_embeddings, pattern_embeddings, char_embeddings, parses, X, Y, memory_mask],
                          axis=3)

            with tf.variable_scope('attend'):
                # x = tf.nn.relu(dilated_block(x))
                for i in range(4):
                    x = tf.nn.relu(dilated_block(x))

                x = layers.dropout(x, self.keep_prob, is_training=self.is_training_ph)
                pre_att_logits = x
                att_logits = layers.conv2d(x, train.n_memories, 3, activation_fn=None,
                                           weights_regularizer=self.regularizer)  # (bs, h, w, n_memories)
                att_logits = memory_mask * att_logits - (
                        1.0 - memory_mask) * 1000  # TODO only sum the memory_mask idx, in the softmax

                logits = tf.reshape(att_logits, (bs, -1))  # (bs, h * w * n_memories)
                logits -= tf.reduce_max(logits, axis=1, keepdims=True)
                lp = tf.nn.log_softmax(logits, axis=1)  # (bs, h * w * n_memories)
                p = tf.nn.softmax(logits, axis=1)  # (bs, h * w * n_memories)

                spatial_attention = tf.reshape(p,
                                               (bs, h * w * train.n_memories, 1, 1))  # (bs, h * w * n_memories, 1, 1)

                p_uniform = memory_mask / tf.reduce_sum(memory_mask, axis=(1, 2, 3), keepdims=True)
                cross_entropy_uniform = -tf.reduce_sum(p_uniform * tf.reshape(lp, (bs, h, w, train.n_memories)),
                                                       axis=(1, 2, 3))  # (bs, 1)

                cp = tf.reduce_sum(tf.reshape(p, (bs, h, w, train.n_memories)), axis=3, keepdims=True)

                context = tf.reduce_sum(cp * pre_att_logits, axis=(1, 2))  # (bs, 4*n_hidden)

            return spatial_attention, cross_entropy_uniform, context

        spatial_attention, cross_entropy_uniform, context = util.batch_parallel(
            attend,
            self.devices,
            pixels=self.pixels_ph,
            word_indices=self.word_indices_ph,
            pattern_indices=self.pattern_indices_ph,
            char_indices=self.char_indices_ph,
            memory_mask=self.memory_mask_ph,
            parses=self.parses_ph
        )

        context = tf.concat(context, axis=0)  # (bs, 128)
        spatial_attention = tf.concat(spatial_attention, axis=0)  # (bs, h * w * n_mem, 1, 1)
        cross_entropy_uniform = tf.concat(cross_entropy_uniform, axis=0)  # (bs, 1)

        with tf.variable_scope('copy'):
            memories = tf.sparse_reshape(self.memories_ph,
                                         (self.batch_size, h * w * train.n_memories, train.seq_in, n_out))
            x = tf.reshape(tf.sparse_reduce_sum(spatial_attention * memories, axis=1),
                           (bs, seq_in, n_out))  # (bs, seq_in, n_out)

        with tf.name_scope('parse'):
            parser = self.field_parsers[self.field]
            parsed = parser.parse(x, context, self.is_training_ph)
            target = self.targets[self.field]
            output = self.output(parsed, targets=target, scope=self.field)
            self.outputs = {self.field: output}

        reg_loss = tf.losses.get_regularization_loss()
        cross_entropy_uniform_loss = self.frac_ce_loss * tf.reduce_mean(cross_entropy_uniform)
        field_loss = tf.reduce_mean(self.outputs[self.field]['cross_entropy'])  # (bs, )

        self.loss = field_loss + reg_loss + cross_entropy_uniform_loss

        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads, vars = zip(*self.optimizer.compute_gradients(self.loss, colocate_gradients_with_ops=True))
            self.train_step = self.optimizer.apply_gradients(zip(grads, vars), global_step=self.global_step)

        # Savers
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())

        if self.restore_all_path:
            print("Restoring all " + self.restore_all_path + "...")
            self.saver.restore(self.session, self.restore_all_path)
        else:
            restore = parser.restore()
            if restore is not None:
                scope, fname = restore
                vars = tf.trainable_variables(scope=scope)
                saver = tf.train.Saver(var_list=vars)
                print("Restoring %s parser %s..." % (self.field, fname))
                for var in vars:
                    print("-- restoring %s" % var)
                saver.restore(self.session, fname)

        self.found = tf.reduce_mean(self.found_ph[:, field_idx])

    def output(self, logits, targets, scope, optional=None):
        with tf.variable_scope(scope):
            if optional:
                logoutput_p, empty_answer = optional
                output_p = tf.exp(logoutput_p)
                output_p = tf.reshape(output_p, (self.batch_size, 1, 1))
                empty_logits = tf.exp(tf.get_variable('empty-multiplier', shape=(), dtype=tf.float32,
                                                      initializer=tf.initializers.constant(0.0))) * empty_answer
                logits = output_p * logits + (1 - output_p) * empty_logits

            mask = tf.logical_not(tf.equal(targets, self.train.pad_idx))  # (bs, seq)
            label_cross_entropy = tf.reduce_sum(
                tf.losses.sparse_softmax_cross_entropy(targets, logits, reduction=Reduction.NONE) * tf.to_float(mask),
                axis=1) / tf.reduce_sum(tf.to_float(mask), axis=1)

            chars = tf.argmax(logits, axis=2, output_type=tf.int32)
            equal = tf.equal(targets, chars)
            correct = tf.to_float(tf.reduce_all(tf.logical_or(equal, tf.logical_not(mask)), axis=1))

            return {'cross_entropy': label_cross_entropy, 'actual': chars, 'targets': targets, 'correct': correct}

    def iterator(self, data, n_repeat=-1):
        shapes, types = data.shapes_types()
        ds = tf.data.Dataset.from_generator(
            data.sample_generator,
            types,
            shapes
        ).map(lambda i, v, s, *args: (tf.SparseTensor(i, v, s),) + args) \
            .repeat(n_repeat) \
            .apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)) \
            .prefetch(2)
        return ds.make_one_shot_iterator()

    def train_batch(self):
        batch = self.session.run(self.next_train_batch)
        placeholders = self.get_placeholders(batch, True)
        _, loss, outputs, step = self.session.run([self.train_step, self.loss, self.outputs, self.global_step],
                                                  placeholders)
        return loss

    def val_batch(self):
        batch = self.session.run(self.next_valid_batch)
        placeholders = self.get_placeholders(batch, False)
        loss, outputs, step = self.session.run([self.loss, self.outputs, self.global_step], placeholders)
        return loss

    def test_set(self, out_path="./predictions/"):
        actuals = []
        while True:
            try:
                batch = self.session.run(self.next_test_batch)
                placeholders = self.get_placeholders(batch, False)
                output = self.session.run(self.outputs, placeholders)
                actuals.extend(self.train.array_to_str(output[self.field]['actual']))
            except tf.errors.OutOfRangeError:
                break

        os.makedirs(out_path, exist_ok=True)
        extracts = {}
        for actual, filename in zip(actuals, self.test.filenames):
            print("Prediciton: {}\t\tFilename: {}".format(actual, filename))
            predictions = {}
            if os.path.exists(os.path.join(out_path, filename)):
                with open(os.path.join(out_path, filename), 'r') as fp:
                    predictions = json.load(fp)
            with open(os.path.join(out_path, filename), 'w') as fp:
                predictions[self.field] = actual
                fp.write(json.dumps(predictions))
            extracts[filename] = predictions
        print("Predictions stored in '{}'".format(out_path))
        return extracts

    def save(self, name):
        self.saver.save(self.session, "./models/invoicenet/%s/%s" % (self.field, name))

    def load(self, name):
        self.saver.restore(self.session, name)

    def get_placeholders(self, batch, is_training):
        memories, pixels, word_indices, pattern_indices, char_indices, memory_mask, parses, vendorname, invoicedate, invoicenumber, amountnet, amounttax, amounttotal, vatrate, vatid, taxid, iban, bic, found = batch
        return {
            self.is_training_ph: is_training,
            self.memories_ph: memories,
            self.pixels_ph: pixels,
            self.word_indices_ph: word_indices,
            self.pattern_indices_ph: pattern_indices,
            self.char_indices_ph: char_indices,
            self.memory_mask_ph: memory_mask,
            self.parses_ph: parses,
            self.vendorname_ph: vendorname,
            self.invoicedate_ph: invoicedate,
            self.invoicenumber_ph: invoicenumber,
            self.amountnet_ph: amountnet,
            self.amounttax_ph: amounttax,
            self.amounttotal_ph: amounttotal,
            self.vatrate_ph: vatrate,
            self.vatid_ph: vatid,
            self.taxid_ph: taxid,
            self.iban_ph: iban,
            self.bic_ph: bic,
            self.found_ph: found
        }
