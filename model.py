import numpy as np
import os
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras import regularizers
from sklearn.utils.class_weight import compute_class_weight


np.random.seed(1337)


class InvoiceNet:

    def __init__(self, data_handler, config):
        coordinates = Input(shape=(data_handler.train_data['coordinates'].shape[1],), dtype='float32', name='coordinates')
        words_input = Input(shape=(data_handler.max_length,), dtype='int32', name='words_input')
        words = Embedding(data_handler.embeddings.shape[0], data_handler.embeddings.shape[1],
                          weights=[data_handler.embeddings],
                          trainable=False)(words_input)

        conv1 = Convolution1D(filters=config.num_filters,
                              kernel_size=3,
                              padding='same',
                              activation='relu',
                              strides=1,
                              kernel_regularizer=regularizers.l2(config.reg_rate))(words)
        pool1 = GlobalMaxPooling1D()(conv1)

        conv2 = Convolution1D(filters=config.num_filters,
                              kernel_size=4,
                              padding='same',
                              activation='relu',
                              strides=1,
                              kernel_regularizer=regularizers.l2(config.reg_rate))(words)
        pool2 = GlobalMaxPooling1D()(conv2)

        conv3 = Convolution1D(filters=config.num_filters,
                              kernel_size=5,
                              padding='same',
                              activation='relu',
                              strides=1,
                              kernel_regularizer=regularizers.l2(config.reg_rate))(words)
        pool3 = GlobalMaxPooling1D()(conv3)

        output = concatenate([pool1, pool2, pool3])
        output = Dropout(0.5)(output)
        output = concatenate([output, coordinates])
        output = Dense(config.num_hidden, activation='relu')(output)
        output = Dropout(0.5)(output)
        output = Dense(data_handler.num_classes, activation='softmax')(output)

        self.model = Model(inputs=[words_input, coordinates], outputs=[output])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        # self.model.summary()
        self.data_handler = data_handler
        self.config = config

    def train(self):
        print("\nInitializing training...")

        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)

        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        if not os.path.exists(self.config.model_path):
            os.makedirs(self.config.model_path)

        tensorboard = keras.callbacks.TensorBoard(log_dir=self.config.log_dir, histogram_freq=1, write_graph=True)
        modelcheckpoints = keras.callbacks.ModelCheckpoint(os.path.join(self.config.checkpoint_dir, "InvoiceNet_") +
                                                           ".{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5",
                                                           monitor='val_loss', verbose=0, save_best_only=True,
                                                           save_weights_only=False, mode='auto')

        class_weights = compute_class_weight('balanced', np.unique(self.data_handler.train_data['labels']), self.data_handler.train_data['labels'])
        d_class_weights = dict(enumerate(class_weights))

        self.model.fit([self.data_handler.train_data['inputs'], self.data_handler.train_data['coordinates']],
                       self.data_handler.train_data['labels'],
                       batch_size=self.config.batch_size,
                       verbose=True,
                       epochs=self.config.num_epochs,
                       callbacks=[tensorboard, modelcheckpoints],
                       validation_split=0.125,
                       shuffle=self.config.shuffle,
                       class_weight=d_class_weights)

        self.model.save_weights(os.path.join(self.config.model_path, "InvoiceNet.model"))

    def load_weights(self, path):
        """Loads weights from the given model file"""
        self.model.load_weights(path)
        print("\nSuccessfully loaded weights from {}".format(path))

    def predict(self, tokens, coordinates):
        """Performs inference on the given tokens and coordinates"""
        inp, coords = self.data_handler.process_data(tokens, coordinates)
        pred = self.model.predict([inp, coords], verbose=True)
        pred = pred.argmax(axis=-1)
        return pred

    def evaluate(self):
        predictions = self.model.predict([self.data_handler.train_data['inputs'], self.data_handler.train_data['coordinates']], verbose=True)
        predictions = predictions.argmax(axis=-1)
        acc = np.sum(predictions == self.data_handler.train_data['labels']) / float(len(self.data_handler.train_data['labels']))
        print("\nTest Accuracy: {}".format(acc))
        return predictions

    @staticmethod
    def get_precision(predictions, true_labels, target_label):
        target_label_count = 0
        correct_target_label_count = 0

        for idx in xrange(len(predictions)):
            if predictions[idx] == target_label:
                target_label_count += 1
                if predictions[idx] == true_labels[idx]:
                    correct_target_label_count += 1

        if correct_target_label_count == 0:
            return 0
        return float(correct_target_label_count) / target_label_count

    def f1_score(self, predictions):
        f1_sum = 0
        f1_count = 0
        for target_label in xrange(0, max(self.data_handler.train_data['labels'])):
            precision = self.get_precision(predictions, self.data_handler.train_data['labels'], target_label)
            recall = self.get_precision(self.data_handler.train_data['labels'], predictions, target_label)
            f1 = 0 if (precision+recall) == 0 else 2*precision*recall/(precision+recall)
            f1_sum += f1
            f1_count += 1

        macrof1 = f1_sum / float(f1_count)
        print("\nMacro-Averaged F1: %.4f\n" % macrof1)
        return macrof1

