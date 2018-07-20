import os
import pickle
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import L1L2
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(1337)


class InvoiceNet:

    def __init__(self, data_handler, config):
        coordinates = Input(shape=(data_handler.train_data['coordinates'].shape[1],),
                            dtype='float32', name='coordinates')
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

        tensorboard = TensorBoard(log_dir=self.config.log_dir, histogram_freq=1, write_graph=True)
        modelcheckpoints = ModelCheckpoint(os.path.join(self.config.checkpoint_dir, "InvoiceNet") +
                                           ".{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5",
                                           monitor='val_loss', verbose=0, save_best_only=True,
                                           save_weights_only=False, mode='auto')

        class_weights = compute_class_weight('balanced', np.unique(self.data_handler.train_data['labels']),
                                             self.data_handler.train_data['labels'])
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
        predictions = self.model.predict([self.data_handler.train_data['inputs'],
                                          self.data_handler.train_data['coordinates']], verbose=True)
        predictions = predictions.argmax(axis=-1)
        acc = np.sum(predictions == (self.data_handler.train_data['labels']) /
                     float(len(self.data_handler.train_data['labels'])))
        print("\nTest Accuracy: {}".format(acc))
        return predictions

    @staticmethod
    def get_precision(predictions, true_labels, target_label):
        target_label_count = 0
        correct_target_label_count = 0

        for idx in range(len(predictions)):
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
        for target_label in range(0, max(self.data_handler.train_data['labels'])):
            precision = self.get_precision(predictions, self.data_handler.train_data['labels'], target_label)
            recall = self.get_precision(self.data_handler.train_data['labels'], predictions, target_label)
            f1 = 0 if (precision+recall) == 0 else 2*precision*recall/(precision+recall)
            f1_sum += f1
            f1_count += 1

        macrof1 = f1_sum / float(f1_count)
        print("\nMacro-Averaged F1: %.4f\n" % macrof1)
        return macrof1


class InvoiceNetCloudScan:

    def __init__(self, config):
        features = Input(shape=(config.num_input*5,), dtype='float32', name='features')

        if config.num_layers == 2:
            output = Dense(config.num_hidden, activation='relu')(features)
            output = Dropout(0.5)(output)
        else:
            output = features
        output = Dense(config.num_output,
                       activation='softmax',
                       kernel_regularizer=L1L2(l1=0.0, l2=0.1))(output)
        self.model = Model(inputs=[features], outputs=[output])
        self.model.compile(optimizer='Adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        print(self.model.summary())
        self.config = config

    def train(self, data):
        print("\nInitializing training...")

        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)

        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        if not os.path.exists(self.config.model_path):
            os.makedirs(self.config.model_path)

        x_train, y_train = self.prepare_data(data)

        tensorboard = TensorBoard(log_dir=self.config.log_dir, histogram_freq=1, write_graph=True)
        modelcheckpoints = ModelCheckpoint(os.path.join(self.config.checkpoint_dir, "InvoiceNetCloudScan") +
                                           ".{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5",
                                           monitor='val_loss', verbose=0, save_best_only=True,
                                           save_weights_only=False, mode='auto')

        classes = np.unique(data['label'].values)
        class_weights = compute_class_weight('balanced', classes, data['label'].values)
        d_class_weights = dict(enumerate(class_weights))

        self.model.fit([x_train], y_train,
                       batch_size=self.config.batch_size,
                       verbose=True,
                       epochs=self.config.epochs,
                       callbacks=[tensorboard, modelcheckpoints],
                       validation_split=0.125,
                       shuffle=self.config.shuffle,
                       class_weight=d_class_weights)

        self.model.save_weights(os.path.join(self.config.model_path, "InvoiceNetCloudScan.model"))

    def prepare_data(self, data):
        if self.config.mode in 'train':
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(data.processed_text)
            with open('data/tokenizer.pk', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=3)
        else:
            with open('data/tokenizer.pk', 'rb') as handle:
                tokenizer = pickle.load(handle)
        seq = tokenizer.texts_to_sequences(data.processed_text)
        padded_seq = pad_sequences(seq, maxlen=4)
        feature_list = ['length', 'line_size', 'position_on_line', 'has_digits', 'bottom_margin', 'top_margin',
                        'left_margin', 'right_margin', 'page_width', 'page_height', 'parses_as_amount',
                        'parses_as_date', 'parses_as_number']
        features = np.zeros([padded_seq.shape[0], len(feature_list)], dtype=np.float32)
        for i in range(len(feature_list)):
            features[:, i] = data[feature_list[i]].values
        features = np.concatenate((padded_seq, features), axis=1)

        spatial_features = np.zeros([features.shape[0], features.shape[1]*4], dtype=np.float32)

        zero_vec = np.zeros(features.shape[1], dtype=np.float32)
        for i in range(features.shape[0]):
            vectors = [zero_vec if j == -1 else features[j] for j in data.at[i, 'closest_ngrams']]
            spatial_features[i, :] = np.concatenate(vectors)

        features = np.concatenate((features, spatial_features), axis=1)

        if self.config.oversample == 0:
            return features, data['label'].values
        else:
            features = np.concatenate((features,
                                       np.repeat(features[data['label'].values[data['label'].values != 0]]
                                                 , self.config.oversample, axis=0)),
                                      axis=0)
            labels = np.concatenate((data['label'].values,
                                     np.repeat(data['label'].values[data['label'].values != 0],
                                               self.config.oversample, axis=0)),
                                    axis=0)
            return features, labels


    def load_weights(self, path):
        """Loads weights from the given model file"""
        self.model.load_weights(path)
        print("\nSuccessfully loaded weights from {}".format(path))

    # def predict(self, tokens, coordinates):
    #     """Performs inference on the given tokens and coordinates"""
    #     inp, coords = self.data_handler.process_data(tokens, coordinates)
    #     pred = self.model.predict([inp, coords], verbose=True)
    #     pred = pred.argmax(axis=-1)
    #     return pred

    def evaluate(self, data):
        x_test, y_test = self.prepare_data(data)
        predictions = self.model.predict([x_test], verbose=True)
        predictions = predictions.argmax(axis=-1)
        acc = np.sum(predictions == y_test) / float(len(y_test))
        print("\nTest Accuracy: {}".format(acc))
        return predictions

    @staticmethod
    def get_precision(predictions, true_labels, target_label):
        target_label_count = 0
        correct_target_label_count = 0

        for idx in range(len(predictions)):
            if predictions[idx] == target_label:
                target_label_count += 1
                if predictions[idx] == true_labels[idx]:
                    correct_target_label_count += 1

        if correct_target_label_count == 0:
            return 0
        return float(correct_target_label_count) / target_label_count

    def f1_score(self, predictions, ground_truth):
        f1_sum = 0
        f1_count = 0
        for target_label in range(0, max(ground_truth)):
            precision = self.get_precision(predictions, ground_truth, target_label)
            recall = self.get_precision(ground_truth, predictions, target_label)
            f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
            f1_sum += f1
            f1_count += 1

        macrof1 = f1_sum / float(f1_count)
        print("\nMacro-Averaged F1: %.4f\n" % macrof1)
        return macrof1
