import numpy as np
import os
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras import regularizers


np.random.seed(1337)


class InvoiceNet:

    def __init__(self, data_handler, config):
        midpoints = Input(shape=(2,), dtype='float32', name='midpoints')
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
        output = concatenate([output, midpoints])
        output = Dense(config.num_hidden, activation='relu')(output)
        output = Dropout(0.5)(output)
        output = Dense(data_handler.num_classes, activation='softmax')(output)

        self.model = Model(inputs=[words_input, midpoints], outputs=[output])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        self.model.summary()
        self.data_handler = data_handler
        self.config = config

    def train(self):
        print("Start training")

        tensorboard = keras.callbacks.TensorBoard(log_dir=self.config.log_dir, histogram_freq=1, write_graph=True)
        modelcheckpoints = keras.callbacks.ModelCheckpoint(os.path.join(self.config.checkpoint_dir, "InvoiceNet_") +
                                                           ".{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5",
                                                           monitor='val_loss', verbose=0, save_best_only=False,
                                                           save_weights_only=False, mode='auto')

        self.model.fit([self.data_handler.train_data['inputs'], self.data_handler.train_data['midpoints']],
                       self.data_handler.train_data['labels'],
                       batch_size=self.config.batch_size,
                       verbose=True,
                       epochs=self.config.num_epoch,
                       callbacks=[tensorboard, modelcheckpoints],
                       validation_split=0.125,
                       shuffle=self.config.shuffle)

        self.model.save_weights(os.path.join(self.config.model_path, "InvoiceNet.model"))
