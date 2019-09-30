import os
import cv2
import json
import random
import pdf2image
import numpy as np
import tensorflow as tf
import pytesseract
from pytesseract import Output

from tools.utils import Parser


class InvoiceData:

    def __init__(self, height, width, char_vocab, pattern_vocab, ngram_length=4, seq_length=128):
        self.height = height
        self.width = width
        self.ngram_length = ngram_length
        self.seq_length = seq_length
        self.char_vocab = char_vocab
        self.pattern_vocab = pattern_vocab
        self.parser = Parser()
        self.filenames = []
        self.labels = []

    def load_data_dir(self, filenames, key):
        for filename in filenames:
            with open(filename, 'r') as fp:
                labels = json.loads(fp.read())
            self.filenames.append(os.path.join(os.path.dirname(filename), labels['filename']))
            if isinstance(labels[key], list):
                self.labels.append(labels[key][0])
            else:
                self.labels.append(labels[key])

    def load_file(self, path):
        if path.endswith('.pdf'):
            page = pdf2image.convert_from_path(path)[0]
            path = path[:-3] + 'jpg'
            page.save(path, 'JPEG')
        self.filenames.append(path)
        self.labels.append('')

    def load_files(self, paths):
        for path in paths:
            self.load_file(path)

    def get_indices(self, ngrams):
        indices = []
        mask = np.zeros((self.height, self.width, self.ngram_length), dtype=np.float32)
        for ngram in ngrams:
            x = int(self.width * ngram['x'])
            y = int(self.height * ngram['y'])
            text = ngram['text'][:self.seq_length]
            text = list(text) + ['<pad>'] * (self.seq_length - len(text))
            mask[y, x, ngram['length'] - 1] = 1.0
            for i, c in enumerate(text):
                indices.append([y, x, ngram['length'] - 1, i, self.char_vocab[c]])
        return indices, mask

    def create_memory(self, ngrams):
        memory_idx, mask = self.get_indices(ngrams)
        return memory_idx, mask

    @staticmethod
    def extract_words(img):
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(data['level'])
        words = [{'word': data['text'][i].lower(),
                  'x': data['left'][i] / img.shape[1],
                  'y': data['top'][i] / img.shape[0],
                  'w': data['width'][i] / img.shape[1],
                  'h': data['height'][i] / img.shape[0]}
                 for i in range(n_boxes) if data['text'][i]]
        return words

    @staticmethod
    def divide_into_lines(words):
        cur = words[0]
        lines = []
        line = []
        for word in words:
            if word['y'] - cur['y'] > 0.005:
                # if difference between y-coordinate of current word and previous word
                # is more than 0.5% of the height, consider the current word to be in the next line
                lines.append(line)
                line = [word]
            elif word['x'] - (cur['x'] + cur['w']) > 0.05:
                # if difference between x-coordinate of current word and previous word
                # is more than 5% of the width, consider the current word to be in a different line
                lines.append(line)
                line = [word]
            else:
                line.append(word)
            cur = word
        lines.append(line)
        return lines

    @staticmethod
    def create_ngrams(words, length=4):
        lines = InvoiceData.divide_into_lines(words)
        tokens = [line[i:i + N] for line in lines for N in range(1, length + 1) for i in range(len(line) - N + 1)]
        ngrams = [{'text': ' '.join([word['word'] for word in token]),
                   'length': len(token),
                   'x': token[0]['x'],
                   'y': token[0]['y'],
                   'h': token[0]['h'],
                   'w': (token[-1]['x'] - token[0]['x']) + token[-1]['w']}
                  for token in tokens]
        return ngrams

    def _create_char_position_matrix(self, img):
        chars = pytesseract.image_to_boxes(img, output_type=Output.DICT)
        char_mat = np.zeros((self.height, self.width), dtype=np.int32)
        for char, x1, y1, x2, y2 in zip(chars['char'], chars['left'], chars['top'], chars['right'], chars['bottom']):
            x1 = int((x1 / img.shape[1]) * self.width)
            x2 = int((x2 / img.shape[1]) * self.width) + 1
            y1 = int(((img.shape[0] - y1) / img.shape[0]) * self.height)
            y2 = int(((img.shape[0] - y2) / img.shape[0]) * self.height) + 1
            char_mat[y1:y2, x1:x2] = self.char_vocab[char.lower()]
        return char_mat

    def _create_pattern_position_matrix(self, words):
        pattern_mat = np.zeros((self.height, self.width), dtype=np.int32)
        for word in words:
            x1 = int(word['x'] * self.width)
            y1 = int(word['y'] * self.height)
            x2 = int((word['x'] + word['w']) * self.width) + 1
            y2 = int((word['y'] + word['h']) * self.height) + 1
            p = ''
            for c in word['word']:
                if c.isalpha():
                    p += 'x'
                elif c.isnumeric():
                    p += '0'
                else:
                    p += '.'
            pattern_mat[y1:y2, x1:x2] = self.pattern_vocab[p]
        return pattern_mat

    def _create_feature_position_matrix(self, ngrams):
        feature_mat = np.zeros((self.height, self.width, 5), dtype=np.float32)
        for ngram in ngrams:
            x = int(self.width * ngram['x'])
            y = int(self.height * ngram['y'])
            feature_mat[y, x, 0] = 1.0 if self.parser.parse(ngram['text'], 'amount') else 0.0
            feature_mat[y, x, 1] = 1.0 if self.parser.parse(ngram['text'], 'date') else 0.0
            feature_mat[y, x, 2] = ngram['x']
            feature_mat[y, x, 3] = ngram['y']
            feature_mat[y, x, 4] = 1.0
        return feature_mat

    def create_feature_representation(self, img, words, ngrams):
        char_mat = self._create_char_position_matrix(img)
        pattern_mat = self._create_pattern_position_matrix(words)
        feature_mat = self._create_feature_position_matrix(ngrams)
        img = cv2.resize(img, (self.width, self.height)).astype(np.float32)
        return {
            "image": img,
            "pattern_idx": pattern_mat,
            "char_idx": char_mat,
            "features": feature_mat
        }

    def encode_label(self, label):
        if type(label) != str:
            label = str(label)
        return np.array([[self.char_vocab[c]] for c in label] + [[0]] * (self.seq_length - len(label)),
                        dtype=np.int64)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        invoice = np.array(pdf2image.convert_from_path(self.filenames[idx])[0].convert('RGB'))
        words = InvoiceData.extract_words(invoice)
        ngrams = InvoiceData.create_ngrams(words, length=self.ngram_length)
        memory_idx, mask = self.create_memory(ngrams)
        features = self.create_feature_representation(invoice, words, ngrams)
        label = self.encode_label(self.labels[idx])
        return (memory_idx, mask, features["image"], features["pattern_idx"],
                features["char_idx"], features["features"], label)


def get_data(key, data_path, char_vocab, pattern_vocab, training=True,
             height=128, width=128, ngram_length=4, seq_length=128, batch_size=8, shuffle=True):

    data = InvoiceData(height=height, width=width, ngram_length=ngram_length, seq_length=seq_length,
                       char_vocab=char_vocab, pattern_vocab=pattern_vocab)

    if training:
        data.load_data_dir(filenames=data_path, key=key)
    else:
        data.load_files(data_path)

    def _gen_series():
        if not shuffle:
            i = 0
            while True:
                yield data[i]
                i += 1
                if i == len(data):
                    i = 0
        else:
            idx = list(range(len(data)))
            while True:
                random.shuffle(idx)
                for i in idx:
                    yield data[i]

    def _to_sparse_tensor(memory_idx, *args):
        inputs = (tf.SparseTensor(indices=memory_idx,
                                  values=tf.ones([tf.shape(memory_idx)[0]], dtype=tf.float32),
                                  dense_shape=[height, width, ngram_length, seq_length, len(char_vocab)]),
                  *args[:-1])
        return inputs, args[-1]     # output not used but needed

    dataset = tf.data.Dataset.from_generator(
        _gen_series,
        output_types=(tf.int64, tf.float32, tf.float32, tf.int32, tf.int32, tf.float32, tf.int32),
        output_shapes=([None, 5], [height, width, ngram_length], [height, width, 3],
                       [height, width], [height, width], [height, width, 5], [seq_length, 1])) \
        .map(_to_sparse_tensor) \
        .batch(batch_size)

    if training:
        dataset = dataset.repeat()

    return dataset, len(data)
