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

import re
import glob
import hashlib
import json
import random
import string
import pdf2image

import numpy as np
import tensorflow as tf
from PIL import Image
from decimal import Decimal

from .. import FIELDS, FIELD_TYPES
from ..common import util
from ..common.data import Data, UnkDict

random.seed(0)


class InvoiceData(Data):

    im_size = 128, 128
    chars = ['<PAD>', '<EOS>', '<UNK>'] + list(string.printable)
    output_dict = UnkDict(chars)
    n_output = len(output_dict)
    pad_idx = 0
    eos_idx = 1
    unk_idx = 2
    word_hash_size = 2 ** 16
    pattern_hash_size = 2 ** 14
    seq_in = 4 * 32
    seq_amount = 16
    seq_date = 11
    seq_long = 128

    seq_out = {
        FIELD_TYPES["general"]: seq_long,
        FIELD_TYPES["optional"]: seq_long,
        FIELD_TYPES["amount"]: seq_amount,
        FIELD_TYPES["date"]: seq_date
    }

    n_memories = 4
    parses_idx = {'date': 0, 'amount': 1}

    def __init__(self, field, data_dir=None):
        self.field = field
        self.filenames = []
        if data_dir:
            self.filenames = glob.glob(data_dir + "**/*.json", recursive=True)

    def types(self):
        return (
            tf.int64,  # i
            tf.float32,  # v
            tf.int64,  # s
            tf.float32,  # pixels
            tf.int32,  # word_indices
            tf.int32,  # pattern_indices
            tf.int32,  # char_indices
            tf.float32,  # memory_mask
            tf.float32,  # parses
            tf.int32  # target
        )

    def shapes(self):
        return (
            (None, 5),  # i
            (None,),  # v
            (None,),  # s
            InvoiceData.im_size + (3,),  # pixels
            InvoiceData.im_size,  # word_indices
            InvoiceData.im_size,  # pattern_indices
            InvoiceData.im_size,  # char_indices
            InvoiceData.im_size,  # memory_mask
            InvoiceData.im_size + (self.n_memories, 2),  # parses
            (InvoiceData.seq_out[FIELDS[self.field]],)  # target
        )

    def _encode_ngrams(self, n_grams, height, width):
        v_ar = self.im_size[0] / height
        h_ar = self.im_size[1] / width

        max_v = self.im_size[0] - 1
        max_h = self.im_size[1] - 1

        pattern_indices = np.zeros(self.im_size)
        word_indices = np.zeros(self.im_size, np.int32)
        char_indices = np.zeros(self.im_size, np.int32)
        memory_mask = np.zeros(self.im_size, np.float32)

        parses = np.zeros(self.im_size + (self.n_memories, 2))
        memory_indices = []
        for n_gram in n_grams:
            words = n_gram["words"]
            m_idx = len(words) - 1
            word = words[0]

            left = min(round(word['left'] * h_ar), max_h)
            right = min(round(word['right'] * h_ar), max_h)
            top = min(round(word['top'] * v_ar), max_v)
            bottom = min(round(word['bottom'] * v_ar), max_v)

            for k, v in n_gram['parses'].items():
                parses[top:bottom + 1, left:right + 1, m_idx, self.parses_idx[k]] = 1.0

            chars = " ".join([w['text'] for w in words])[:self.seq_in - 1]
            char_idx = [self.output_dict[c] for c in chars] + [self.eos_idx]
            char_pos = range(len(char_idx))

            self.append_indices(top, bottom, left, right, m_idx, char_idx, char_pos, memory_indices)

            if len(words) == 1:
                text = words[0]['text']
                memory_mask[top, left] = 1.0

                pattern = text
                pattern = re.sub(r"[A-Z]", "X", pattern)
                pattern = re.sub(r"[a-z]", "x", pattern)
                pattern = re.sub(r"[0-9]", "0", pattern)
                pattern = re.sub(r"[^Xx0]", "-", pattern)

                pattern_idx = (int(hashlib.md5(str.encode(pattern)).hexdigest(), 16) % (self.pattern_hash_size - 1)) + 1
                pattern_indices[top:bottom + 1, left:right + 1] = pattern_idx

                w_idx = (int(hashlib.md5(str.encode(text)).hexdigest(), 16) % (self.word_hash_size - 1)) + 1
                word_indices[top:bottom + 1, left:right + 1] = w_idx

                for cidx, p in zip(char_idx[:-1], np.linspace(left, right, len(char_idx[:-1]))):
                    char_indices[top:bottom + 1, int(round(p))] = cidx

        assert len(memory_indices) > 0
        memory_values = [1.] * len(memory_indices)
        memory_dense_shape = self.im_size + (self.n_memories, self.seq_in, self.n_output)

        return (
            word_indices,
            pattern_indices,
            char_indices,
            memory_mask,
            parses,
            memory_indices,
            memory_values,
            memory_dense_shape
        )

    def append_indices(self, top, bottom, left, right, m_idx, char_idx, char_pos, indices):
        assert 0 <= m_idx < self.n_memories, m_idx

        assert top <= bottom, (top, bottom)
        assert left <= right, (left, right)
        assert 0 <= top < self.im_size[0], top
        assert 0 <= bottom < self.im_size[0], bottom
        assert 0 <= left < self.im_size[1], left
        assert 0 <= right < self.im_size[1], right

        for cp_idx, ci_idx in zip(char_pos, char_idx):
            assert 0 <= cp_idx < self.seq_in, cp_idx
            assert 0 <= ci_idx < self.n_output, ci_idx

        for cp_idx, ci_idx in zip(char_pos, char_idx):
            indices.append((top, left, m_idx, cp_idx, ci_idx))

    def encode_image(self, page):
        im = Image.open(page["filename"])
        im = im.convert('RGB').resize(self.im_size[::-1], Image.ANTIALIAS)
        pixels = (np.asarray(im, np.float32) / 255. - 0.5) * 2.
        return pixels

    @staticmethod
    def _preprocess_amount(value):
        return '{:f}'.format(Decimal(value).normalize())

    def _load_document(self, doc_id):
        with open(doc_id, encoding="utf8") as fp:
            page = json.load(fp)

        pixels = self.encode_image(page)
        n_grams = page['nGrams']

        word_indices, pattern_indices, char_indices, memory_mask, parses, i, v, s = self._encode_ngrams(n_grams,
                                                                                                        page['height'],
                                                                                                        page['width'])

        target = page['fields'][self.field]
        if FIELDS[self.field] == FIELD_TYPES["amount"]:
            target = self._preprocess_amount(target)
        target = InvoiceData.encode_sequence(target, self.seq_out[FIELDS[self.field]])

        return i, v, s, pixels, word_indices, pattern_indices, char_indices, memory_mask, parses, target

    def array_to_str(self, arr):
        """
        :param arr: (bs, seq) int32
        """
        strs = []
        for r in arr:
            s = ""
            for c in r:
                if c == self.eos_idx:
                    break
                else:
                    s += self.output_dict.idx2key(c)
            strs.append(s)
        return strs

    def sample_generator(self):
        exceptions = 0
        np.random.seed(0)
        random.shuffle(self.filenames)

        for i, doc_id in enumerate(self.filenames):
            try:
                yield self._load_document(doc_id.strip())
            except GeneratorExit:
                return
            except Exception as exp:
                print("Exception: {} : {}".format(doc_id, exp))
                exceptions += 1

    def _process_pdf(self, path):
        pixels = pdf2image.convert_from_path(path)[0]
        height = pixels.size[1]
        width = pixels.size[0]

        ngrams = util.create_ngrams(pixels, height, width)
        for ngram in ngrams:
            if "amount" in ngram["parses"]:
                ngram["parses"]["amount"] = util.normalize(ngram["parses"]["amount"], key="amount")
            if "date" in ngram["parses"]:
                ngram["parses"]["date"] = util.normalize(ngram["parses"]["date"], key="date")

        page = {
            "nGrams": ngrams,
            "height": height,
            "width": width,
            "filename": path
        }

        pixels = pixels.convert('RGB').resize(self.im_size[::-1], Image.ANTIALIAS)
        pixels = (np.asarray(pixels, np.float32) / 255. - 0.5) * 2.

        n_grams = page['nGrams']

        word_indices, pattern_indices, char_indices, memory_mask, parses, i, v, s = self._encode_ngrams(n_grams,
                                                                                                        page['height'],
                                                                                                        page['width'])

        return i, v, s, pixels, word_indices, pattern_indices, char_indices, memory_mask, parses

    def generate_test_data(self, paths: list):
        if not isinstance(paths, list):
            raise Exception("This function assumes the input is a list of paths")

        def _generator():
            exceptions = 0
            for idx, path in enumerate(paths):
                try:
                    yield self._process_pdf(path)
                except Exception as exp:
                    print("Exception: {} : {}".format(path, exp))
                    exceptions += 1

        return _generator

    @staticmethod
    def encode_sequence(value, max_len):
        encoded = [InvoiceData.output_dict[c] for c in list(value)[:max_len - 1]] + [InvoiceData.eos_idx]
        encoded += [InvoiceData.pad_idx] * (max_len - len(encoded))
        return encoded

    @staticmethod
    def create_dataset(data_dir, field, batch_size):
        data = InvoiceData(field=field, data_dir=data_dir)
        shapes, types = data.shapes(), data.types()

        def _transform(i, v, s, *args):
            return (tf.SparseTensor(i, v, s),) + args

        return tf.data.Dataset.from_generator(
            data.sample_generator,
            types,
            shapes
        ).map(_transform) \
            .repeat(-1) \
            .batch(batch_size=batch_size, drop_remainder=True) \
            .prefetch(2)
