import os
import glob
import hashlib
import json
import logging
import random
import string
import time

import numpy as np
import re
import tensorflow as tf
from PIL import Image
from decimal import Decimal

random.seed(0)


class Data:
    def sample_generator(self):
        raise NotImplementedError

    def types(self):
        raise NotImplementedError

    def shapes(self):
        raise NotImplementedError

    def array_to_str(self, arr):
        raise NotImplementedError


class UnkDict:
    unk = '<UNK>'

    def __init__(self, items):
        if self.unk not in items:
            raise ValueError("items must contain %s", self.unk)

        self.delegate = dict([(c, i) for i, c in enumerate(items)])
        self.rdict = {i: c for c, i in self.delegate.items()}

    def __getitem__(self, item):
        if item in self.delegate:
            return self.delegate[item]
        else:
            return self.delegate[self.unk]

    def __len__(self):
        return len(self.delegate)

    def idx2key(self, idx):
        return self.rdict[idx]


class RealData(Data):
    im_size = 128, 128  # height, width TODO consider respecting aspect ratio
    chars = ['<PAD>', '<EOS>', '<UNK>'] + list(string.printable)  # TODO proper multi-lingual char set/bpe encoding
    pad_idx = 0
    eos_idx = 1
    unk_idx = 2
    word_hash_size = 2 ** 16
    pattern_hash_size = 2 ** 14
    seq_in = 4 * 32
    seq_long = 128
    seq_amount = 16
    seq_date = 11

    n_memories = 4
    parses_idx = {'date': 0, 'amount': 1}
    fields = [
        'number',
        'date',
        'amount'
    ]

    def __init__(self, data_dir):
        self.logger = logging.getLogger("data")
        self.data_dir = data_dir
        self.output_dict = UnkDict(self.chars)
        self.n_output = len(self.output_dict)
        self.filenames = []

    def shapes_types(self):
        return zip(*(
            ((None, 5), tf.int64),  # i
            ((None,), tf.float32),  # v
            ((None,), tf.int64),  # s
            (self.im_size + (3,), tf.float32),  # pixels
            (self.im_size, tf.int32),  # word_indices
            (self.im_size, tf.int32),  # pattern_indices
            (self.im_size, tf.int32),  # char_indices
            (self.im_size, tf.float32),  # memory_mask
            (self.im_size + (4, 2), tf.float32),  # parses
            ((self.seq_long,), tf.int32),  # number
            ((self.seq_date,), tf.int32),  # date
            ((self.seq_amount,), tf.int32),  # amount
            ((3,), tf.float32)  # found
        ))

    def _encode_ngrams(self, n_grams, height, width):
        v_ar = self.im_size[0] / height
        h_ar = self.im_size[1] / width

        max_v = self.im_size[0] - 1
        max_h = self.im_size[1] - 1

        pattern_indices = np.zeros(self.im_size)
        word_indices = np.zeros(self.im_size, np.int32)
        char_indices = np.zeros(self.im_size, np.int32)
        memory_mask = np.zeros(self.im_size, np.float32)

        parses = np.zeros(self.im_size + (4, 2))
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
                parses[top:bottom, left:right, m_idx, self.parses_idx[k]] = 1.0

            chars = " ".join([w['text'] for w in words])[:self.seq_in - 1]
            char_idx = [self.output_dict[c] for c in chars] + [self.eos_idx]
            char_pos = range(len(char_idx))

            self.append_indices(top, bottom, left, right, m_idx, char_idx, char_pos, memory_indices)

            if len(words) == 1:
                text = words[0]['text']
                memory_mask[top, left] = 1.0

                pattern = text
                pattern = re.sub(r"\\p{Lu}", "X", pattern)
                pattern = re.sub(r"\\p{Ll}", "x", pattern)
                pattern = re.sub(r"\\p{N}", "0", pattern)
                pattern = re.sub(r"[^Xx0]", "-", pattern)

                pattern_idx = (int(hashlib.md5(str.encode(pattern)).hexdigest(), 16) % (self.pattern_hash_size - 1)) + 1
                pattern_indices[top:bottom, left:right] = pattern_idx

                w_idx = (int(hashlib.md5(str.encode(text)).hexdigest(), 16) % (self.word_hash_size - 1)) + 1
                word_indices[top:bottom, left:right] = w_idx

                for cidx, p in zip(char_idx[:-1], np.linspace(left, right - 1, len(char_idx[:-1]))):
                    char_indices[top:bottom, int(round(p))] = cidx

        assert len(memory_indices) > 0
        memory_values = [1.] * len(memory_indices)
        memory_dense_shape = self.im_size + (self.n_memories, self.seq_in, self.n_output)

        return word_indices, pattern_indices, char_indices, memory_mask, parses, memory_indices, memory_values, memory_dense_shape

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

    def _load_document(self, doc_id):
        if not doc_id.endswith('.json'):
            doc_id += ".json"

        with open(self.data_dir + doc_id, encoding="utf8") as fp:
            page = json.load(fp)

        pixels = self.encode_image(page)
        n_grams = page['nGrams']

        def preprocess_amount(value):
            return '{:f}'.format(Decimal(value).normalize())

        strings_in_doc = {""}
        for n in n_grams:
            s = " ".join([w["text"] for w in n["words"]])
            strings_in_doc.add(s)
            for k, p in n["parses"].items():
                strings_in_doc.add(p)
                if k == "amount":
                    strings_in_doc.add(preprocess_amount(p))

        word_indices, pattern_indices, char_indices, memory_mask, parses, i, v, s = self._encode_ngrams(n_grams, page['height'], page['width'])

        fields = page['fields']
        amount_fields = {'amount'}
        fields = {k: (preprocess_amount(v) if k in amount_fields else v) for k, v in fields.items()}

        found = [1.0 if fields[f] in strings_in_doc else 0.0 for f in self.fields]

        number = self._encode_sequence(fields['number'], self.seq_long)
        date = self._encode_sequence(fields['date'], self.seq_date)
        total = self._encode_sequence(fields['amount'], self.seq_amount)

        return i, v, s, pixels, word_indices, pattern_indices, char_indices, memory_mask, parses, number, date, total, found

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
        doc_ids = [os.path.basename(f) for f in glob.glob(self.data_dir + "**/*.json", recursive=True)]

        exceptions = 0
        np.random.seed(0)
        random.shuffle(doc_ids)
        self.filenames = doc_ids

        spent_loading = 0
        for i, doc_id in enumerate(doc_ids):
            try:
                start = time.perf_counter()
                doc = self._load_document(doc_id.strip())
                spent_loading += (time.perf_counter() - start)
                if i % 100 == 0:
                    print("Data loader dps: %f" % ((i + 1) / spent_loading))
                yield doc
            except GeneratorExit:
                return
            except:
                self.logger.exception(doc_id + "%d/%d" % (exceptions, i))
                exceptions += 1

    def _encode_sequence(self, value, max_len):
        encoded = [self.output_dict[c] for c in list(value)[:max_len - 1]] + [self.eos_idx]
        encoded += [self.pad_idx] * (max_len - len(encoded))

        return encoded
