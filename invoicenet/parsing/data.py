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

import random
from decimal import Decimal

import tensorflow as tf
from tensorflow.python.util import deprecation

from ..acp.data import RealData

deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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


class ParseData(Data):
    chars = RealData.chars
    pad_idx = RealData.pad_idx
    eos_idx = RealData.eos_idx
    unk_idx = RealData.unk_idx
    input_length = RealData.seq_in

    def __init__(self, samples_fname, output_length):
        self.samples_fname = samples_fname
        self.output_dict = UnkDict(self.chars)
        self.n_output = len(self.output_dict)
        self.output_length = output_length

    def types(self):
        # source, target
        return tf.int32, tf.int32

    def shapes(self):
        # source, target
        return self.input_length, self.output_length

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

    def normalize(self, str):
        return '{:f}'.format(Decimal(str).normalize())

    def _encode_str(self, field, max_length):
        encoded = [self.output_dict[c] for c in list(field)[:max_length - 1]] + [self.eos_idx]
        encoded += [self.pad_idx] * (max_length - len(encoded))

        return encoded


class TabSeparated(ParseData):
    def sample_generator(self):
        with open(self.samples_fname) as samples_file:
            samples = samples_file.readlines()

        while True:
            for s in random.sample(samples, len(samples)):
                source, target = s.strip().split("\t")
                yield self._encode_str(source, self.input_length), self._encode_str(target, self.output_length)
