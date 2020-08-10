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

from ..common.data import Data, UnkDict
from ..acp.data import InvoiceData


class ParseData(Data):
    chars = InvoiceData.chars
    output_dict = UnkDict(chars)
    n_output = len(output_dict)
    pad_idx = InvoiceData.pad_idx
    eos_idx = InvoiceData.eos_idx
    unk_idx = InvoiceData.unk_idx
    input_length = InvoiceData.seq_in

    def __init__(self, samples_fname, output_length):
        self.samples_fname = samples_fname
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

    def sample_generator(self):
        with open(self.samples_fname) as samples_file:
            samples = samples_file.readlines()

        while True:
            for s in random.sample(samples, len(samples)):
                source, target = s.strip().split("\t")
                yield (InvoiceData.encode_sequence(source, self.input_length),
                       InvoiceData.encode_sequence(target, self.output_length))

    @staticmethod
    def create_dataset(path, output_length, batch_size):
        data = ParseData(path, output_length)

        def _transform(inputs, targets):
            return (
                (tf.one_hot(inputs, ParseData.n_output),
                 tf.zeros(
                     (128,),
                     dtype=tf.float32,
                     name="empty_context")
                 ), targets)

        return tf.data.Dataset.from_generator(
            data.sample_generator,
            data.types(),
            data.shapes()
        ).map(_transform) \
            .repeat(-1) \
            .batch(batch_size) \
            .prefetch(16)
