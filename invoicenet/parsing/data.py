import random
from decimal import Decimal

import tensorflow as tf

from invoicenet.acp.data import RealData


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


if __name__ == '__main__':
    print("Dates")
    data = TabSeparated('data/date/train.tsv', RealData.seq_date)
    g = data.sample_generator()
    for i in range(10):
        print(next(g))

    print("Amounts")
    data = TabSeparated('data/amount/train.tsv', RealData.seq_amount)
    g = data.sample_generator()
    for i in range(10):
        print(next(g))
