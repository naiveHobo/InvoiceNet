from decimal import Decimal

import torch
from torch.utils.data import Dataset

from ..acp.data import RealData
from ..common.util import UnkDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ParseData(Dataset):
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
        with open(self.samples_fname) as samples_file:
            self.samples = samples_file.readlines()

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

    @staticmethod
    def normalize(text):
        return '{:f}'.format(Decimal(text).normalize())

    def _encode_str(self, field, max_length):
        encoded = [self.output_dict[c] for c in list(field)[:max_length - 1]] + [self.eos_idx]
        encoded += [self.pad_idx] * (max_length - len(encoded))
        return torch.LongTensor(encoded)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source, target = self.samples[idx].strip().split("\t")
        return self._encode_str(source, self.input_length), self._encode_str(target, self.output_length)
