from datetime import datetime
import numpy as np

import torch.nn.functional
from torch.utils.data import DataLoader

from ..common.model import Model
from .parsers import *
from .data import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Parser(Model):
    context_size = 128
    experiment = datetime.now()

    def __init__(self, field, batch_size=128, restore=False):
        self.batch_size = batch_size
        self.type = field
        self.output_length = {"date": RealData.seq_date, "amount": RealData.seq_amount}[self.type]
        self.continue_from = './models/parsers/{}/best'.format(self.type) if restore else None

        train_data = ParseData('invoicenet/parsing/data/%s/train.tsv' % self.type, self.output_length)
        val_data = ParseData('invoicenet/parsing/data/%s/valid.tsv' % self.type, self.output_length)

        self.train_data_loader = DataLoader(train_data,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            num_workers=16,
                                            drop_last=True)

        self.val_data_loader = DataLoader(val_data,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=16,
                                          drop_last=True)

        self.train_data_gen = iter(self.train_data_loader)
        self.val_data_gen = iter(self.val_data_loader)

        self.context = torch.zeros((self.batch_size, self.context_size), dtype=torch.float32).to(device)

        self.parser = {'amount': AmountParser(self.batch_size), 'date': DateParser()}[self.type]
        self.parser = self.parser.to(device)

        self.optimizer = torch.optim.Adam(self.parser.parameters(), lr=1e-4)

        if self.continue_from:
            print("Restoring " + self.continue_from + "...")
            self.load(self.continue_from)

    @staticmethod
    def criterion(prediction, target):
        mask = (target != ParseData.pad_idx).float()
        label_cross_entropy = torch.mean(
            torch.nn.functional.cross_entropy(prediction, target, reduction='none') * mask) / np.log(2.)
        return label_cross_entropy

    def train_step(self):
        try:
            x, y = next(self.train_data_gen)
        except StopIteration:
            self.train_data_gen = iter(self.train_data_loader)
            x, y = next(self.train_data_gen)
        x, y = x.to(device), y.to(device)
        x = torch.nn.functional.one_hot(x, RealData.n_output).float().permute(0, 2, 1)  # (bs, n_out, seq)
        self.parser.train()
        self.optimizer.zero_grad()
        parsed = self.parser(x, self.context)
        loss = self.criterion(parsed, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self):
        try:
            x, y = next(self.val_data_gen)
        except StopIteration:
            self.val_data_gen = iter(self.val_data_loader)
            x, y = next(self.val_data_gen)
        x, y = x.to(device), y.to(device)
        x = torch.nn.functional.one_hot(x, RealData.n_output).float().permute(0, 2, 1)  # (bs, n_out, seq)
        self.parser.eval()
        parsed = self.parser(x, self.context)
        return self.criterion(parsed, y).item()

    def save(self, name):
        torch.save({
            'model_state_dict': self.parser.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, "./models/parsers/%s/%s" % (self.type, name))

    def load(self, path):
        checkpoint = torch.load(path)
        self.parser.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
