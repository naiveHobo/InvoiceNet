import os
import json

import torch
import torch.nn.functional
from torch.utils.data import DataLoader

from ..common.model import Model
from .data import RealData
from .model import AttendCopyParseModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AttendCopyParse(Model):
    n_hid = 32
    frac_ce_loss = 0.0001
    lr = 3e-4
    keep_prob = 0.5

    def __init__(self, field, train_data=None, val_data=None, test_data=None, batch_size=8, restore=False):

        self.field = field
        self.batch_size = batch_size

        self.restore_all_path = './models/invoicenet/{}/best'.format(self.field) if restore else None
        os.makedirs("./models/invoicenet", exist_ok=True)

        if train_data:
            self.train_data_loader = DataLoader(train_data,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                drop_last=True)
            self.train_data_gen = iter(self.train_data_loader)

        if val_data:
            self.val_data_loader = DataLoader(val_data,
                                              batch_size=self.batch_size,
                                              shuffle=False,
                                              drop_last=True)
            self.val_data_gen = iter(self.val_data_loader)

        if test_data:
            self.test_data = test_data
            self.test_data_loader = DataLoader(test_data,
                                               batch_size=1,
                                               shuffle=False)

        self.model = AttendCopyParseModel(field, batch_size).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.restore_all_path:
            if not os.path.exists('./models/invoicenet/{}'.format(self.field)):
                raise Exception("No trained model available for the field '{}'".format(self.field))
            print("Restoring all " + self.restore_all_path + "...")
            self.load(self.restore_all_path)
        else:
            restore = self.model.parser.restore()
            if restore is not None:
                print("Restoring %s parser %s..." % (self.field, restore))
                checkpoint = torch.load(restore)
                self.model.parser.load_state_dict(checkpoint['model_state_dict'])

    def test_set(self, out_path="./predictions/"):
        actuals = []
        self.model.eval()
        for sample, _ in self.test_data_loader:
            for idx in range(len(sample)):
                sample[idx] = sample[idx].to(device)
            parsed, _ = self.model(*sample)
            parsed = torch.argmax(parsed, dim=1).numpy()
            actuals.extend(self.test_data.array_to_str(parsed))
        os.makedirs(out_path, exist_ok=True)
        extracts = {}
        for actual, filename in zip(actuals, self.test_data.filenames):
            filename = '.'.join([os.path.basename(filename).split('.')[0], 'pdf'])
            print("Prediciton: {}\t\tFilename: {}".format(actual, filename))
            filename = filename[:-3] + 'json'
            predictions = {}
            if os.path.exists(os.path.join(out_path, filename)):
                with open(os.path.join(out_path, filename), 'r') as fp:
                    predictions = json.load(fp)
            with open(os.path.join(out_path, filename), 'w') as fp:
                predictions[self.field] = actual
                fp.write(json.dumps(predictions))
            extracts[filename] = predictions
        print("Predictions stored in '{}'".format(out_path))
        return extracts

    def criterion(self, prediction, cross_entropy_uniform, target):
        mask = (target != RealData.pad_idx).float()
        label_cross_entropy = torch.sum(
            torch.nn.functional.cross_entropy(prediction, target, reduction='none') * mask,
            dim=1) / torch.sum(mask, dim=1)
        cross_entropy_uniform_loss = AttendCopyParse.frac_ce_loss * torch.mean(cross_entropy_uniform)
        field_loss = torch.mean(label_cross_entropy)
        loss = field_loss + cross_entropy_uniform_loss
        return loss

    def train_step(self):
        try:
            x, y = next(self.train_data_gen)
        except StopIteration:
            self.train_data_gen = iter(self.train_data_loader)
            x, y = next(self.train_data_gen)
        y = y.to(device)
        for idx in range(len(x)):
            x[idx] = x[idx].to(device)
        self.model.train()
        self.optimizer.zero_grad()
        parsed, cross_entropy_uniform = self.model(*x)
        loss = self.criterion(parsed, cross_entropy_uniform, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self):
        try:
            x, y = next(self.val_data_gen)
        except StopIteration:
            self.val_data_gen = iter(self.val_data_loader)
            x, y = next(self.val_data_gen)
        y = y.to(device)
        for idx in range(len(x)):
            x[idx] = x[idx].to(device)
        self.model.eval()
        parsed, cross_entropy_uniform = self.model(*x)
        return self.criterion(parsed, cross_entropy_uniform, y).item()

    def save(self, name):
        os.makedirs("./models/invoicenet/%s/" % self.field, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, "./models/invoicenet/%s/%s" % (self.field, name))

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
