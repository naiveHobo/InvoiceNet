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

import os
import argparse

from invoicenet import FIELDS
from invoicenet.common import trainer
from invoicenet.acp.acp import AttendCopyParse
from invoicenet.acp.data import RealData


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--field", type=str,  choices=FIELDS.keys(),
                    help="field to train parser for")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="batch size for training")
    ap.add_argument("--restore", action="store_true",
                    help="restore from checkpoint")
    ap.add_argument("--data_dir", type=str, default='processed_data/',
                    help="path to directory containing prepared data")

    args = ap.parse_args()

    train_data = RealData(field=args.field, data_dir=os.path.join(args.data_dir, 'train/'))
    val_data = RealData(field=args.field, data_dir=os.path.join(args.data_dir, 'val/'))

    print("Training...")
    trainer.train(AttendCopyParse(field=args.field, batch_size=args.batch_size, restore=args.restore,
                                  train_data=train_data, val_data=val_data))


if __name__ == '__main__':
    main()
