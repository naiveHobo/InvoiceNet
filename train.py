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
