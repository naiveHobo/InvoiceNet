import argparse
from invoicenet.common import trainer
from invoicenet.acp.acp import AttendCopyParse


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--field", type=str, required=True, choices=["amount", "date", "number"],
                    help="field to train parser for")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="batch size for training")
    ap.add_argument("--restore", action="store_true",
                    help="restore from checkpoint")
    ap.add_argument("--data_dir", type=str, default='processed_data/',
                    help="path to prepared data")

    args = ap.parse_args()

    print("Training...")
    trainer.train(AttendCopyParse(field=args.field, batch_size=args.batch_size, restore=args.restore,
                                  data_dir=args.data_dir))


if __name__ == '__main__':
    main()
