import argparse
from invoicenet.common import trainer
from invoicenet.parsing.parser import Parser


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--field", type=str, required=True, choices=["amount", "date"],
                    help="field to train parser for")
    ap.add_argument("--batch_size", type=int, default=128,
                    help="batch size for training")
    ap.add_argument("--restore", action="store_true",
                    help="restore from checkpoint")

    args = ap.parse_args()

    print("Training...")
    trainer.train(Parser(field=args.field, batch_size=args.batch_size, restore=args.restore))


if __name__ == '__main__':
    main()
