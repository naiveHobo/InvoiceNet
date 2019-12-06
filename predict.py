import argparse
from invoicenet.acp.acp import AttendCopyParse


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--field", type=str, required=True, choices=["amount", "date", "number"],
                    help="field to train parser for")
    ap.add_argument("--data_dir", type=str, default='processed_data/',
                    help="path to prepared data")

    args = ap.parse_args()

    print("Training...")
    model = AttendCopyParse(field=args.field, data_dir=args.data_dir, batch_size=1, restore=True)

    predictions = model.test_set()
    for file in predictions.keys():
        print("File: {}".format(file))
        print("  - {}: {}\n".format(args.field, predictions[file][args.field]))


if __name__ == '__main__':
    main()
