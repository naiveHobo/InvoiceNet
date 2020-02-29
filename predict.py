import os
import argparse
import pdf2image

from invoicenet import FIELDS
from invoicenet.acp.acp import AttendCopyParse
from invoicenet.acp.data import RealData
from invoicenet.common import util


def load_file(path):
    image = pdf2image.convert_from_path(path)[0]
    height = image.size[1]
    width = image.size[0]

    ngrams = util.create_ngrams(image)
    for ngram in ngrams:
        if "amount" in ngram["parses"]:
            ngram["parses"]["amount"] = util.normalize(ngram["parses"]["amount"], key="amount")
        if "date" in ngram["parses"]:
            ngram["parses"]["date"] = util.normalize(ngram["parses"]["date"], key="date")

    fields = {field: '0' for field in FIELDS}

    page = {
        "fields": fields,
        "nGrams": ngrams,
        "height": height,
        "width": width,
        "filename": path
    }

    return {
        'image': image,
        'page': page
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--field", type=str, required=True, choices=FIELDS.keys(),
                    help="field to train parser for")
    ap.add_argument("--invoice", type=str, default=None,
                    help="path to directory containing prepared data")
    ap.add_argument("--data_dir", type=str, default='processed_data/',
                    help="path to directory containing prepared data")
    ap.add_argument("--pred_dir", type=str, default='predictions/',
                    help="path to directory containing prepared data")

    args = ap.parse_args()

    if args.invoice:
        if not os.path.exists(args.invoice):
            print("Could not find file '{}'".format(args.invoice))
            return
        data = load_file(args.invoice)
        test_data = RealData(field=args.field, data_file=data)
    else:
        test_data = RealData(field=args.field, data_dir=os.path.join(args.data_dir, 'predict/'))

    print("Extracting field '{}'...".format(args.field))
    model = AttendCopyParse(field=args.field, test_data=test_data, batch_size=1, restore=True)
    model.test_set(out_path=args.pred_dir)


if __name__ == '__main__':
    main()
