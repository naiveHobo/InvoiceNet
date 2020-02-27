import os
import glob
import argparse
import pdf2image
import simplejson
from tqdm import tqdm

from invoicenet.common import util

FIELDS = [
    "vendorname",
    "invoicedate",
    "invoicenumber",
    "amountnet",
    "amounttax",
    "amounttotal",
    "vatrate",
    "vatid",
    "taxid",
    "iban",
    "bic"
]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True,
                    help="path to directory containing invoice document images")
    ap.add_argument("--out_dir", type=str, default='processed_data/',
                    help="path to save prepared data")
    ap.add_argument("--val_size", type=float, default=0.2,
                    help="validation split ration")
    ap.add_argument("--prediction", action="store_false",
                    help="prepare data for prediction")

    args = ap.parse_args()

    if not args.prediction:
        os.makedirs(os.path.join(args.out_dir, 'test'), exist_ok=True)
        filenames = [os.path.abspath(f) for f in glob.glob(args.data_dir + "**/*.pdf", recursive=True)]
        print("Total: {}".format(len(filenames)))
        for filename in tqdm(filenames):
            try:
                page = pdf2image.convert_from_path(filename)[0]
                page.save(os.path.join(args.out_dir, "test", os.path.basename(filename)[:-3] + 'png'))

                height = page.size[1]
                width = page.size[0]

                ngrams = util.create_ngrams(page)
                for ngram in ngrams:
                    if "amount" in ngram["parses"]:
                        ngram["parses"]["amount"] = util.normalize(ngram["parses"]["amount"], key="amount")
                    if "date" in ngram["parses"]:
                        ngram["parses"]["date"] = util.normalize(ngram["parses"]["date"], key="date")

                fields = {"vendorname": '0',
                          "invoicedate": '0',
                          "invoicenumber": '0',
                          "amountnet": '0',
                          "amounttax": '0',
                          "amounttotal": '0',
                          "vatrate": '0',
                          "vatid": '0',
                          "taxid": '0',
                          "iban": '0',
                          "bic": '0'}

                data = {
                    "fields": fields,
                    "nGrams": ngrams,
                    "height": height,
                    "width": width,
                    "filename": os.path.abspath(
                        os.path.join(args.out_dir, 'test', os.path.basename(filename)[:-3] + 'png'))
                }

                with open(os.path.join(args.out_dir, 'test', os.path.basename(filename)[:-3] + 'json'), 'w') as fp:
                    fp.write(simplejson.dumps(data, indent=2))

            except Exception as exp:
                print("Skipping {} : {}".format(filename, exp))
                continue
        return

    os.makedirs(os.path.join(args.out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'val'), exist_ok=True)

    filenames = [os.path.abspath(f) for f in glob.glob(args.data_dir + "**/*.pdf", recursive=True)]

    idx = int(len(filenames) * args.val_size)
    train_files = filenames[idx:]
    val_files = filenames[:idx]

    print("Total: {}".format(len(filenames)))
    print("Training: {}".format(len(train_files)))
    print("Validation: {}".format(len(val_files)))

    for phase, filenames in [('train', train_files), ('val', val_files)]:
        print("Preparing {} data...".format(phase))

        for filename in tqdm(filenames):
            try:
                page = pdf2image.convert_from_path(filename)[0]
                page.save(os.path.join(args.out_dir, phase, os.path.basename(filename)[:-3] + 'png'))

                height = page.size[1]
                width = page.size[0]

                ngrams = util.create_ngrams(page)
                for ngram in ngrams:
                    if "amount" in ngram["parses"]:
                        ngram["parses"]["amount"] = util.normalize(ngram["parses"]["amount"], key="amount")
                    if "date" in ngram["parses"]:
                        ngram["parses"]["date"] = util.normalize(ngram["parses"]["date"], key="date")

                with open(filename[:-3] + 'json', 'r') as fp:
                    labels = simplejson.loads(fp.read())

                fields = {}
                amount_fields = ["amountnet", "amounttax", "amounttotal"]
                date_fields = ["invoicedate"]

                for field in labels:
                    if field in amount_fields:
                        fields[field] = util.normalize(labels[field], key="amount")
                    elif field in date_fields:
                        fields[field] = util.normalize(labels[field], key="date")
                    elif field in FIELDS:
                        fields[field] = labels[field]

                data = {
                    "fields": fields,
                    "nGrams": ngrams,
                    "height": height,
                    "width": width,
                    "filename": os.path.abspath(
                        os.path.join(args.out_dir, phase, os.path.basename(filename)[:-3] + 'png'))
                }

                with open(os.path.join(args.out_dir, phase, os.path.basename(filename)[:-3] + 'json'), 'w') as fp:
                    fp.write(simplejson.dumps(data, indent=2))

            except Exception as exp:
                print("Skipping {} : {}".format(filename, exp))
                continue


if __name__ == '__main__':
    main()
