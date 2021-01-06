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
import glob
import json
import argparse

from invoicenet import FIELDS
from invoicenet.acp.acp import AttendCopyParse


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--field", nargs='+', type=str, required=True, choices=FIELDS.keys(),
                    help="field to train parser for")
    ap.add_argument("--invoice", type=str, default=None,
                    help="path to invoice pdf file")
    ap.add_argument("--data_dir", type=str, default='invoices/',
                    help="path to directory containing invoice pdf files")
    ap.add_argument("--pred_dir", type=str, default='predictions/',
                    help="path to directory where predictions should be stored")

    args = ap.parse_args()

    paths = []
    fields = []
    predictions = {}

    if args.invoice:
        if not os.path.exists(args.invoice):
            print("ERROR: Could not find file '{}'".format(args.invoice))
            return
        if not args.invoice.endswith('.pdf'):
            print("ERROR: '{}' is not a PDF file".format(args.invoice))
            return
        paths.append(args.invoice)
    else:
        paths = [os.path.abspath(f) for f in glob.glob(args.data_dir + "**/*.pdf", recursive=True)]

    if not os.path.exists('./models/invoicenet/'):
        print("Could not find any trained models!")
        return
    else:
        models = os.listdir('./models/invoicenet/')
        for field in args.field:
            if field in models:
                fields.append(field)
            else:
                print("Could not find a trained model for field '{}', skipping...".format(field))

    for field in fields:
        print("\nExtracting field '{}' from {} invoices...\n".format(field, len(paths)))
        model = AttendCopyParse(field=field, restore=True)
        predictions[field] = model.predict(paths=paths)

    os.makedirs(args.pred_dir, exist_ok=True)
    for idx, filename in enumerate(paths):
        filename = os.path.basename(filename)[:-3] + 'json'
        labels = {}
        if os.path.exists(os.path.join(args.pred_dir, filename)):
            with open(os.path.join(args.pred_dir, filename), 'r') as fp:
                try:
                    labels = json.load(fp)
                except:
                    labels = {}
        with open(os.path.join(args.pred_dir, filename), 'w') as fp:
            print("\nFilename: {}".format(filename))
            for field in predictions.keys():
                labels[field] = predictions[field][idx]
                print("  {}: {}".format(field, labels[field]))
            fp.write(json.dumps(labels, indent=2))
            print('\n')

    print("Predictions stored in '{}'".format(args.pred_dir))


if __name__ == '__main__':
    main()
