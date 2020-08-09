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

    ap.add_argument("--field", type=str, required=True, choices=FIELDS.keys(),
                    help="field to train parser for")
    ap.add_argument("--invoice", type=str, default=None,
                    help="path to directory containing prepared data")
    ap.add_argument("--data_dir", type=str, default='processed_data/',
                    help="path to directory containing prepared data")
    ap.add_argument("--pred_dir", type=str, default='predictions/',
                    help="path to directory containing prepared data")

    args = ap.parse_args()

    paths = []
    if args.invoice:
        if not os.path.exists(args.invoice):
            print("Could not find file '{}'".format(args.invoice))
            return
        paths.append(args.invoice)
    else:
        paths = [os.path.abspath(f) for f in glob.glob(args.data_dir + "**/*.pdf", recursive=True)]

    model = AttendCopyParse(field=args.field, restore=True)

    print("\nExtracting field '{}' from {} invoices...\n".format(args.field, len(paths)))

    predictions = model.predict(paths=paths)

    os.makedirs(args.pred_dir, exist_ok=True)
    for prediction, filename in zip(predictions, paths):
        filename = os.path.basename(filename)[:-3] + 'json'
        labels = {}
        if os.path.exists(os.path.join(args.pred_dir, filename)):
            with open(os.path.join(args.pred_dir, filename), 'r') as fp:
                labels = json.load(fp)
        with open(os.path.join(args.pred_dir, filename), 'w') as fp:
            labels[args.field] = prediction
            fp.write(json.dumps(labels))

        print("\nFilename: {}".format(filename))
        print("{}: {}\n".format(args.field, prediction))

    print("Predictions stored in '{}'".format(args.pred_dir))


if __name__ == '__main__':
    main()
