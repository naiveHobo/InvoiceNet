import os
import re
import glob
import argparse
import pdf2image
import simplejson
import pytesseract

from tqdm import tqdm
from pytesseract import Output
from sklearn.model_selection import train_test_split


class TextParser:

    def __init__(self):
        self.template = dict()
        self.template['amount'] = [r'[1-9]\d{0,2}(?:,\d{2,3})*(?:\.\d+)', r'[1-9]\d*(?:\.\d+)']
        self.template['date'] = [r'\d{1,2}[\/\\\.-]\d{1,2}[\/\\\.-]\d{2,4}', r'\d{2,4}[\/\\\.-]\d{1,2}[\/\\\.-]\d{1,2}']

    def parse(self, text, key):
        if key not in self.template:
            return False
        for regex in self.template[key]:
            if re.findall(regex, text):
                return True
        return False

    def find(self, text, key):
        values = []
        if key not in self.template:
            return values
        for regex in self.template[key]:
            values.extend(re.findall(regex, text))
        values = list(set(values))
        return values

    def replace(self, text, new, key):
        if key not in self.template:
            return text
        for regex in self.template[key]:
            text = re.sub(regex, new, text)
        while '  ' in text:
            text = text.replace('  ', ' ')
        return text


def extract_words(img):
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(data['level'])
    words = [{'text': data['text'][i],
              'left': data['left'][i],
              'top': data['top'][i],
              'right': data['left'][i] + data['width'][i],
              'bottom': data['top'][i] + data['height'][i]}
             for i in range(n_boxes) if data['text'][i]]
    return words


def divide_into_lines(words, height, width):
    cur = words[0]
    lines = []
    line = []
    for word in words:
        if ((word['top'] - cur['top']) / height) > 0.005:
            # if difference between y-coordinate of current word and previous word
            # is more than 0.5% of the height, consider the current word to be in the next line
            lines.append(line)
            line = [word]
        elif ((word['left'] - cur['right']) / width) > 0.05:
            # if difference between x-coordinate of current word and previous word
            # is more than 5% of the width, consider the current word to be in a different line
            lines.append(line)
            line = [word]
        else:
            line.append(word)
        cur = word
    lines.append(line)
    return lines


def create_ngrams(img, length=4):
    words = extract_words(img)
    lines = divide_into_lines(words, height=img.size[1], width=img.size[0])
    tokens = [line[i:i + N] for line in lines for N in range(1, length + 1) for i in range(len(line) - N + 1)]
    ngrams = []
    parser = TextParser()

    for token in tokens:
        text = ' '.join([word['text'] for word in token])
        ngram = {
            "words": token,
            "parses": {}
        }
        if parser.parse(text=text, key='amount'):
            ngram["parses"]["amount"] = parser.find(text=text, key='amount')[0]
        if parser.parse(text=text, key='date'):
            ngram["parses"]["date"] = parser.find(text=text, key='date')[0]
        ngrams.append(ngram)

    return ngrams


def normalize(text, key):
    if key == 'amount':
        text = text.replace(",", '')
        splits = text.split('.')
        if len(splits) == 1:
            text += ".00"
        else:
            text = splits[0] + '.' + splits[1][:2]
    else:
        text = text.replace(".", '-').replace('/', '-')
    return text


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True,
                    help="path to directory containing invoice document images")
    ap.add_argument("--out_dir", type=str, default='invoicenet/acp/data/',
                    help="path to save prepared data")
    ap.add_argument("--val_size", type=float, default=0.2,
                    help="validation split ration")
    ap.add_argument("--vocab_path", type=str, default='vocab.pkl',
                    help="path to save vocabulary file")
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

                ngrams = create_ngrams(page)
                for ngram in ngrams:
                    if "amount" in ngram["parses"]:
                        ngram["parses"]["amount"] = normalize(ngram["parses"]["amount"], key="amount")
                    if "date" in ngram["parses"]:
                        ngram["parses"]["date"] = normalize(ngram["parses"]["date"], key="date")

                fields = {"amount": '0',
                          "date": '0',
                          "number": '0'}

                data = {
                    "fields": fields,
                    "nGrams": ngrams,
                    "height": height,
                    "width": width,
                    "filename": os.path.join(args.out_dir, 'test', os.path.basename(filename)[:-3] + 'png')
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

    train_files, val_files = train_test_split(filenames, test_size=args.val_size)

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

                ngrams = create_ngrams(page)
                for ngram in ngrams:
                    if "amount" in ngram["parses"]:
                        ngram["parses"]["amount"] = normalize(ngram["parses"]["amount"], key="amount")
                    if "date" in ngram["parses"]:
                        ngram["parses"]["date"] = normalize(ngram["parses"]["date"], key="date")

                with open(filename[:-3] + 'json', 'r') as fp:
                    labels = simplejson.loads(fp.read())

                fields = {}
                if "amount" in labels:
                    fields["amount"] = normalize(labels["amount"], key="amount")
                if "date" in labels:
                    fields["date"] = normalize(labels["date"], key="date")
                if "number" in labels:
                    fields["number"] = labels["number"]

                data = {
                    "fields": fields,
                    "nGrams": ngrams,
                    "height": height,
                    "width": width,
                    "filename": os.path.join(args.out_dir, phase, os.path.basename(filename)[:-3] + 'png')
                }

                with open(os.path.join(args.out_dir, phase, os.path.basename(filename)[:-3] + 'json'), 'w') as fp:
                    fp.write(simplejson.dumps(data, indent=2))

            except Exception as exp:
                print("Skipping {} : {}".format(filename, exp))
                continue


if __name__ == '__main__':
    main()
