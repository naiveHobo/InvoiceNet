import sys
import re
import pickle
import pandas as pd
from nltk import ngrams
import argparse
import datefinder
from tqdm import tqdm

"""
FEATURES:

raw_text:               The raw text

raw_text_last_word:     The raw text of the last word in the N-gram

text_pattern:           The raw text, after replacing uppercase characters with X, 
                        lowercase with x, numbers with 0, repeating whitespace with 
                        single whitespace and the rest with ?

bottom_margin:          Vertical coordinate of the bottom margin of the N-gram normalized to the page height

top_margin:             Vertical coordinate of the top margin of the N-gram normalized to the page height

right_margin:           Horizontal coordinate of the right margin of the N-gram normalized to the page width

left_margin:            Horizontal coordinate of the left margin of the N-gram normalized to the page width

has_digits:             Whether there are any digits 0-9 in the N-gram

length:                 Number of characters in the N-gram

page_height:            The height of the page of this N-gram

page_width:             The width of the page of this N-gram

parses_as_amount:       Whether the N-gram parses as a fractional amount

parses_as_date:         Whether the N-gram parses as a date

parses_as_number:       Whether the N-gram parses as an integer
"""


def ngrammer(tokens, length=4):
    """
    Generates n-grams from the given tokens
    :param tokens: list of tokens in the text
    :param length: n-grams of up to this length
    :return: n-grams as tuples
    """
    for n in range(1, min(len(tokens) + 1, length+1)):
        for gram in ngrams(tokens, n):
            yield gram


def extract_features(path):
    """
    Loads a pickled dataframe from the given path, creates n-grams and extracts features
    :param path: path to pickled dataframe
    :return: dataframe containing n-grams and corresponding features
    """

    with open(path, 'rb') as pklfile:
        df = pickle.load(pklfile)

    files = {}

    print("\nExtracting features...\n")

    for i, row in df.iterrows():
        if row['files'] not in files:
            files[row['files']] = {'lines': {'words': [], 'labels': [], 'coords': []},
                                   'xmin': sys.maxsize, 'ymin': sys.maxsize, 'xmax': 0, 'ymax': 0}
        files[row['files']]['lines']['words'].append(row['words'])
        files[row['files']]['lines']['labels'].append(row['labels'])
        files[row['files']]['lines']['coords'].append(row['coords'])
        files[row['files']]['xmin'] = min(files[row['files']]['xmin'], row['coords'][0])
        files[row['files']]['ymin'] = min(files[row['files']]['ymin'], row['coords'][1])
        files[row['files']]['xmax'] = max(files[row['files']]['xmax'], row['coords'][2])
        files[row['files']]['ymax'] = max(files[row['files']]['ymax'], row['coords'][3])

    del df

    grams = {'raw_text': [],
             'raw_text_last_word': [],
             'text_pattern': [],
             'length': [],
             'has_digits': [],
             'bottom_margin': [],
             'top_margin': [],
             'left_margin': [],
             'right_margin': [],
             'page_width': [],
             'page_height': [],
             'parses_as_amount': [],
             'parses_as_date': [],
             'parses_as_number': [],
             'label': []
             }

    with tqdm(total=len(files)) as pbar:
        for key, value in files.items():
            page_height = value['ymax'] - value['ymin']
            page_width = value['xmax'] - value['xmin']
            for i in range(len(value['lines']['words'])):
                tokens = re.sub(r"  ", " ", value['lines']['words'][i].strip()).split(' ')
                for ngram in ngrammer(tokens):
                    raw_text = ' '.join(ngram)
                    grams['raw_text'].append(raw_text)
                    grams['raw_text_last_word'].append(ngram[-1])
                    grams['text_pattern'].append(re.sub('[a-z]', 'x', re.sub('[A-Z]', 'X', re.sub('\d', '0', re.sub(
                        '[^a-zA-Z\d\ ]', '?', raw_text)))))
                    grams['length'].append(len(ngram))
                    grams['has_digits'].append(bool(re.search(r'\d', raw_text)))
                    grams['left_margin'].append((value['lines']['coords'][i][0] - value['xmin']) / page_width)
                    grams['top_margin'].append((value['lines']['coords'][i][1] - value['ymin']) / page_height)
                    grams['right_margin'].append((value['lines']['coords'][i][2] - value['xmin']) / page_width)
                    grams['bottom_margin'].append((value['lines']['coords'][i][3] - value['ymin']) / page_height)
                    grams['page_width'].append(page_width)
                    grams['page_height'].append(page_height)
                    grams['parses_as_date'].append(bool(list(datefinder.find_dates(raw_text))))
                    grams['parses_as_amount'].append(
                        bool(re.search(r'\d\.\d', raw_text)) and not grams['parses_as_date'][-1])
                    grams['parses_as_number'].append(bool(re.search(r'\d', raw_text)))
                    grams['label'].append(value['lines']['labels'][i])
            pbar.update(1)

    return pd.DataFrame(data=grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dftrain.pk", help="path to training data")
    ap.add_argument("--save_as", default="data/features.pk", help="save extracted features with this name")
    args = ap.parse_args()
    features = extract_features(args.data)
    features.to_pickle(args.save_as, protocol=3)
    print("\nSaved features as {}".format(args.save_as))


if __name__ == '__main__':
    main()
