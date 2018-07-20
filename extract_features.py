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

processed_text:         The raw text of the last word in the N-gram

text_pattern:           The raw text, after replacing uppercase characters with X, 
                        lowercase with x, numbers with 0, repeating whitespace with 
                        single whitespace and the rest with ?

bottom_margin:          Vertical coordinate of the bottom margin of the 
                        N-gram normalized to the page height

top_margin:             Vertical coordinate of the top margin of the 
                        N-gram normalized to the page height

right_margin:           Horizontal coordinate of the right margin of the
                        N-gram normalized to the page width

left_margin:            Horizontal coordinate of the left margin of the
                        N-gram normalized to the page width

has_digits:             Whether there are any digits 0-9 in the N-gram

length:                 Number of characters in the N-gram

position_on_line:       Count of words to the left of this N-gram normalized 
                        to the count of total words on this line

line_size:              The number of words on this line

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

    # Filters the data into individual files and finds out the minimum and maximum
    # x and y coordinates to estimate the width and height of each file.
    # Also estimates the x coordinate for each token in each line for every file.
    for i, row in df.iterrows():
        if row['files'] not in files:
            files[row['files']] = {'lines': {'words': [], 'labels': [], 'ymin': [], 'ymax': []},
                                   'xmin': sys.maxsize, 'ymin': sys.maxsize, 'xmax': 0, 'ymax': 0}
        tokens = row['words'].strip().split(' ')
        char_length = (row['coords'][2] - row['coords'][0]) / len(row['words'].strip())
        token_coords = [{'xmin': row['coords'][0],
                         'xmax': row['coords'][0] + (char_length * len(tokens[0]))}]
        for idx in range(1, len(tokens)):
            token_coords.append({'xmin': token_coords[-1]['xmax'] + char_length,
                                 'xmax': token_coords[-1]['xmax'] + (char_length * (len(tokens[idx])+1))})
        files[row['files']]['lines']['words'].append({'tokens': tokens, 'coords': token_coords})
        files[row['files']]['lines']['labels'].append(row['labels'])
        files[row['files']]['lines']['ymin'].append(row['coords'][1])
        files[row['files']]['lines']['ymax'].append(row['coords'][3])
        files[row['files']]['xmin'] = min(files[row['files']]['xmin'], row['coords'][0])
        files[row['files']]['ymin'] = min(files[row['files']]['ymin'], row['coords'][1])
        files[row['files']]['xmax'] = max(files[row['files']]['xmax'], row['coords'][2])
        files[row['files']]['ymax'] = max(files[row['files']]['ymax'], row['coords'][3])

    del df

    grams = {'raw_text': [],
             'processed_text': [],
             'text_pattern': [],
             'length': [],
             'line_size': [],
             'position_on_line': [],
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
             'label': [],
             'closest_ngrams': []
             }

    label_dict = {0: 0, 1: 1, 2: 2, 18: 3}

    # Calculates N-grams of lengths ranging from 1-4 for each line in each
    # file and calculates 17 features for each N-gram.
    with tqdm(total=len(files)) as pbar:
        for key, value in files.items():
            num_ngrams = len(grams['raw_text'])
            page_height = value['ymax'] - value['ymin']
            page_width = value['xmax'] - value['xmin']
            for i in range(len(value['lines']['words'])):
                tokens = value['lines']['words'][i]['tokens']
                token_coords = value['lines']['words'][i]['coords']
                for ngram in ngrammer(tokens):
                    grams['parses_as_date'].append(0.0)
                    grams['parses_as_amount'].append(0.0)
                    grams['parses_as_number'].append(0.0)
                    processed_text = []
                    for word in ngram:
                        if bool(list(datefinder.find_dates(word))):
                            processed_text.append('date')
                            grams['parses_as_date'][-1] = 1.0
                        elif bool(re.search(r'\d\.\d', word)) or '$' in word:
                            processed_text.append('amount')
                            grams['parses_as_amount'][-1] = 1.0
                        elif word.isnumeric():
                            processed_text.append('number')
                            grams['parses_as_number'][-1] = 1.0
                        else:
                            processed_text.append(word.lower())
                    raw_text = ' '.join(ngram)
                    grams['raw_text'].append(raw_text)
                    grams['processed_text'].append(' '.join(processed_text))
                    grams['text_pattern'].append(re.sub('[a-z]', 'x', re.sub('[A-Z]', 'X', re.sub('\d', '0', re.sub(
                        '[^a-zA-Z\d\ ]', '?', raw_text)))))
                    grams['length'].append(len(' '.join(ngram)))
                    grams['line_size'].append(len(tokens))
                    grams['position_on_line'].append(tokens.index(ngram[0])/len(tokens))
                    grams['has_digits'].append(1.0 if bool(re.search(r'\d', raw_text)) else 0.0)
                    grams['left_margin'].append((token_coords[tokens.index(ngram[0])]['xmin'] - value['xmin']) / page_width)
                    grams['top_margin'].append((value['lines']['ymin'][i] - value['ymin']) / page_height)
                    grams['right_margin'].append((token_coords[tokens.index(ngram[-1])]['xmax'] - value['xmin']) / page_width)
                    grams['bottom_margin'].append((value['lines']['ymax'][i] - value['ymin']) / page_height)
                    grams['page_width'].append(page_width)
                    grams['page_height'].append(page_height)
                    grams['label'].append(label_dict[value['lines']['labels'][i]])

            # Finds the closest N-grams on all 4 sides for each N-gram
            for i in range(num_ngrams, len(grams['raw_text'])):
                grams['closest_ngrams'].append([-1] * 4)
                distance = [sys.maxsize] * 6
                for j in range(num_ngrams, len(grams['raw_text'])):
                    d = [grams['top_margin'][i] - grams['bottom_margin'][j],
                         grams['top_margin'][j] - grams['bottom_margin'][i],
                         grams['left_margin'][i] - grams['right_margin'][j],
                         grams['left_margin'][j] - grams['right_margin'][i],
                         abs(grams['left_margin'][i] - grams['left_margin'][j])]
                    if i == j:
                        continue
                    # If in the same line, check for closest ngram to left and right
                    if d[0] == d[1]:
                        if distance[2] > d[2] > 0:
                            distance[2] = d[2]
                            grams['closest_ngrams'][i][2] = j
                        if distance[3] > d[3] > 0:
                            distance[3] = d[3]
                            grams['closest_ngrams'][i][3] = j
                    # If this ngram is above current ngram
                    elif distance[0] > d[0] >= 0 and distance[4] > d[4]:
                        distance[0] = d[0]
                        distance[4] = d[4]
                        grams['closest_ngrams'][i][0] = j
                    # If this ngram is below current ngram
                    elif distance[1] > d[1] >= 0 and distance[5] > d[4]:
                        distance[1] = d[1]
                        distance[5] = d[4]
                        grams['closest_ngrams'][i][1] = j

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
