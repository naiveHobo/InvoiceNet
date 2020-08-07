# Copyright (c) 2019 Tradeshift
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import pytesseract
from pytesseract import Output
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import datefinder
import datetime

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def print_vars(vars):
    total = 0
    for var in vars:
        shape = tf.shape(var)
        print(var.name, shape)
        total += np.prod(shape)
    print(total)


def get_devices():
    gpus = [x.name for x in (device_lib.list_local_devices()) if x.device_type == 'GPU']
    if len(gpus) > 0:
        devices = gpus
    else:
        print("WARNING: No GPU's found. Using CPU")
        devices = ['cpu:0']

    print("Using devices: ", devices)
    return devices


def batch_parallel(map_fn, devices, **kwargs):
    """
    Parallelize map_fn across devices.

    :param map_fn: function that takes kwargs and returns a tuple of tensors
    :param devices: A list of devices to parallelize over
    :param kwargs: kwargs of input to map, will be split along axis 0.
    :return: The outputs of map_fn
    The inner list is the output from each device.
    """
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, len(devices))

    out_splits = {}
    for i, device in enumerate(devices):
        with tf.device(device):
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                outs = map_fn(**{k: v[i] for k, v in in_splits.items()})
                for j, out in enumerate(outs):
                    if j not in out_splits:
                        out_splits[j] = []
                    out_splits[j].append(out)

    return [out_splits[i] for i in range(len(out_splits))]


class TextParser:

    def __init__(self):
        self.template = dict()
        self.template['amount'] = [r'\d+[,\d]*\.\d+']
        self.template['date'] = [r'\d{1,2}[\/\\\.\,-]\d{1,2}[\/\\\.\,-]\d{2,4}',
                                 r'\d{2,4}[\/\\\.\,-]\d{1,2}[\/\\\.\,-]\d{1,2}']

    def parse(self, text, key):
        if key == 'date':
            try:
                matches = [date for date in datefinder.find_dates(text) if date <= datetime.datetime.today()]
                if matches:
                    return True
                else:
                    return False
            except Exception:
                return False
        if key not in self.template:
            return False
        for regex in self.template[key]:
            if re.findall(regex, text):
                return True
        return False

    def find(self, text, key):
        if key == 'date':
            try:
                matches = [date for date in datefinder.find_dates(text) if date <= datetime.datetime.today()]
                if len(matches) > 0:
                    return [match.strftime('%m-%d-%Y') for match in matches]
                else:
                    return []
            except Exception:
                return []
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
        if parser.parse(text=text, key='date'):
            ngram["parses"]["date"] = parser.find(text=text, key='date')[0]
        elif parser.parse(text=text, key='amount'):
            ngram["parses"]["amount"] = parser.find(text=text, key='amount')[0]
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
        matches = [date for date in datefinder.find_dates(text) if date <= datetime.datetime.today()]
        if matches:
            text = matches[0].strftime('%m-%d-%Y')
    return text
