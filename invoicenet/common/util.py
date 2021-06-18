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

import re
import io
import math
import datetime
import datefinder
import pytesseract
from pytesseract import Output

import os
from PIL import Image
from googleapiclient.discovery import build
import base64
from google.cloud import vision

## API keys for google ocr
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="google_api_keys.json"


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


def extract_words(img, height, width, ocr_engine='pytesseract'):
    if ocr_engine == 'pytesseract':
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(data['text'])
        words = [
            {
                'text': data['text'][i],
                'left': data['left'][i],
                'top': data['top'][i],
                'right': data['left'][i] + data['width'][i],
                'bottom': data['top'][i] + data['height'][i]
            }
            for i in range(n_boxes) if data['text'][i]
        ]
        return words

    
    elif ocr_engine=='google_ocr':
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        client = vision.ImageAnnotatorClient()
        content=img_byte_arr
        image_ = vision.Image(content=content)
        response = client.text_detection(image=image_)
        texts = response.text_annotations

        words=[]
        first=True
        for text in texts:
            if first:
                first=False
                continue
            data={}
            data['text']=text.description
            x_vert=[]
            y_vert=[]
            for vertex in text.bounding_poly.vertices:
                x_vert.append(vertex.x)
                y_vert.append(vertex.y)
            data['left']=min(x_vert)
            data['right']=max(x_vert)
            data['top']=min(y_vert)
            data['bottom']=max(y_vert)
            words.append(data)
        return words
    
    
    elif ocr_engine == 'aws_textract':

        import boto3

        # use aws textract
        client = boto3.client('textract')

        # convert PpmImageFile to byte
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # call aws-textract API
        response = client.detect_document_text(Document={'Bytes': img_byte_arr})

        # get image weight and height to convert normalized coordinate from response
        words = [
            {
                'text': data['Text'],
                'left': math.floor((data['Geometry']['BoundingBox']['Left']) * width),
                'top': math.floor((data['Geometry']['BoundingBox']['Top']) * height),
                'right': math.ceil(
                    (data['Geometry']['BoundingBox']['Left'] + data['Geometry']['BoundingBox']['Width']) * width),
                'bottom': math.ceil(
                    (data['Geometry']['BoundingBox']['Top'] + data['Geometry']['BoundingBox']['Height']) * height)
            } for data in response['Blocks'] if "Text" in data
        ]
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


def create_ngrams(img, height, width, length=4, ocr_engine='pytesseract'):##change ocr_engine here while running predict.py  
    words = extract_words(img, height=height, width=width, ocr_engine=ocr_engine)
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
