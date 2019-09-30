import os
import re
import yaml
import pickle
import numpy as np
import pytesseract
import pdf2image
import chars2vec
from gensim.models import Word2Vec
from tqdm import tqdm


TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "templates", "template.yaml")


class Vocabulary:

    def __init__(self):
        self.PAD = "<pad>"
        self.UNKNOWN = "<unknown>"
        self.token2idx = {
            self.PAD: 0,
            self.UNKNOWN: 1
        }
        self.idx2token = {
            0: self.PAD,
            1: self.UNKNOWN
        }

    def add_token(self, token):
        token = token.lower()
        if token not in self.token2idx:
            self.token2idx[token] = len(self.token2idx)
            self.idx2token[self.token2idx[token]] = token

    def save_data(self, path):
        data = {
            "token2idx": self.token2idx,
            "idx2token": self.idx2token
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print("\nData saved as [{}]\n".format(path))

    def load_data(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.token2idx = data['token2idx']
        self.idx2token = data['idx2token']
        print("\nData loaded from [{}]\n".format(path))

    def __getitem__(self, item):
        if isinstance(item, (int, np.int32, np.int64)):
            return self.idx2token[item]
        else:
            if item in self.token2idx:
                return self.token2idx[item]
            else:
                return self.token2idx[self.UNKNOWN]

    def __len__(self):
        return len(self.token2idx)


class Parser:

    def __init__(self, template_path=TEMPLATE_PATH):
        self.template_path = template_path
        self.template = dict()
        with open(template_path, 'r') as stream:
            try:
                self.template = yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)

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


def create_embeddings(paths, embed_size=50, save_as='vocab+embeddings.pkl'):

    print("Number of invoices: {}".format(len(paths)))

    patterns = []

    pattern_vocab = Vocabulary()

    for path in tqdm(paths, total=len(paths)):
        img = pdf2image.convert_from_path(path)[0].convert('RGB')
        text = pytesseract.image_to_string(img).lower()
        text = text.replace('\n', ' ')
        text = ' '.join([word for word in text.split(' ') if word])
        words = text.split()

        pattern = []
        for word in words:
            p = ''
            for c in word:
                if c.isalpha():
                    p += 'x'
                elif c.isnumeric():
                    p += '0'
                else:
                    p += '.'
            pattern.append(p)

        patterns.append(pattern)

    # Pattern embeddings
    pattern_model = Word2Vec(patterns, size=embed_size, window=5, min_count=3, workers=4)
    patterns = list(pattern_model.wv.vocab)
    pattern_embeddings = list()

    pattern_embeddings.append(np.zeros(embed_size, dtype=np.float32))
    pattern_embeddings.append(np.random.uniform(-0.1, 0.1, embed_size))

    for pattern in patterns:
        vector = pattern_model.wv[pattern]
        pattern_embeddings.append(vector)
        pattern_vocab.add_token(pattern)

    pattern_embeddings = np.array(pattern_embeddings, dtype=np.float32)
    print("\n*************************************************")
    print("Generated pattern embeddings from data!")
    print("Pattern vocab size: {}\b".format(len(pattern_vocab)))
    print("*************************************************\n")

    # Character embeddings
    char_vocab = Vocabulary()
    char_embeddings = list()

    char_embeddings.append(np.zeros(embed_size, dtype=np.float32))
    char_embeddings.append(np.random.uniform(-0.1, 0.1, embed_size))
    char_embeddings.append(np.random.uniform(-0.1, 0.1, embed_size))
    char_vocab.add_token(' ')

    model = chars2vec.load_model('eng_{}'.format(embed_size))

    for char in model.char_to_ix:
        char_embeddings.append(model.vectorize_words([char])[0])
        char_vocab.add_token(char)

    char_embeddings = np.array(char_embeddings, dtype=np.float32)
    print("\n*************************************************")
    print("Generated character embeddings from data!")
    print("Character vocab size: {}".format(len(char_vocab)))
    print("*************************************************\n")

    # Save embeddings
    data = {
        "char_vocab": char_vocab,
        "char_embeddings": char_embeddings,
        "pattern_vocab": pattern_vocab,
        "pattern_embeddings": pattern_embeddings
    }

    with open(save_as, 'wb') as f:
        pickle.dump(data, f)

    print("Saved vocabulary and embeddings as {}!".format(save_as))
