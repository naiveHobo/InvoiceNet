import numpy as np
import pickle
import gzip
from gensim.models import Word2Vec


class DataHandler:
    def __init__(self, data=None, max_len=10):
        self.data = data
        self.max_length = max_len
        self.vocab_size = 0
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None
        self.embed_size = 300
        self.PAD = '<pad>'
        self.UNKNOWN = '<unk>'
        self.START = '<start>'
        self.END = '<end>'
        self.label_dict = {0: 0, 1: 1, 2: 2, 8: 3, 14: 4, 18: 5}
        self.num_classes = len(self.label_dict)
        self.train_data = {}
        # self.type_dict = {'text': 0.1, 'number': 0.2, 'email': 0.3, 'date': 0.4, '': 0.5, 'money': 0.6, 'phone': 0.7}

    def read(self, data, max_len=10):
        """Read DataFrame"""
        self.data = data
        self.max_length = max_len

    def process_data(self, tokens, coordinates):
        tokens = [self.START] + tokens[:self.max_length - 2] + [self.END]
        tokens += [self.PAD] * (self.max_length - len(tokens))
        inp = np.array([self.get_word_id(token) for token in tokens])
        coordinates = np.array(coordinates)
        return inp, coordinates

    def prepare_data(self):
        """Prepares data for training"""
        inputs = []
        labels = []
        coordinates = []

        for i, row in self.data.iterrows():
            text = row['type']
            coords = row['coords']
            label = self.label_dict[int(row['label'])]
            tokens = text[0].strip().split(' ')
            # dtypes = [self.type_dict[dtype] for dtype in text[1].split(',')]
            height = float(text[-2])
            width = float(text[-1])
            min_x = float(coords[0]) / width
            min_y = float(coords[1]) / height
            max_x = float(coords[2]) / width
            max_y = float(coords[3]) / height

            tokens = [self.START] + tokens[:self.max_length - 2] + [self.END]
            tokens += [self.PAD] * (self.max_length - len(tokens))
            inp = [self.get_word_id(token) for token in tokens]

            inputs.append(np.array(inp))
            labels.append(np.array(label))
            coordinates.append(np.array([min_x, min_y, max_x, max_y]))

        self.train_data['inputs'] = np.array(inputs)
        self.train_data['labels'] = np.array(labels)
        self.train_data['coordinates'] = np.array(coordinates)

    def load_embeddings(self, model_path):
        """Loads pre-trained gensim model"""
        print("\nLoading pre-trained embeddings...")

        model = Word2Vec.load(model_path)
        words = list(model.wv.vocab)
        embed_size = model.layer1_size

        embed = []
        word2idx = {self.PAD: 0, self.UNKNOWN: 1, self.START: 2, self.END: 3}
        idx2word = {0: self.PAD, 1: self.UNKNOWN, 2: self.START, 3: self.END}

        embed.append(np.zeros(embed_size, dtype=np.float32))
        embed.append(np.random.uniform(-0.1, 0.1, embed_size))
        embed.append(np.random.uniform(-0.1, 0.1, embed_size))
        embed.append(np.random.uniform(-0.1, 0.1, embed_size))

        for word in words:
            vector = model.wv[word]
            embed.append(vector)
            word2idx[word] = len(word2idx)
            idx2word[word2idx[word]] = word

        self.vocab_size = len(word2idx)
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.embeddings = np.array(embed, dtype=np.float32)

        print("\nSuccessfully loaded pre-trained embeddings!")

    def get_word_id(self, token):
        """Returns the id of a token"""
        token = token.lower()
        if token in self.word2idx:
            return self.word2idx[token]
        return self.word2idx[self.UNKNOWN]

    def save_data(self, out_path='./data/processed.pkl.gz'):
        """Saves the embeddings and vocab as a zipped pickle file"""
        assert (self.embeddings is not None or self.word2idx), "Data has not been processed yet"
        pkl = {'embeddings': self.embeddings,
               'word2idx': self.word2idx,
               'idx2word': self.idx2word
               }
        with gzip.open(out_path, 'wb') as out_file:
            pickle.dump(pkl, out_file)
        print("\nData stored as {}".format(out_path))

    def load_data(self, path):
        """Loads embeddings and vocab from a zipped pickle file"""
        with gzip.open(path, 'rb') as in_file:
            pkl = pickle.load(in_file)
        self.embeddings = pkl['embeddings']
        self.embed_size = self.embeddings.shape[1]
        self.word2idx = pkl['word2idx']
        self.vocab_size = len(self.word2idx)
        self.idx2word = pkl['idx2word']
        print("\nSuccessfully loaded data from {}".format(path))
