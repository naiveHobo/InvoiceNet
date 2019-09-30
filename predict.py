import os
import glob
import pickle
import argparse
import marshal
import numpy as np

from tools.data import get_data


model_file = open(os.path.join('tools', 'model.pyc'), 'rb')
model_file.seek(12)
model_obj = marshal.load(model_file)
exec(model_obj)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--height", type=int, default=128,
                    help="height of invoice image")
    ap.add_argument("--width", type=int, default=128,
                    help="width of invoice image")
    ap.add_argument("--seq_length", type=int, default=128,
                    help="maximum length of sequence")
    ap.add_argument("--ngram_length", type=int, default=4,
                    help="length of ngrams")
    ap.add_argument("--embeddings_path", type=str, default="vocab+embeddings.pkl",
                    help="path to file containing embeddings")
    ap.add_argument("--model_path", type=str, default="model/invoicenet",
                    help="path to trained model")
    ap.add_argument("--data_dir", type=str, required=True,
                    help="path to directory containing invoice document images")

    args = ap.parse_args()

    if args.data_dir[0] != '/':
        args.data_dir += '/'

    filenames = [os.path.abspath(f) for f in glob.glob(args.data_dir + "**/*.pdf", recursive=True)]

    filenames = filenames[5:10]

    with open(args.embeddings_path, 'rb') as f:
        data = pickle.load(f)

    char_vocab = data["char_vocab"]
    pattern_vocab = data["pattern_vocab"]

    char_embeddings = data["char_embeddings"]
    pattern_embeddings = data["pattern_embeddings"]

    print("Loaded trained embeddings and vocabulary from [{}]".format(args.embeddings_path))

    data, num_samples = get_data(data_path=filenames, training=False, key='invoice_number',
                                 char_vocab=char_vocab, pattern_vocab=pattern_vocab,
                                 height=args.height, width=args.width,
                                 seq_length=args.seq_length, ngram_length=args.ngram_length, batch_size=1)

    model = InvoiceNet(height=args.height, width=args.width, channels=158,
                       ngram_len=args.ngram_length, seq_len=args.seq_length, n_out=len(char_vocab),
                       char_embeddings=char_embeddings,
                       pattern_embeddings=pattern_embeddings)

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy")

    model.load_weights(args.model_path)
    print("Loaded InvoiceNet model from [{}]".format(args.model_path))

    predictions = model.predict(data, steps=num_samples)
    predictions = np.argmax(predictions, axis=2)

    for filename, prediction in zip(filenames, predictions):
        print("Filename: {}".format(filename[:-3] + 'jpg'))
        pred = ''
        for idx in prediction:
            if idx == 0:
                break
            else:
                pred += char_vocab[int(idx)]
        print("Invoice Number: {}".format(pred))
        print("\n\n")


if __name__ == '__main__':
    main()
