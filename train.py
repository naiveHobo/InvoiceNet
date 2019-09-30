import os
import glob
import pickle
import argparse
import marshal
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tools.data import get_data


model_file = open(os.path.join('tools', 'model.pyc'), 'rb')
model_file.seek(12)
model_obj = marshal.load(model_file)
exec(model_obj)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True,
                    help="directory containing training data")
    ap.add_argument("--epochs", type=int, default=10,
                    help="number of epochs for training")
    ap.add_argument("--height", type=int, default=128,
                    help="height of invoice image")
    ap.add_argument("--width", type=int, default=128,
                    help="width of invoice image")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="batch size for training")
    ap.add_argument("--seq_length", type=int, default=128,
                    help="maximum length of sequence")
    ap.add_argument("--ngram_length", type=int, default=4,
                    help="length of ngrams")
    ap.add_argument("--learning_rate", type=float, default=0.001,
                    help="learning rate for training")
    ap.add_argument("--val_size", type=float, default=0.2,
                    help="fraction of training data to be used as validation data")
    ap.add_argument("--embeddings_path", type=str, default="vocab+embeddings.pkl",
                    help="path to file containing embeddings")
    ap.add_argument("--save_model", type=str, default="model/invoicenet",
                    help="save model as")
    ap.add_argument("--log_dir", type=str, default="logs/",
                    help="path to save tensorboard logs")
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints/",
                    help="directory to store model checkpoints")

    args = ap.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    with open(args.embeddings_path, 'rb') as f:
        data = pickle.load(f)

    char_vocab = data["char_vocab"]
    pattern_vocab = data["pattern_vocab"]

    char_embeddings = data["char_embeddings"]
    pattern_embeddings = data["pattern_embeddings"]

    if args.data_dir[-1] != '/':
        args.data_dir += '/'
    filenames = [os.path.abspath(f) for f in glob.glob(args.data_dir + "**/*.json", recursive=True)]

    train_files, val_files = train_test_split(filenames, test_size=args.val_size)

    train_dataset, train_num_samples = get_data(data_path=train_files, key='invoice_number', training=True,
                                                char_vocab=char_vocab, pattern_vocab=pattern_vocab,
                                                height=args.height, width=args.width,
                                                seq_length=args.seq_length, ngram_length=args.ngram_length,
                                                batch_size=args.batch_size, shuffle=True)

    val_dataset, val_num_samples = get_data(data_path=val_files, key='invoice_number',
                                            char_vocab=char_vocab, pattern_vocab=pattern_vocab,
                                            height=args.height, width=args.width,
                                            seq_length=args.seq_length, ngram_length=args.ngram_length,
                                            batch_size=args.batch_size)

    model = InvoiceNet(height=args.height, width=args.width, channels=158,
                       ngram_len=args.ngram_length, seq_len=args.seq_length, n_out=len(char_vocab),
                       char_embeddings=char_embeddings,
                       pattern_embeddings=pattern_embeddings)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    print(model.summary())

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=args.log_dir)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                     patience=5, min_lr=0.0001)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.checkpoint_dir, 'invoicenet-{epoch}-{val_loss:.4f}.h5'),
        save_best_only=True, monitor='val_loss', verbose=1)

    model.fit(train_dataset, epochs=args.epochs,
              steps_per_epoch=train_num_samples // args.batch_size,
              validation_data=val_dataset,
              validation_steps=val_num_samples // args.batch_size,
              callbacks=[tensorboard, reduce_lr, checkpoint])

    model.save_weights(args.save_model, save_format='tf')
    print("Saved InvoiceNet model as [{}]".format(args.save_model))


if __name__ == '__main__':
    main()
