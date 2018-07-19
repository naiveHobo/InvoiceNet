import pickle
from data_handler import DataHandler
from model import InvoiceNet
import argparse


ap = argparse.ArgumentParser()

ap.add_argument("--mode", type=str, help="train|test", choices=["train", "test"], required=True)
ap.add_argument("--data", help="path to training data", default="data/train_api.pk")
ap.add_argument("--model_path", default="./model", help="path to directory where trained model should be stored")
ap.add_argument("--load_weights", default="./checkpoints/InvoiceNet_.157-0.53-0.48.hdf5", help="path to load weights")
ap.add_argument("--word2vec", default="model.bin", help="path to word2vec model")
ap.add_argument("--checkpoint_dir", default="./checkpoints", help="path to directory where checkpoints should be stored")
ap.add_argument("--log_dir", default="./logs", help="path to directory where tensorboard logs should be stored")
ap.add_argument("--num_epochs", type=int, default=200, help="number of epochs")
ap.add_argument("--num_hidden", type=int, default=128, help="size of hidden layer")
ap.add_argument("--num_filters", type=int, default=100, help="number of filters")
ap.add_argument("--batch_size", type=int, default=64, help="size of mini-batch")
ap.add_argument("--reg_rate", type=float, default=0.0001, help="rate of regularization")
ap.add_argument("--shuffle", action='store_true', help="shuffle dataset")

args = ap.parse_args()


label_dict = {0: "Other",
              1: "Invoice Date",
              2: "Invoice Number",
              3: "Buyer GST",
              4: "Seller GST",
              5: "Total Amount"}

with open(args.data, 'rb') as pklfile:
    df = pickle.load(pklfile)

data = DataHandler(df, max_len=12)
data.load_embeddings(args.word2vec)
data.prepare_data()

print(data.train_data['inputs'].shape)
print(data.train_data['labels'].shape)
print(data.train_data['coordinates'].shape)

net = InvoiceNet(data_handler=data, config=args)

if args.mode == 'train':
    net.train()
else:
    net.load_weights(args.load_weights)
    predictions = net.evaluate()
    net.f1_score(predictions)
    for i in range(predictions.shape[0]):
        print(predictions[i], net.data_handler.train_data['labels'][i], df.iloc[i])

