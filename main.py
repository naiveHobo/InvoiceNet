import pickle
from data_handler import DataHandler
from model import InvoiceNet
import argparse


ap = argparse.ArgumentParser()

ap.add_argument("--model_path", default="./model", help="path to directory where trained model should be stored")
ap.add_argument("--checkpoint_dir", default="./checkpoints", help="path to directory where checkpoints should be stored")
ap.add_argument("--log_dir", default="./logs", help="path to directory where tensorboard logs should be stored")
ap.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
ap.add_argument("--num_hidden", type=int, default=128, help="size of hidden layer")
ap.add_argument("--num_filters", type=int, default=100, help="number of filters")
ap.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
ap.add_argument("--batch_size", type=int, default=32, help="size of mini-batch")
ap.add_argument("--reg_rate", type=float, default=0.0001, help="rate of regularization")
ap.add_argument("--shuffle", action='store_true', help="shuffle dataset")

args = ap.parse_args()

with open('df_train_api.pk', 'rb') as pklfile:
    df = pickle.load(pklfile)

data = DataHandler(df, max_len=12)
data.load_embeddings('model.bin')
data.prepare_training_data()

print(data.train_data['inputs'].shape)
print(data.train_data['labels'].shape)
print(data.train_data['midpoints'].shape)

net = InvoiceNet(data_handler=data, config=args)

# v = []
# for i, row in df.iterrows():
#     text = row['type']
#     if len(text[0].split(' ')) != len(text[1].split(',')):
#         print(text)
#     for t in row['type'][1].split(','):
#         if t not in v:
#             v.append(t)
#
# print(v)