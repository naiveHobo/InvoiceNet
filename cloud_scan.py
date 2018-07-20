import argparse
from model import InvoiceNetCloudScan
from extract_features import extract_features


ap = argparse.ArgumentParser()

ap.add_argument("--mode", type=str, choices=["train", "test"],
                required=True, help="train|test")
ap.add_argument("--data", default="data/dftrain.pk",
                help="path to training data")
ap.add_argument("--model_path", default="./model",
                help="path to directory where trained model should be stored")
ap.add_argument("--load_weights", default="./model/InvoiceNetCloudScan.model",
                help="path to load weights")
ap.add_argument("--checkpoint_dir", default="./checkpoints",
                help="path to directory where checkpoints should be stored")
ap.add_argument("--log_dir", default="./logs",
                help="path to directory where tensorboard logs should be stored")
ap.add_argument("--num_hidden", type=int, default=256,
                help="size of hidden layer")
ap.add_argument("--epochs", type=int, default=20,
                help="number of epochs")
ap.add_argument("--batch_size", type=int, default=128,
                help="size of mini-batch")
ap.add_argument("--num_layers", type=int, default=1,
                help="number of layers")
ap.add_argument("--num_input", type=int, default=17,
                help="size of input layer")
ap.add_argument("--num_output", type=int, default=4,
                help="size of output layer")
ap.add_argument("--shuffle", action='store_true',
                help="shuffle dataset")
ap.add_argument("--oversample", type=int, default=0,
                help="oversample minority classes to prevent class imbalance")

args = ap.parse_args()

net = InvoiceNetCloudScan(config=args)
features = extract_features(args.data)

if args.mode == 'train':
    net.train(features)
else:
    net.load_weights(args.load_weights)
    predictions = net.evaluate(features)
    net.f1_score(predictions, features.label.values)
    # for i in range(predictions.shape[0]):
    #     print(predictions[i], features.label.values[i], features.iloc[i])
