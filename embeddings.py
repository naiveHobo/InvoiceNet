import os
import glob
import argparse

from tools.utils import create_embeddings


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True,
                    help="path containing data")
    ap.add_argument("--save_as", type=str, default="vocab+embeddings.pkl",
                    help="pickle and save vocabulary and embeddings")

    args = ap.parse_args()

    if args.data_dir[0] != '/':
        args.data_dir += '/'

    filenames = [os.path.abspath(f) for f in glob.glob(args.data_dir + "**/*.pdf", recursive=True)]

    create_embeddings(filenames, save_as=args.save_as)


if __name__ == '__main__':
    main()
