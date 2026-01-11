"""Compress trained regressors for meta-learning."""
import argparse
import bz2
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Converting traces")
    parser.add_argument("--input", type=str, help="input path", required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input, 'rb') as f:
        regressors = pickle.load(f)
    with bz2.BZ2File(args.input + '.pbz2', 'w') as f:
        pickle.dump(regressors, f)


if __name__ == "__main__":
    main()