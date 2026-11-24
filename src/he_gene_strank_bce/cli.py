"""Command-line entry point for training, inference and the two protocols."""

import argparse
import sys

from . import predict, protocols, train


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="he-gene-strank-bce",
        description="Train or apply STRank+BCE and evaluate its gene selections.",
    )
    parser.add_argument("command", choices=("train", "predict", "calibrate", "evaluate"))
    command = parser.parse_args(args[:1]).command
    if command == "train":
        return train.main(args[1:])
    if command == "predict":
        return predict.main(args[1:])
    return protocols.main(args)
