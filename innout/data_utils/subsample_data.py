"""
Subsample a dataset to get learning curves.
Specifically, take a random fraction of the training examples.
"""

import argparse
import json
import os
import random
from extrapolation.utils import Bunch, readlines, writelines

def process(name, keep_frac):
    # Read
    src_lines = readlines(os.path.join(args.input_dir, name + '.src'))
    tgt_lines = readlines(os.path.join(args.input_dir, name + '.tgt'))
    print('Read {} source and {} target lines'.format(len(src_lines), len(tgt_lines)))
    assert len(src_lines) == len(tgt_lines)

    # Randomly permute and take first keep_frac examples.
    examples = list(zip(src_lines, tgt_lines))
    random.shuffle(examples)
    end = int(keep_frac * len(examples))
    sub_examples = examples[:end]

    # Output
    writelines([src for src, tgt in sub_examples], os.path.join(args.output_dir, name + '.src'))
    writelines([tgt for src, tgt in sub_examples], os.path.join(args.output_dir, name + '.tgt'))

def main(args_dict=None):
    if args_dict:
        # convert programmatic arg dict to namespace-like
        global args
        args = Bunch(args_dict)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    random.seed(args.rand_seed)
    process('train', args.train_frac)
    process('valid', 1)
    process('test', 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--rand-seed', help='Random seed', type=int, default=1)
    parser.add_argument('-f', '--train-frac', help='Fraction of the training set to keep', type=float, default=1)
    parser.add_argument('-i', '--input-dir', help='Directory containing input dataset', required=True)
    parser.add_argument('-o', '--output-dir', help='Directory containing output dataset', required=True)
    args = parser.parse_args()

