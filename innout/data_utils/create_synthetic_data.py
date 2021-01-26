"""
Generate synthetic data based on seq2seq tasks to stress test properties of various models.
"""

import argparse
import json
import os
import random
from extrapolation.utils import Bunch

def generate_char(n=None):
    if not n:
        n = args.vocab_size
    return str(random.randint(0, n - 1))

def generate_example(length, prefix):
    if prefix == 'test':
        x = ['w' + generate_char() for i in range(length)]
    else:
        x = []
        for i in range(length):
            # 50% probability of repeating the previous token instead of a new random token
            if i == 0 or random.random() < 0.1:
                c = 'w' + generate_char()
                x.append(c)
            else:
                x.append(x[i-1])

    # identity map
    y = [x[i] for i in range(length)]

    # x = []
    # y = []
    # for i in range(length):
    #     c = 'w' + generate_char()

    #     # Add repetitions
    #     assert args.source_repeat_stop_prob > 0
    #     while True:
    #         if args.source_synonymy > 1:
    #             x.append(c + '-' + generate_char(args.source_synonymy))
    #         else:
    #             x.append(c)
    #         if random.random() < args.source_repeat_stop_prob:
    #             break

    #     # Add noise
    #     assert args.source_noise_prob < 1
    #     while random.random() < args.source_noise_prob:
    #         x.append('n' + generate_char())

    #     # Output is clean
    #     y.append(c)

    return (x, y)

def generate_dataset(prefix, num_examples, length):
    src_out = open(os.path.join(args.out_dir, prefix + '.src'), 'w')
    tgt_out = open(os.path.join(args.out_dir, prefix + '.tgt'), 'w')
    for i in range(num_examples):
        x, y = generate_example(length, prefix)
        print(' '.join(x), file=src_out)
        print(' '.join(y), file=tgt_out)
    src_out.close()
    tgt_out.close()

def main(args_dict=None):
    if args_dict:
        # convert programmatic arg dict to namespace-like
        global args
        args = Bunch(args_dict)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    with open(os.path.join(args.out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    random.seed(args.rand_seed)
    generate_dataset('train', args.num_train_examples, args.length)
    generate_dataset('valid', args.num_valid_examples, args.length)
    generate_dataset('test', args.num_test_examples, int(args.length * args.test_length_factor))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', help='Length of sequences', type=int, default=10)
    parser.add_argument('--test-length-factor', help='Test sequences are this times longer', type=float, default=1)
    parser.add_argument('-v', '--vocab-size', help='Vocabulary size', type=int, default=5)
    parser.add_argument('-n', '--num-train-examples', help='Number of train examples to generate', type=int, default=1000)
    parser.add_argument('--num-valid-examples', help='Number of test examples to generate', type=int, default=1000)
    parser.add_argument('--num-test-examples', help='Number of test examples to generate', type=int, default=1000)

    parser.add_argument('--source-repeat-stop-prob', help='Geometric probability of repeating source token (e.g., a a a)', type=float, default=1)
    parser.add_argument('--source-noise-prob', help='Probability of adding noise token in the source', type=float, default=0)
    parser.add_argument('--source-synonymy', help='Number of variants of the same source token', type=int, default=1)

    parser.add_argument('-s', '--rand-seed', help='Random seed', type=int, default=1)
    parser.add_argument('-d', '--out-dir', help='Where to write all the files', default='.')
    args = parser.parse_args()

    main()
