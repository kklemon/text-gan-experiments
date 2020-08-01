import os
import math
import random
import argparse
import itertools
import tempfile
import pickle
import toolz
import youtokentome as yttm

from pathlib import Path
from unidecode import unidecode
from collections import Counter

from vq_text_gan.utils import compute_entropy


def combine_files(files, process_line=None):
    if process_line is None:
        process_line = lambda x: x

    # Load and chain all lines from all files
    lines = itertools.chain.from_iterable(map(lambda f: f.read_text().split('\n'), files))

    # Remove empty lines and apply process function
    lines = map(process_line, filter(bool, lines))

    return list(lines)


def create_subset(samples, filter_cond, max_samples, shuffle=True):
    filtered = list(filter(filter_cond, samples))
    if shuffle:
        random.shuffle(filtered)
    return filtered[:max_samples]


def save_subset(save_path, samples, seq_length, bpe_path):
    save_path.write_bytes(
        pickle.dumps(dict(
            bpe_path=str(bpe_path),
            samples=samples,
            seq_length=seq_length
        ))
    )
    print(f'Wrote {len(samples)} samples to {save_path}')


def prepare_gigaword(args):
    root_dir = Path(args.root_dir)
    target_dir = Path(args.target_dir)

    print(f'Loading raw data from {root_dir}')

    train_files = list((root_dir / 'training-monolingual.tokenized.shuffled').glob('*'))
    test_files = list((root_dir / 'heldout-monolingual.tokenized.shuffled').glob('*'))

    process_line = toolz.curried.identity
    if args.lower:
        process_line = toolz.curried.compose(str.lower, process_line)
    if args.unidecode:
        process_line = toolz.curried.compose(unidecode, process_line)

    train_samples = combine_files(train_files, process_line=process_line)
    test_samples = combine_files(test_files, process_line=process_line)

    print(f'Found {len(train_samples)} training and {len(test_samples)} heldout samples')
    print(f'Number of unique characters: {len(set(itertools.chain.from_iterable(train_samples)))}')

    if args.use_bpe_model:
        bpe_path = args.use_bpe_model
        bpe = yttm.BPE(bpe_path)
    else:
        with tempfile.NamedTemporaryFile('w', delete=False) as fp:
            tmp_path = fp.name
            fp.write('\n'.join(train_samples))

        print('Training BPE model')

        bpe_path = target_dir / f'{args.save_prefix}.model'
        bpe = yttm.BPE.train(data=tmp_path, model=str(bpe_path), vocab_size=args.bpe_vocab_size)

        os.remove(tmp_path)

    train_encoded = bpe.encode(train_samples, eos=True)
    test_encoded = bpe.encode(test_samples, eos=True)

    filter_condition = lambda x: args.min_len <= len(x) <= args.max_len

    train_subset = create_subset(train_encoded, filter_condition, args.max_train_samples)
    test_subset = create_subset(test_encoded, filter_condition, args.max_test_samples)

    if args.compute_entropy_stats:
        print('Computing entropy statistics...')

        train_counts = Counter(itertools.chain.from_iterable(train_subset))
        test_counts = Counter(itertools.chain.from_iterable(test_subset))

        train_entropy = compute_entropy(train_counts)
        test_entropy = compute_entropy(test_counts)

        print(f'Optimal entropy of vocabulary: {math.log2(bpe.vocab_size()):.4f} bpc')
        print(f'Entropy of training subset: {train_entropy:.4f} bpc')
        print(f'Entropy of test subset: {test_entropy:.4f} bpc')

    train_save_path = target_dir / (args.save_prefix + '_train.p')
    test_save_path = target_dir / (args.save_prefix + '_test.p')

    save_subset(train_save_path, train_subset, args.max_len, bpe_path)
    save_subset(test_save_path, test_subset, args.max_len, bpe_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='dataset', required=True)

    gigaword_args = subparsers.add_parser('gigaword')
    gigaword_args.add_argument('root_dir', type=str)
    gigaword_args.add_argument('target_dir', type=str, default='data')
    gigaword_args.add_argument('--lower', action='store_true')
    gigaword_args.add_argument('--unidecode', action='store_true')
    gigaword_args.add_argument('--use-bpe-model', type=str)
    gigaword_args.add_argument('--bpe-vocab-size', type=int, default=256)
    gigaword_args.add_argument('--max-len', type=int, default=64)
    gigaword_args.add_argument('--min-len', type=int, default=32)
    gigaword_args.add_argument('--max-train-samples', type=int, default=5_000_000)
    gigaword_args.add_argument('--max-test-samples', type=int, default=100_000)
    gigaword_args.add_argument('--save-prefix', type=str, default='gigaword')
    gigaword_args.add_argument('--compute-entropy-stats', action='store_true')
    gigaword_args.add_argument('--shuffle', action='store_true')

    args = parser.parse_args()

    if args.dataset == 'gigaword':
        prepare_gigaword(args)
