#!/usr/bin/env python
import sys
import argparse
import torch
import logging

from vq_vae_text.train import train
from vq_vae_text.extract import extract_codes


sos_token = '◁'
eos_token = '▷'
pad_token = '﹏'
special_tokens = [sos_token, eos_token, pad_token]


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', help='Device to use.')

    subparsers = parser.add_subparsers(dest='subcommand')

    train_parser = subparsers.add_parser('train')

    # Model arguments
    train_parser.add_argument('--model', '-m', required=True,
                              help='Name of model to use. See configs.py')
    train_parser.add_argument('--dataset', type=str, required=True,
                              help='Name of dataset to use. See configspy')
    train_parser.add_argument('--dataset-args', type=str,
                              help='Arguments to pass to dataset builder function. See config.py.')
    #train_parser.add_argument('--seq-length', type=int, required=True)
    train_parser.add_argument('--model-args', type=str, default=None)
    train_parser.add_argument('--ignore-quant', action='store_true')
    train_parser.add_argument('--wdecay', type=float, default=1.2e-6,
                              help='weight decay applied to all weights')

    # Training arguments
    train_parser.add_argument('--batch-size', '-bs', type=int, dest='batch_size', default=128,
                              help='Batch size for training.')
    train_parser.add_argument('--lr', type=float, default=0.0003,
                              help='Initial learning rate.')
    train_parser.add_argument('--gamma', type=float, default=0.80,
                              help='Learning rate decay gamma.')
    train_parser.add_argument('--epochs', type=int, default=10,
                              help='Number of training epochs.')
    train_parser.add_argument('--debug', action='store_true',
                              help='Whether to make a debug run.'
                                   'In this case, only a small fraction of the data will be loaded and used.')
    train_parser.add_argument('--debug-overfit', action='store_true', dest='debug_overfit',
                              help='Whether to overfit on a single batch for debugging purposes.')
    train_parser.add_argument('--data-parallel', action='store_true',
                              help='Use data parallelism for multi GPU training.')
    train_parser.add_argument('--sched', choices=['cycle', '1cycle'], default=None)

    # Logging arguments
    train_parser.add_argument('--log-level', '-l', dest='log_level', default='INFO',
                              help='Log level.')
    train_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                              help='how many batches to wait before logging training status')
    train_parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
                              help='Directory for run logs.')
    train_parser.add_argument('--save-name', default='',
                              help='Name of folder to save run by. Defaults to timestamp.')
    train_parser.add_argument('--data-format', default='json',
                              help='In which format to save the data')
    train_parser.add_argument('--num-compute-latents', type=int, default=None,
                              help='Number of evaluation latents to compute and log for each epoch.')
    train_parser.add_argument('--comment', type=str)

    extract_parser = subparsers.add_parser('extract_codes')
    extract_parser.add_argument('--checkpoint', '--ckpt', type=str, dest='ckpt', required=True)
    extract_parser.add_argument('--data-path', type=str, required=True)
    extract_parser.add_argument('--seq-length', type=int, required=True)
    extract_parser.add_argument('--save-path', type=str, default=None)
    extract_parser.add_argument('--batch-size', type=int, default=128)

    args = parser.parse_args(args)

    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info(f'Using device {device}')

    if args.subcommand == 'train':
        train(device, args)
    if args.subcommand == 'extract_codes':
        extract_codes(device, args)


if __name__ == '__main__':
    main(sys.argv[1:])
