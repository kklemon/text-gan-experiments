import argparse
import logging
import itertools
import numpy as np
import torch

from collections import OrderedDict, Counter

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vq_text_gan.utils import get_default_device, setup_run, setup_logging, compute_entropy, repr_list
from vq_text_gan.datasets import BPEDataset
from vq_text_gan.models.vq_vae import MultiStageTextVQVAE


def create_keys(key, num):
    return [f'{key}{i}' for i in range(num)]


def train(model, batches, optimizer, device, epoch, scheduler, steps_per_epoch, log_every):
    model.train()

    if steps_per_epoch:
        steps_per_epoch = min(len(batches), steps_per_epoch)
    else:
        steps_per_epoch = len(batches)

    batches = itertools.islice(batches, 0, steps_per_epoch)

    acc_keys, nll_keys, ppl_keys, bpc_keys, vq_loss_keys = [create_keys(key, model.depth)
                                                            for key in ['acc', 'nll', 'ppl', 'bpc', 'vq']]

    stats_collection = OrderedDict()

    def add_stat(key, value):
        stats_collection.setdefault(key, []).append(float(value))

    def get_mean_stats(keyset):
        return [np.mean(values) for k, values in stats_collection.items() if k in keyset]

    for batch_idx, batch in enumerate(batches):
        batch = batch.to(device)

        out = model(batch)
        loss, stats = model.loss_function(out, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        for k, v in stats.items():
            add_stat(k, v)

        if batch_idx % log_every == 0 or batch_idx + 1 == steps_per_epoch:
            nll_sum = np.mean(stats_collection['nll_sum'])
            acc_avg = np.mean(stats_collection['acc_avg'])

            accuracies = get_mean_stats(acc_keys)
            likelihoods = get_mean_stats(nll_keys)
            perplexities = np.exp(likelihoods)
            bpcs = np.divide(likelihoods, np.log(2))

            logging.info(f'[TRAIN] [EPOCH {epoch:03d}] [{batch_idx:05d}/{steps_per_epoch:05d}] ' +
                         f'nll_sum: {nll_sum:.4f}, acc_avg: {acc_avg:.4f}, ' +
                         f'nll: [{repr_list(likelihoods)}], acc: [{repr_list(accuracies)}], ' +
                         f'ppl: [{repr_list(perplexities)}], bpc: [{repr_list(bpcs)}], ' +
                         f'lr: {scheduler.get_lr()[0]:.4f}')

            stats_collection.clear()

        yield batch_idx


@torch.no_grad()
def evaluate(model, batches, device, epoch):
    model.eval()

    stats_collection = OrderedDict()

    def add_stat(key, value):
        stats_collection[key] = stats_collection.get(key, 0) + float(value)

    acc_keys, nll_keys, ppl_keys, bpc_keys = [create_keys(key, model.depth) for key in ['acc', 'nll', 'ppl', 'bpc']]

    def get_keyset(keyset):
        return [values for k, values in stats_collection.items() if k in keyset]

    latent_counts = [Counter() for _ in range(model.depth)]

    for batch_idx, batch in enumerate(batches):
        batch = batch.to(device)

        out = model(batch)
        loss, stats = model.loss_function(out, batch)

        for k, v in stats.items():
            add_stat(k, v)

        codes = out[-1]

        for codes_for_depth, latent_count in zip(codes, latent_counts):
            latent_count.update(codes_for_depth.view(-1).detach().cpu().numpy().tolist())

    for k in stats_collection:
        stats_collection[k] /= len(batches)

    accuracies = get_keyset(acc_keys)
    likelihoods = get_keyset(nll_keys)
    perplexities = np.exp(likelihoods)
    bpcs = np.divide(likelihoods, np.log(2))

    nll_sum = stats_collection['nll_sum']
    acc_avg = stats_collection['acc_avg']

    logging.info(f'[VALID] [EPOCH {epoch:03d}] ' +
                 f'nll_sum: {nll_sum:.4f}, acc_avg: {acc_avg:.4f}, ' +
                 f'nll: [{repr_list(likelihoods)}], acc: [{repr_list(accuracies)}], '
                 f'ppl: [{repr_list(perplexities)}], bpc: [{repr_list(bpcs)}]')

    entropies = [compute_entropy(latent_counts) for latent_counts in latent_counts]
    optimal_entropy = np.log2(model.num_vq_embeds)

    logging.info(f'[VALID] [EPOCH {epoch:03d}] Optimal entropy: {optimal_entropy:.2f}, '
                 f'depth wise entropies: {repr_list(entropies)}')


def compute_reconstructions(model, input):
    logps, _, _, _ = model(input)
    return [logp.argmax(-1) for logp in logps]


def main(args):
    result_dir = setup_run(args.run_name, create_dirs=['checkpoints', 'reconstructions'])
    setup_logging(result_dir / 'log.txt')

    logging.info(args)

    device = get_default_device(args.device)

    sample_dir = result_dir / 'reconstructions'
    checkpoint_dir = result_dir / 'checkpoints'

    train_data = BPEDataset(args.train_data)
    valid_data = BPEDataset(args.valid_data)

    if args.debug:
        args.num_workers = 0

    train_batches = DataLoader(train_data,
                               batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_batches = DataLoader(valid_data,
                               batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    logging.info(f'Loaded {len(train_data)} training samples')
    logging.info(f'Loaded {len(valid_data)} validation samples')

    vocab_size = train_data.vocab_size
    logging.info(f'Vocab size: {vocab_size}')

    logging.info('Preparing model')

    model_args = dict(
        vocab_size=vocab_size,
        channel=256,
        res_channel=64,
        n_res_blocks=2,
        depth=4,
        tau=1.0,
        n_heads=1,
        pad_idx=train_data.pad_idx,
        num_vq_embeds=2 ** 12,
        vq_embeds_dim=64,
        vq_loss_alpha=1.0,
    )
    if args.model_args:
        model_args.update(dict(eval(args.model_args)))

    logging.info(model_args)

    model = MultiStageTextVQVAE(**model_args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.steps_per_epoch, args.gamma)

    # Write model architecture
    (result_dir / 'model.txt').write_text(str(model))

    summary_writer = SummaryWriter(str(result_dir))

    train_sample_batch = next(iter(train_batches))[:args.max_samples].to(device)
    valid_sample_batch = next(iter(valid_batches))[:args.max_samples].to(device)

    try:
        global_step = 0

        for epoch in range(args.epochs):
            for batch_idx in train(model, train_batches, optimizer, device, epoch, scheduler,
                                   args.steps_per_epoch, args.log_every):
                if global_step % args.sample_every == 0:
                    for split, batch in zip(['train', 'valid'], [train_sample_batch, valid_sample_batch]):
                        groups = [train_data.seq_to_text(torch.stack(list(recons)))
                                  for recons in zip(batch, *compute_reconstructions(model, batch))]
                        [g.insert(1, '+' * len(g[0])) for g in groups]

                        (sample_dir / f'{split}_{global_step:07d}.txt').write_text(
                            '\n\n'.join(map(lambda g: '\n'.join(g), groups))
                        )

                if args.debug:
                    break

                global_step += 1

            evaluate(model, valid_batches, device, epoch)

            torch.save(model, str(checkpoint_dir / f'model_{epoch:03d}.pth'))
    except KeyboardInterrupt:
        logging.info('Aborting training.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='data/gigaword_train.p')
    parser.add_argument('--valid-data', type=str, default='data/gigaword_test.p')
    parser.add_argument('--device', type=str)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--run-name', type=str, default='vq-vae')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps-per-epoch', type=int, default=10_000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--sample-every', type=int, default=1_000)
    parser.add_argument('--max-samples', type=int, default=8)
    parser.add_argument('--model-args', type=str)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args)
