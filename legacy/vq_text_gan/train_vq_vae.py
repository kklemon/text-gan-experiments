import argparse
import logging
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import random

from pathlib import Path
from collections import OrderedDict, Counter

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vq_text_gan.utils import get_default_device, setup_run, setup_logging, compute_entropy, repr_list
from vq_text_gan.datasets import BPEDataset
from vq_text_gan.models.vq_vae import SingleStageTextVQVAE


def create_keys(key, num):
    return [f'{key}{i}' for i in range(num)]


def train(model, batches, optimizer, device, epoch, scheduler, steps_per_epoch, log_every):
    model.train()

    if steps_per_epoch:
        steps_per_epoch = min(len(batches), steps_per_epoch)
    else:
        steps_per_epoch = len(batches)

    batches = itertools.islice(batches, 0, steps_per_epoch)

    stats_collection = OrderedDict()

    def update_stat(key, value):
        stats_collection.setdefault(key, []).append(float(value))

    def get_mean_stats():
        return OrderedDict((k, np.mean(v)) for k, v in stats_collection.items())

    for batch_idx, batch in enumerate(batches):
        batch = batch.to(device)

        out = model(batch)
        loss = model.loss_function(out, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        latest_losses = model.latest_losses()
        for k, v in latest_losses.items():
            update_stat(k, v)

        if batch_idx % log_every == 0 or batch_idx + 1 == steps_per_epoch:
            mean_stats = get_mean_stats()

            logging.info(f'[TRAIN] [EPOCH {epoch:03d}] [{batch_idx:05d}/{steps_per_epoch:05d}] ' +
                         ', '.join(f'{k}: {v:.4f}' for k, v in mean_stats.items()))

            stats_collection.clear()

        yield batch_idx


@torch.no_grad()
def evaluate(model, batches, device, epoch):
    model.eval()

    stats_collection = OrderedDict()

    stats_collection = OrderedDict()

    def update_stat(key, value):
        stats_collection.setdefault(key, []).append(float(value))

    def get_mean_stats():
        return OrderedDict((k, np.mean(v)) for k, v in stats_collection.items())

    latent_count = Counter()

    for batch_idx, batch in enumerate(batches):
        batch = batch.to(device)

        out = model(batch)
        model.loss_function(out, batch)

        latest_losses = model.latest_losses()
        for k, v in latest_losses.items():
            update_stat(k, v)

        codes = out[-1]

        latent_count.update(codes.view(-1).detach().cpu().numpy().tolist())

    mean_stats = get_mean_stats()

    logging.info(f'[TRAIN] [EPOCH {epoch:03d}] ' + ', '.join(f'{k}: {v:.4f}' for k, v in mean_stats.items()))

    current_entropy = compute_entropy(latent_count)
    optimal_entropy = np.log2(model.num_vq_embeds)

    logging.info(f'[VALID] [EPOCH {epoch:03d}] Optimal entropy: {optimal_entropy:.2f}, current entropy: {current_entropy}')


def compute_reconstructions(model, input):
    logp, _, _, _ = model(input)
    return logp.argmax(-1)


def save_random_latent_modifications(filename, model, samples, decode_func, mods_per_sample=8):
    out = ''

    num_embeds = model.quantize.n_embed

    recons_p, _, _, codes = model(samples)
    recons = decode_func(recons_p.argmax(-1))

    for recon, code in zip(recons, codes):
        mod_code = code.unsqueeze(0).repeat(mods_per_sample, 1)

        rand_indices = torch.randint(0, mod_code.size(1), (len(mod_code), ))
        rand_mods = torch.randint(0, num_embeds, (len(mod_code), ), device=mod_code.device)

        mod_code[torch.arange(len(mod_code)), rand_indices] = rand_mods

        out += f'Applying random latent modification at positions ' \
               f'[{", ".join(map(str, rand_indices.numpy().tolist()))}] ' \
               f'reconstructed sample:\n{recon}\n\n'

        mods_decoded = decode_func(
            model.decode_code(mod_code).argmax(-1)
        )
        out += '\n'.join(mods_decoded)
        out += '\n\n'

    Path(filename).write_text(out)


def save_nearest_neighbor_latent_modifications(filename, model, samples, decode_func, mods_per_sample=8):
    out = ''

    embed = model.quantize.embed.T

    recons_p, _, _, codes = model(samples)
    recons = decode_func(recons_p.argmax(-1))

    for recon, code in zip(recons, codes):
        rand_index = random.randint(0, len(code) - 1)
        mod_index = code[rand_index]

        cosine_distances = F.cosine_similarity(embed[mod_index].unsqueeze(0), embed)
        closest_indices = cosine_distances.argsort()[:mods_per_sample]

        mod_code = code.unsqueeze(0).repeat(mods_per_sample, 1)
        mod_code[torch.arange(mods_per_sample), rand_index] = closest_indices

        out += f'Applying nearest neighbor latent modifications at position {rand_index} for' \
               f'reconstructed sample:\n{recon}\n\n'

        mods_decoded = decode_func(
            model.decode_code(mod_code).argmax(-1)
        )
        out += '\n'.join(mods_decoded)
        out += '\n\n'

    Path(filename).write_text(out)


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
        n_res_blocks=4,
        depth=args.depth,
        tau=1.0,
        n_heads=1,
        pad_idx=train_data.pad_idx,
        num_vq_embeds=2 ** 12,
        vq_embeds_dim=16,
        vq_loss_alpha=5.0,
    )
    if args.model_args:
        model_args.update(dict(eval(args.model_args)))

    logging.info(model_args)

    model = SingleStageTextVQVAE(**model_args).to(device)

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
                        # Save reconstructions
                        recon_pairs = [train_data.seq_to_text(torch.stack(list(recons)))
                                       for recons in zip(batch, compute_reconstructions(model, batch))]

                        recons_filename = sample_dir / f'reconstructions_{split}_{global_step:07d}.txt'
                        recons_filename.write_text(
                            '\n\n'.join(map(lambda g: '\n'.join(g), recon_pairs))
                        )

                        # Save random latent modifications
                        random_latents_mod_filename = sample_dir / f'random_latent_mods_{split}_{global_step:07d}.txt'
                        save_random_latent_modifications(
                            random_latents_mod_filename, model, batch, train_data.seq_to_text
                        )

                        # Save nearest neighbor latent modifications
                        nn_latents_mod_filename = sample_dir / f'nearest_neighbor_latent_mods_{split}_{global_step:07d}.txt'
                        save_nearest_neighbor_latent_modifications(
                            nn_latents_mod_filename, model, batch, train_data.seq_to_text
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
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--run-name', type=str, default='vq-vae')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--steps-per-epoch', type=int, default=10_000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--sample-every', type=int, default=1_000)
    parser.add_argument('--max-samples', type=int, default=8)
    parser.add_argument('--model-args', type=str)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args)
