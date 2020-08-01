#!/usr/bin/env python
import os
import copy
import torch
import torch.nn as nn
import logging
import config
import numpy as np

from pathlib import Path
from functools import partial

from vq_vae_text.models.textcnn import TextCNN
from vq_vae_text.datasets import ByteLevelTextDataset, BPEDataset, LMDBDataset
from vq_vae_text.utils import setup_logging_from_args, save_checkpoint
from vq_vae_text.trainer import ModelTrainer, SummaryWriterCallback

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR, ExponentialLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter

# Workaround for bug occurring with PyTorch and TF 2.0.
# See https://github.com/pytorch/pytorch/issues/30966#issuecomment-582747929
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def compute_reconstructions(model, batches, device, max_samples=None):
    codes = []
    recons = []
    n_processed = 0

    for batch in batches:
        batch = batch.to(device)

        logp, z, diff, ids = model(batch)

        codes.append([id.cpu() for id in ids])
        recons.append(logp.argmax(-1).cpu())

        n_processed += len(batch)
        if max_samples is not None and n_processed >= max_samples:
            break

    if max_samples is None:
        max_samples = n_processed

    recons = torch.cat(recons)[:max_samples]
    codes = [torch.cat(stage_code)[:max_samples] for stage_code in zip(*codes)]

    return codes, recons


def save_random_latent_modifications(filename, model, samples, quantization_layers, decode_func, mods_per_sample_and_latent=8):
    out = ''

    def do_random_change(sample, latent_idx):
        latents = [latent.squeeze() for latent in model.encode(sample.unsqueeze(0))[2]]
        mod_idx = np.random.randint(0, len(latents[latent_idx]))
        latents[latent_idx][mod_idx] = np.random.randint(0, quantization_layers[latent_idx].n_embed)

        dec = model.decode_code([latent.unsqueeze(0) for latent in latents])
        return decode_func(dec.argmax(-1)), mod_idx

    for sample in samples:
        decode_sample = model(sample.unsqueeze(0))[0].squeeze().argmax(-1)
        out += f'Applying random latent modification to sample:\n{decode_func(decode_sample)}\n\n'

        for latent_idx in range(len(quantization_layers)):
            results = []
            for _ in range(mods_per_sample_and_latent):
                results.append(do_random_change(sample, latent_idx))

            decodes, mod_indices = zip(*results)
            out += f'Applying random for latent index {latent_idx} at positions {mod_indices}\n'
            for decode in decodes:
                out += decode[0] + '\n'
            out += '\n'

    Path(filename).write_text(out)


def save_reconstructions(truth, recons, epoch, save_path, name):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    output = '\n\n'.join(f'{x}\n{y}' for x, y in zip(truth, recons))
    (Path(save_path) / f'{name}_{epoch}.txt').write_text(output)


def train(device, args):
    if args.debug and args.debug_overfit:
        raise Exception('--debug and --debug-overfit are mutually exclusive.')

    save_path = setup_logging_from_args(args)

    logging.info(f'Logging to {save_path}')

    logging.info('Loading data')


    data_files = ['train.txt', 'val.txt']
    #load_data = partial(ByteLevelTextDataset, seq_len=args.seq_length, pad_idx=0, eos_idx=1)
    #load_data = partial(BPEDataset, bpe_model='notebooks/model-lowercased.p', seq_len=args.seq_length)

    # data_files = ['train_latents', 'val_latents']
    # load_data = partial(LMDBDataset)

    dataset_builder = config.datasets.get(args.dataset)
    if not dataset_builder:
        raise Exception(f'Unknown dataset \'{args.dataset}\'. Valid options are {str(list(config.datasets.keys()))}')

    datasets_args = dict(eval(args.dataset_args))
    train_data, val_data = datasets = dataset_builder(**datasets_args)

    # train_data, val_data = datasets = [
    #     load_data(str(Path(args.data_dir) / name))
    #     for name in data_files
    # ]
    train_gen, val_gen = [
        DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        for dataset in datasets
    ]

    print(f'Vocab size: {train_data.vocab_size}')

    logging.info('Preparing model')

    model_cls = config.model_names.get(args.model)
    if not model_cls:
        raise ValueError(f'No model with name "{args.model}"')

    model_args = dict(
        vocab_size=train_data.vocab_size
    )
    model_args.update(config.model_configs[model_cls])
    if args.model_args:
        model_args.update(dict(eval(args.model_args)))
    if args.ignore_quant:
        model_args['ignore_quant'] = True

    logging.info(f'Model args: {model_args}')

    model = model_cls(**model_args).to(device)

    if args.data_parallel:
        model = nn.DataParallel(model, dim=0).to(device)

    logging.info(str(model))

    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, std=0.02)

    model.apply(weights_init)

    model.apply(weights_init)

    _, z, _, _ = model(next(iter(train_gen)).to(device))
    if not isinstance(z, (list, tuple)):
        z = [z]
    logging.info(f'Shape of latent space: {",".join(str(list(code.shape[1:])) for code in z)}')

    summary_writer = SummaryWriter(save_path)

    params = model.parameters()
    if args.sched == 'cycle':
        optimizer = torch.optim.SGD(params, lr=args.lr)
        scheduler = CyclicLR(optimizer,
                             mode='triangular2',
                             base_lr=args.lr / 100.0,
                             max_lr=args.lr,
                             gamma=args.gamma,
                             step_size_up=10000,
                             cycle_momentum=False)
        logging.info('Using cyclic learning rate scheduler')
    elif args.sched == '1cycle':
        optimizer = torch.optim.SGD(params, lr=args.lr)
        scheduler = OneCycleLR(optimizer, max_lr=5.0, total_steps=30000, div_factor=100, final_div_factor=100000)
        logging.info('Using one-cycle learning rate scheduler')
    else:
        from pytorch_lamb import Lamb
        #optimizer = Lamb(params, lr=args.lr, weight_decay=args.wdecay, min_trust=0.25)
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)

    trainer = ModelTrainer(model, optimizer, device, scheduler, [SummaryWriterCallback(summary_writer)])

    epochs = args.epochs
    if args.debug:
        logging.info('Debug mode set. Will train a single batch for a single epoch')
        epochs = 1
        train_gen = [next(iter(train_gen))]
        val_gen = [next(iter(val_gen))]

    if args.debug_overfit:
        logging.info('Debugging model by overfitting a single batch.')
        train_gen = [next(iter(train_gen))] * 1000

    recons_path = os.path.join(save_path, 'reconstructions')

    logging.info('Starting training')

    try:
        train_eval_batch = next(iter(train_gen)).to(device)
        valid_eval_batch = next(iter(val_gen)).to(device)

        for epoch in range(0, epochs):
            logging.info(f'Epoch {epoch + 1} - LR: {scheduler.get_lr()[0]:.8f}')

            train_loss = trainer.train(train_gen, epoch)
            val_loss = trainer.evaluate(val_gen)

            logging.info(f'Train loss: {train_loss:.4f}, validation loss: {val_loss:.4f}')

            if not args.debug:
                save_checkpoint(model, epoch, save_path)

            with torch.no_grad():
                model.eval()

                all_latents = []
                for batch, name in zip([train_eval_batch, valid_eval_batch], ['train', 'valid']):
                    latents, recons = compute_reconstructions(model, [batch], device)
                    save_reconstructions(
                        train_data.seq_to_text(batch),
                        train_data.seq_to_text(recons),
                        epoch,
                        recons_path,
                        f'{name}'
                    )

                    for i, stage_latents in enumerate(latents):
                        summary_writer.add_histogram(f'latents_{name}_{i}', stage_latents.view(-1), global_step=epoch)

                save_random_latent_modifications(Path(recons_path) / f'latent_mods_{epoch}.txt',
                                                 model,
                                                 valid_eval_batch[:8],
                                                 model.get_quantization_layers(),
                                                 train_data.seq_to_text)

                if args.num_compute_latents is not None:
                    latents, recons = compute_reconstructions(model, val_gen, device,
                                                              max_samples=args.num_compute_latents)

                    recons = train_data.seq_to_text(recons)
                    latents = latents.view(latents.size(0), -1)

                    logging.info('Writing evaluation latents')

                    summary_writer.add_embedding(latents, recons, global_step=epoch, tag='embeds_eval')

            summary_writer.flush()
    except KeyboardInterrupt:
        logging.info('Training interrupted')

    summary_writer.close()
