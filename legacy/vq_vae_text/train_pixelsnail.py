import argparse
import logging
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from itertools import islice
from pathlib import Path

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vq_vae_text.utils import setup_logging_from_args, save_checkpoint
from vq_vae_text.datasets import LMDBDataset, ByteLevelTextDataset
from vq_vae_text.models.pixelsnail import PixelSNAIL

try:
    import apex
    from apex import amp
except ImportError:
    amp = None

amp = None


def load_data(path, args):
    dataset = LMDBDataset(path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return data_loader


def log_summary(summary_writer, prefix, tag_value_pairs, step):
    for tag, val in tag_value_pairs:
        summary_writer.add_scalar(prefix + tag, val, global_step=step)


def compute_accuracy(x, x_hat):
    correct = (x == x_hat).float()
    accuracy = correct.sum() / x.numel()
    return accuracy


def train(model, optimizer, scheduler, criterion, data_loader, target_shape, summary_writer, device, epoch):
    model.train()

    if args.steps_per_epoch is not None:
        data_loader = islice(data_loader, 0, args.steps_per_epoch)
        pbar = tqdm(data_loader, total=args.steps_per_epoch)
    else:
        pbar = tqdm(data_loader)

    for idx, batch in enumerate(pbar):
        model.zero_grad()

        batch = batch.view(batch.size(0), *target_shape).to(device)

        out, _ = model(batch)

        loss = criterion(out, batch)

        if amp is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        accuracy = compute_accuracy(batch, out.argmax(1))

        lr = optimizer.param_groups[0]['lr']

        pbar.set_description(f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                             f'acc: {accuracy:.5f}; lr: {lr:.5f}')

        #step = optimizer.state[optimizer.param_groups[0]['params'][0]]['step']
        #log_summary(summary_writer, 'train_', zip(['loss', 'acc', 'lr'], [loss, accuracy, lr]), step=step)


@torch.no_grad()
def evaluate(model, criterion, data_loader, target_shape, summary_writer, device, epoch):
    model.eval()

    loss_accum = []
    acc_accum = []

    for batch in data_loader:
        batch = batch.view(batch.size(0), *target_shape).to(device)

        out, _ = model(batch)
        loss = criterion(out, batch)

        accuracy = compute_accuracy(batch, out.argmax(1))

        loss_accum.append(loss.item())
        acc_accum.append(accuracy.item())

    loss = np.mean(loss_accum)
    acc = np.mean(acc_accum)

    log_summary(summary_writer, 'val_', zip(['loss', 'acc'], [loss, acc]), step=epoch)


@torch.no_grad()
def sample_latents(latent_predictor, device, size, num, temperature):
    row = torch.zeros(num, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = latent_predictor(row[:, : i + 1, :], cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


@torch.no_grad()
def sample(latent_predictor, ae_model, device, latent_size, num, temperature):
    latents = sample_latents(latent_predictor, device, latent_size, num, temperature)
    latents = latents.view(latents.size(0), -1)

    pred = ae_model.decode_codes(latents).cpu()
    pred = pred.argmax(-1)
    texts = ByteLevelTextDataset.seq_to_text(pred)

    return texts


def save_samples(samples, epoch, result_dir):
    samples_dir = Path(result_dir) / 'samples'
    samples_dir.mkdir(exist_ok=True)

    file = samples_dir / f'samples_{epoch}.txt'
    file.write_text('\n\n'.join(samples))


def main(args):
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.data_parallel and amp is not None:
        logging.info('Using APEX Distributed Data Parallel training')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

    save_path = setup_logging_from_args(args)

    logging.info(f'Using device {device}')

    logging.info('Loading data')

    train_gen = load_data(args.train_data, args)
    if args.val_data:
        val_gen = load_data(args.val_data, args)
    else:
        val_gen = None

    if args.debug:
        logging.info('Debug mode active')
        train_gen = [next(iter(train_gen))]
        if val_gen:
           val_gen = [next(iter(val_gen))]

    logging.info('Preparing model')

    shape = list(eval(args.latents_shape))
    model_args = dict(
        shape=shape,
        n_class=512,
        embed_dim=64,
        channel=128,
        kernel_size=5,
        n_block=2,
        n_res_block=2,
        res_channel=128
    )
    if args.model_args:
        model_args.update(dict(eval(args.model_args)))

    model = PixelSNAIL(**model_args).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    if args.sched == 'cycle':
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            mode='triangular2',
            base_lr=args.lr / 100.0,
            max_lr=args.lr,
            gamma=args.gamma,
            step_size_up=10000,
            cycle_momentum=False
        )
    elif args.sched == '1cycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5.0, total_steps=50000, div_factor=100, final_div_factor=100000)
    else:
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=args.gamma
        )

    if amp is not None:
        logging.info('Using AMP for training')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    if args.data_parallel:
        if amp is not None:
            logging.info('Using APEX Distributed Data Parallel training')
            model = apex.parallel.DistributedDataParallel(model)
        else:
            logging.info('Using PyTorch Data Parallel training')
            model = nn.DataParallel(model)

    ae_model = None
    if args.ae_model_checkpoint:
        logging.info(f'Loading autoencoder model checkpoint from {args.ae_model_checkpoint}')
        ae_model = torch.load(args.ae_model_checkpoint).to(device)
        ae_model.eval()

    criterion = nn.CrossEntropyLoss()

    summary_writer = SummaryWriter(save_path)

    try:
        samples = sample(model, ae_model, device, latent_size=shape, num=args.samples_per_epoch, temperature=1.0)
        save_samples(samples, 0, save_path)

        for epoch in range(args.epochs):
            logging.info(f'Epoch {epoch + 1}')

            train(model, optimizer, scheduler, criterion, train_gen, shape, summary_writer, device, epoch)

            if val_gen:
                evaluate(model, criterion, val_gen, shape, summary_writer, device, epoch)

            #if scheduler:
            #    scheduler.step()

            if ae_model:
                logging.info('Sampling')

                samples = sample(model, ae_model, device, latent_size=shape, num=args.samples_per_epoch, temperature=1.0)
                save_samples(samples, epoch + 1, save_path)

                summary_writer.add_text('samples', '\n\n'.join(samples), global_step=epoch)

            if not args.debug:
                save_checkpoint(model, epoch, save_path)
    except KeyboardInterrupt:
        logging.info('Gracefully interrupting training')

    summary_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='pixelsnail', dest='model')
    parser.add_argument('--device', type=str)
    parser.add_argument('--model-args', type=str)
    parser.add_argument('--latents-shape', type=str, required=True)
    parser.add_argument('--ae-model-checkpoint', type=str)
    parser.add_argument('--samples-per-epoch', type=int, default=16)
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--val-data', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data-parallel', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Learning rate decay gamma.')
    parser.add_argument('--sched', choices=['cycle', '1cycle'], default=None)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args)
