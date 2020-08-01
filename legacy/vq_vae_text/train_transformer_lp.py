import argparse
import logging
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from itertools import islice

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vq_vae_text.utils import setup_logging_from_args, save_checkpoint
from vq_vae_text.datasets import LMDBDataset, ByteLevelTextDataset
from vq_vae_text.models.transformer import TransformerEncoderModel
# try:
#     import apex
#     from apex import amp
# except ImportError:
#     amp = None

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

        #if scheduler is not None:
        #    scheduler.step()

        accuracy = compute_accuracy(batch, out.argmax(1))

        lr = optimizer.param_groups[0]['lr']

        pbar.set_description(f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                             f'acc: {accuracy:.5f}; lr: {lr:.5f}')

        step = optimizer.state[optimizer.param_groups[0]['params'][0]]['step']
        log_summary(summary_writer, 'train_', zip(['loss', 'acc', 'lr'], [loss, accuracy, lr]), step=step)


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
    latents = latents.view(latents.size(0), -1, latents.size(-1))

    pred = ae_model.decode_codes(latents).cpu()
    pred = pred.argmax(-1)
    texts = ByteLevelTextDataset.seq_to_text(pred)

    return texts


def main(args):
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    logging.info(f'Loading autoencoder model checkpoint from {args.ae_model_checkpoint}')
    ae_model = torch.load(args.ae_model_checkpoint).to('cpu')
    ae_model.eval()

    shape = list(eval(args.latents_shape))
    model_args = dict(
        vocab_size=ae_model.num_vq_embeds,
        embed_dim=64,
        num_heads=1,
        hidden_dim=512,
        num_layers=1,
    )
    if args.model_args:
        model_args.update(dict(eval(args.model_args)))

    model = TransformerEncoderModel(**model_args).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    if args.sched:
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            mode='triangular2',
            base_lr=args.lr / 100.0,
            max_lr=args.lr,
            gamma=args.gamma,
            step_size_up=10000,
            cycle_momentum=False
        )
    else:
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=args.gamma
        )

    criterion = nn.CrossEntropyLoss()

    summary_writer = SummaryWriter(save_path)

    try:
        for epoch in range(args.epochs):
            logging.info(f'Epoch {epoch + 1}')

            train(model, optimizer, scheduler, criterion, train_gen, shape, summary_writer, device, epoch)

            if val_gen:
                evaluate(model, criterion, val_gen, shape, summary_writer, device, epoch)

            if scheduler:
                scheduler.step()

            # if ae_model:
            #     logging.info('Sampling')
            #
            #     samples = sample(model, ae_model, device, latent_size=shape, num=args.samples_per_epoch, temperature=1.0)
            #     desc = '\n\n'.join(samples)
            #     logging.info(desc)
            #
            #     summary_writer.add_text('samples', desc, global_step=epoch)

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
    parser.add_argument('--sched', choices=['cycle'], default=None)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--stage', choices=['bottom', 'top'])
    args = parser.parse_args()

    main(args)
