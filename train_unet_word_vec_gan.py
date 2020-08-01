import argparse
import logging
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from bpemb import BPEmb
from torch.utils.data import DataLoader, IterableDataset

from text_gans.model import TextGANModel
from text_gans.sampling import get_sampling_model
from text_gans.utils import get_default_device, setup_logging, setup_run, init_weights, get_log2_seq_length
from text_gans.nets.gan import Generator, UNetDiscriminator
from text_gans.losses import name_to_loss_class, get_loss_function_by_name


def prepare_data(dataset_path, seq_length, bpe, max_samples=1_000_000):
    lines = Path(dataset_path).read_text().split('\n')[:max_samples]

    data = torch.full((len(lines), seq_length), bpe.vocab_size, dtype=torch.long)

    for i, encoded_sample in enumerate(bpe.encode_ids_with_bos_eos(lines)):
        l = min(seq_length, len(encoded_sample))
        data[i, :l] = torch.tensor(encoded_sample, dtype=torch.long)[:l]

    return data


def build_log_string(args):
    log_string_base = '[EPOCH {epoch:03d}] [{step:05d} / {num_batches:05d}]'
    log_string_parts = OrderedDict(loss_d='{d_loss:.5f}', loss_g='{g_loss:.5f}')
    if args.attn:
        log_string_parts.update(G_attn_gamma='{G_attn:.2f}', D_attn_gamma='{D_attn:.2f}')
    log_string_parts['batches/s'] = '{batches_per_sec:02.2f}'

    log_string = ', '.join(map(': '.join, log_string_parts.items()))

    return ' '.join([log_string_base, log_string])


class EmbeddedDataset(IterableDataset):
    def __init__(self, data, embedding):
        self.data = data
        self.embedding = embedding

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(map(lambda t: self.embedding(t).T, self.data))


def main(args):
    result_dir = setup_run(args.run_name, create_dirs=['checkpoints', 'samples'])
    setup_logging(result_dir / 'log.txt')

    logging.info(args)

    device = get_default_device(args.device)

    log2_seq_length = get_log2_seq_length(args.seq_length)

    bpe = BPEmb(lang='de', vs=args.vocab_size, dim=args.embedding_dim, add_pad_emb=True)
    data = prepare_data(args.dataset, seq_length=64, bpe=bpe)

    vocab_size = bpe.vocab_size + 1

    inter_dim = bpe.dim

    embedding = nn.Embedding(vocab_size, inter_dim, _weight=torch.tensor(bpe.vectors, dtype=torch.float))
    embedding.weight.requires_grad = False

    embeded_data = EmbeddedDataset(data, embedding)

    batches = DataLoader(embeded_data, args.batch_size, pin_memory=True, num_workers=args.num_workers)

    attn_at = None
    if args.attn:
        attn_at = 2

    depth = log2_seq_length - 1

    # Spiegel model
    G = Generator(args.latent_size, [args.max_channels] * depth, out_dim=inter_dim, attn_at=attn_at).to(device)
    D = UNetDiscriminator(16, max_channel=args.max_channels, depth=depth, in_dim=inter_dim, attn_at=attn_at).to(device)

    G.apply(init_weights)
    D.apply(init_weights)

    G.train()
    D.train()

    (result_dir / 'G.txt').write_text(str(G))
    (result_dir / 'D.txt').write_text(str(D))

    sampling_model = get_sampling_model(G, args.use_ema)

    G_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.0, 0.99))
    D_opt = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.0, 0.99))

    z_sample = torch.randn(args.n_sample, args.latent_size, 1).to(device)

    loss_f = get_loss_function_by_name(D, args.loss)

    def decode(embeds):
        flatten = embeds.transpose(1, 2).cpu()
        flatten = flatten.reshape(-1, flatten.size(-1))

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embedding.weight.T
            + embedding.weight.T.pow(2).sum(0, keepdim=True)
        )

        _, ids = (-dist).max(1)
        ids = ids.view(embeds.size(0), -1)

        decoded = []
        for seq in ids:
            seq = list(seq.detach().cpu().numpy())
            seq = list(filter(lambda x: x != vocab_size - 1, seq))
            dec = bpe.decode_ids(np.array(seq))
            decoded.append(dec or '')

        return decoded

    model = TextGANModel(
        G, D, G_opt, D_opt, loss_f, batches, args.latent_size, device, result_dir, sampling_model, decode, z_sample
    )
    model.train(args.epochs, args.log_every, args.sample_every)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='unet-word-vec-gan')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--device', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--loss', choices=list(name_to_loss_class.keys()), default='wgan_gp')
    parser.add_argument('--seq-length', type=int, default=64)
    parser.add_argument('--max-channels', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--g-lr', type=float, default=5e-5)
    parser.add_argument('--d-lr', type=float, default=1e-4)
    parser.add_argument('--vocab-size', type=int, default=1_000)
    parser.add_argument('--embedding-dim', type=int, default=25)
    parser.add_argument('--latent-size', type=int, default=256)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--sample-every', type=int, default=1_000)
    parser.add_argument('--n-sample', type=int, default=32)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--attn', action='store_true')
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    main(args)
