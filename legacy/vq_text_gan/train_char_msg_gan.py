import argparse
import logging
import copy
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from operator import attrgetter
from torch.utils.data import DataLoader

from vq_text_gan.utils import update_average, get_default_device, setup_logging, setup_run, repr_list
from vq_text_gan.datasets import LMDBDataset, BPEDataset
from vq_text_gan.models.msg_gan import Generator, Discriminator
from vq_text_gan.models.gan_base import SelfAttention, DiscriminatorBlock, conv
from vq_text_gan.losses import WGAN_GP, RelativisticAverageHingeLoss


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight, 0.0, 0.02)


def apply_spectral_norm(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.utils.spectral_norm(m)


class TransformNetwork(nn.Module):
    def __init__(self, num_input_channels, block_channels, extract_dim):
        super().__init__()

        channels = [num_input_channels] + block_channels

        self.blocks = nn.ModuleList()
        self.extract_layers = nn.ModuleList()

        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            self.blocks.append(DiscriminatorBlock(in_channels, out_channels))
            self.extract_layers.append(conv(channels[i + 1], extract_dim, kernel_size=1))

    def forward(self, input):
        x = input
        outputs = []
        for block, extract_layer in zip(self.blocks, self.extract_layers):
            x = block(x)
            if extract_layer is not None:
                outputs.append(extract_layer(x))
        return outputs


def main(args):
    result_dir = setup_run(args.run_name, create_dirs=['checkpoints', 'samples'])
    setup_logging(result_dir / 'log.txt')

    logging.info(args)

    device = get_default_device(args.device)

    sample_dir = result_dir / 'samples'
    checkpoint_dir = result_dir / 'checkpoints'

    dataset = BPEDataset(args.original_dataset)

    depth = math.log2(dataset.seq_length)

    assert int(depth) == depth

    depth = int(depth)

    vocab_size = dataset.vocab_size

    batches = DataLoader(dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    inter_dim = 32

    extract_dims = [inter_dim] * 4 + [vocab_size]
    inject_dims = [vocab_size] + [inter_dim] * 4

    G = Generator(args.latent_size, [128, 128, 128, 128, 128], extract_dims, attn=args.attn).to(device)
    D = Discriminator([128, 128, 128, 128, 128], inject_dims, attn=args.attn).to(device)

    T = TransformNetwork(vocab_size, [64, 64, 64, 64], inter_dim).to(device)

    if args.attn:
        D_gammas = list(map(attrgetter('gamma'), filter(lambda m: isinstance(m, SelfAttention), D.modules())))
        G_gammas = list(map(attrgetter('gamma'), filter(lambda m: isinstance(m, SelfAttention), G.modules())))

    #G.apply(init_weights)
    #G.apply(apply_spectral_norm)

    #D.apply(init_weights)
    #D.apply(apply_spectral_norm)

    G.train()
    D.train()

    (result_dir / 'G.txt').write_text(str(G))
    (result_dir / 'D.txt').write_text(str(D))

    if args.use_ema:
        G_shadow = copy.deepcopy(G)
        G_sample = G_shadow
        update_average(G_shadow, G, beta=0.0)
    else:
        G_sample = G

    G_orig = G

    if args.data_parallel:
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    G_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.9))
    D_opt = torch.optim.Adam(list(D.parameters()) + list(T.parameters()), lr=args.d_lr, betas=(0.5, 0.9))

    z_sample = torch.randn(args.n_sample, args.latent_size, 1).to(device)

    #loss_f = RelativisticAverageHingeLoss(D)
    loss_f = WGAN_GP(D)

    try:
        global_step = 0
        for epoch in range(args.epochs):
            g_loss_sum = 0
            d_loss_sum = 0

            p_fake_sum = 0
            p_real_sum = 0

            D_gammas_sum = OrderedDict()
            G_gammas_sum = OrderedDict()

            start_time = time.time()

            cur_step = 0

            for step, reals in enumerate(batches):
                reals = reals.to(device)
                reals_one_hot = F.one_hot(reals, num_classes=vocab_size).transpose(1, 2).type(torch.float)
                batch_size = reals.size(0)

                reals_t = T(reals_one_hot)
                reals_input = [reals_one_hot] + reals_t

                z = torch.randn(batch_size, args.latent_size, 1).to(device)

                # Optimize the discriminator
                fake_out = G(z)
                fake_probs = torch.softmax(fake_out[-1], dim=1)
                fake_input = (fake_out[:-1] + [fake_probs])[::-1]
                fake_input = [t.detach() for t in fake_input]

                D_opt.zero_grad()

                loss_d, p_fake, p_real = loss_f.d_loss(reals_input, fake_input)
                loss_d.backward()

                D_opt.step()

                # Optimize generator
                fake_out = G(z)
                fake_probs = torch.softmax(fake_out[-1], dim=1)
                fake_input = (fake_out[:-1] + [fake_probs])[::-1]

                G_opt.zero_grad()

                reals_input = [t.detach() for t in reals_input]

                loss_g = loss_f.g_loss(reals_input, fake_input)
                loss_g.backward()

                G_opt.step()

                if args.use_ema:
                    update_average(G_shadow, G_orig, beta=0.999)

                g_loss_sum += float(loss_g)
                d_loss_sum += float(loss_d)

                p_fake_sum += float(p_fake)
                p_real_sum += float(p_real)

                if args.attn:
                    for i, (d_gamma, g_gamma) in enumerate(zip(D_gammas, G_gammas)):
                        D_gammas_sum[i] = D_gammas_sum.get(i, 0) + d_gamma
                        G_gammas_sum[i] = G_gammas_sum.get(i, 0) + g_gamma

                if global_step % args.log_every == 0:
                    cur_step = min(step + 1, args.log_every)
                    batches_per_sec = cur_step / (time.time() - start_time)

                    if args.attn:
                        D_gammas_avg = repr_list([gamma / cur_step for gamma in D_gammas_sum.values()])
                        G_gammas_avg = repr_list([gamma / cur_step for gamma in G_gammas_sum.values()])

                    logging.info(f'[EPOCH {epoch + 1:03d}] [{step:05d} / {len(batches):05d}] ' +
                                 #f'grow_index: {current_grow_index}/{depth - 1}, ' +
                                 f'loss_d: {d_loss_sum / cur_step:.5f}, loss_g: {g_loss_sum / cur_step:.5f}, ' +
                                 f'p_fake: {p_fake_sum / cur_step:.5f}, p_real: {p_real_sum / cur_step:.5f}, ' +
                                 (f'd_attn_gammas: [{D_gammas_avg}], g_attn_gammas: [{G_gammas_avg}], ' if args.attn else '') +
                                 f'batches/s: {batches_per_sec:02.2f}')

                    g_loss_sum = d_loss_sum = 0
                    p_fake_sum = p_real_sum = 0

                    D_gammas_sum = OrderedDict()
                    G_gammas_sum = OrderedDict()

                    start_time = time.time()

                if global_step % args.sample_every == 0:

                    samples_decoded = dataset.seq_to_text(G_sample(z_sample)[-1].argmax(1))
                    reals_decoded = dataset.seq_to_text(reals[:args.n_sample])

                    (sample_dir / f'fakes_{global_step:06d}.txt').write_text('\n'.join(samples_decoded))
                    (sample_dir / f'reals_{global_step:06d}.txt').write_text('\n'.join(reals_decoded))

                cur_step += 1
                global_step += 1

            torch.save(G, str(checkpoint_dir / f'G_{global_step:06d}.pth'))
            torch.save(D, str(checkpoint_dir / f'D_{global_step:06d}.pth'))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='char-msg-gan')
    parser.add_argument('--device', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--g-lr', type=float, default=0.0001)
    parser.add_argument('--d-lr', type=float, default=0.0001)
    parser.add_argument('--latent-size', type=int, default=128)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--sample-every', type=int, default=250)
    parser.add_argument('--n-sample', type=int, default=32)
    parser.add_argument('--n-print-samples', type=int, default=8)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--steps-per-stage', default=75_000)
    parser.add_argument('--original-dataset', required=True)
    parser.add_argument('--n-mc-samples', type=int, default=8)
    parser.add_argument('--mc-sample-tau', type=float, default=1.0)
    parser.add_argument('--vq-dataset', required=True)
    parser.add_argument('--vq-model', required=True)
    parser.add_argument('--attn', action='store_true')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--data-parallel', action='store_true')
    args = parser.parse_args()

    main(args)
