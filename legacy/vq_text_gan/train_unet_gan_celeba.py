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
from vq_text_gan.models.char_gan import Generator, UnetDiscriminator
from vq_text_gan.models.gan_base import SelfAttention, DiscriminatorBlock, conv
from vq_text_gan.losses import WGAN_GP, RelativisticAverageHingeLoss

from vq_vae_text.datasets import ByteLevelTextDataset


adversarial_loss = torch.nn.BCELoss()


def loss_d(D, reals, fakes):
    real_global_scores, real_pixel_scores = D(reals)
    real_global_pred = torch.sigmoid(real_global_scores)
    real_pixel_pred = torch.sigmoid(real_pixel_scores)

    fake_global_scores, fake_pixel_scores = D(fakes)
    fake_global_pred = torch.sigmoid(fake_global_scores)
    fake_pixel_pred = torch.sigmoid(fake_pixel_scores)

    valid = torch.empty(reals.size(0), 1).fill_(1.0)
    fake = torch.empty(reals.size(0), 1).fill_(0.0)

    real_loss_g = F.binary_cross_entropy_with_logits(real_global_scores, torch.full_like(real_global_pred, 0.9))
    fake_loss_g = F.binary_cross_entropy_with_logits(fake_global_scores, torch.full_like(fake_global_pred, 0.1))

    real_loss_l = F.binary_cross_entropy_with_logits(real_pixel_scores, torch.full_like(real_pixel_pred, 0.9))
    fake_loss_l = F.binary_cross_entropy_with_logits(fake_pixel_scores, torch.full_like(fake_pixel_pred, 0.1))

    real_loss = real_loss_g + real_loss_l
    fake_loss = fake_loss_g + fake_loss_l

    # real_loss = adversarial_loss(real_global_pred, valid)

    #loss_enc = - torch.log(real_global_pred).mean() - torch.log(1 - fake_global_pred).mean()
    #loss_dec = - torch.log(real_pixel_pred).mean() - torch.log(1 - fake_pixel_pred).mean()

    # real_loss = - torch.log(real_global_pred).mean() - torch.log(real_pixel_pred).mean()
    # fake_loss = - torch.log(1 - fake_global_pred).mean() - torch.log(1 - fake_pixel_pred).mean()

    loss = real_loss + fake_loss
    #
    # rf_diff_g = real_global_scores - torch.mean(fake_global_scores)
    # fr_diff_g = fake_global_scores - torch.mean(real_global_scores)
    #
    # rf_diff_l = real_pixel_scores - torch.mean(fake_pixel_scores)
    # fr_diff_l = fake_pixel_scores - torch.mean(real_pixel_scores)
    #
    # loss_g = F.relu(1 - rf_diff_g).mean() + F.relu(1 + fr_diff_g).mean()
    # loss_l = F.relu(1 - rf_diff_l).mean() + F.relu(1 + fr_diff_l).mean()
    #
    # loss = loss_g + loss_l

    #return loss_enc + loss_dec

    p_real_g = (real_global_pred >= 0.5).type(torch.float).mean()
    p_real_l = (real_pixel_pred >= 0.5).type(torch.float).mean()

    p_fake_g = (fake_global_pred <= 0.5).type(torch.float).mean()
    p_fake_l = (fake_pixel_pred <= 0.5).type(torch.float).mean()

    return loss, p_real_g, p_real_l, p_fake_g, p_fake_l


    loss = fake_global_scores.mean() + fake_pixel_scores.mean() \
           - real_global_scores.mean() - real_pixel_scores.mean()

    return loss


def loss_g(D, reals, fakes):
    real_global_scores, real_pixel_scores = D(reals)
    fake_global_scores, fake_pixel_scores = D(fakes)

    # rf_diff_g = real_global_scores - torch.mean(fake_global_scores)
    # fr_diff_g = fake_global_scores - torch.mean(real_global_scores)
    #
    # rf_diff_l = real_pixel_scores - torch.mean(fake_pixel_scores)
    # fr_diff_l = fake_pixel_scores - torch.mean(real_pixel_scores)
    #
    # loss_g = F.relu(1 + rf_diff_g).mean() + F.relu(1 - fr_diff_g).mean()
    # loss_l = F.relu(1 + rf_diff_l).mean() + F.relu(1 - fr_diff_l).mean()
    #
    # return loss_g + loss_l

    fake_global_pred = torch.sigmoid(fake_global_scores)
    fake_pixel_pred = torch.sigmoid(fake_pixel_scores)

    fake_loss_g = adversarial_loss(fake_global_pred, torch.full_like(fake_global_pred, 1.0))
    fake_loss_l = adversarial_loss(fake_pixel_pred, torch.full_like(fake_pixel_pred, 1.0))

    return fake_loss_l + fake_loss_g

    return - (torch.log(fake_global_pred) + torch.log(fake_pixel_pred).mean(1)).mean()

    return - fake_global_scores.mean() - fake_pixel_scores.mean()


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight, 0, 0.02)


def apply_spectral_norm(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.utils.spectral_norm(m)


def main(args):
    result_dir = setup_run(args.run_name, create_dirs=['checkpoints', 'samples'])
    setup_logging(result_dir / 'log.txt')

    logging.info(args)

    device = get_default_device(args.device)

    sample_dir = result_dir / 'samples'
    checkpoint_dir = result_dir / 'checkpoints'

    seq_length = 32

    dataset = ByteLevelTextDataset(args.dataset, seq_length)

    depth = math.log2(seq_length)

    assert int(depth) == depth

    depth = int(depth)

    vocab_size = dataset.vocab_size

    batches = DataLoader(dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    inter_dim = 8

    embedding = nn.Embedding(vocab_size, inter_dim, max_norm=1.0).to(device)
    embedding.weight.requires_grad = False

    G = Generator(args.latent_size, [256, 128, 64, 32], out_dim=inter_dim).to(device)
    D = UnetDiscriminator(32, depth=4, in_dim=inter_dim).to(device)

    # G.apply(apply_spectral_norm)
    # D.apply(apply_spectral_norm)

    G.apply(init_weights)
    D.apply(init_weights)

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

    G_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.0, 0.999))
    D_opt = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.0, 0.999))

    z_sample = torch.randn(args.n_sample, args.latent_size, 1).to(device)

    #loss_f = RelativisticAverageHingeLoss(D)
    loss_f = WGAN_GP(D)

    def decode(embeds):
        flatten = embeds.transpose(1, 2)
        flatten = flatten.reshape(-1, flatten.size(-1))

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embedding.weight.T
            + embedding.weight.T.pow(2).sum(0, keepdim=True)
        )

        _, samples = (-dist).max(1)
        return samples.view(samples_embeds.size(0), -1)

    try:
        global_step = 0
        for epoch in range(args.epochs):
            g_loss_sum = 0
            d_loss_sum = 0

            p_fake_g_sum = 0
            p_fake_l_sum = 0

            p_real_g_sum = 0
            p_real_l_sum = 0

            start_time = time.time()

            cur_step = 0

            for step, reals in enumerate(batches):
                reals = reals.to(device)
                reals_embed = embedding(reals).transpose(1, 2)
                reals_embed += torch.randn_like(reals_embed) * 0.01

                batch_size = reals.size(0)

                z = torch.randn(batch_size, args.latent_size, 1).to(device)

                # Optimize the discriminator
                fake_out = G(z)

                D_opt.zero_grad()

                d_loss, p_real_g, p_real_l, p_fake_g, p_fake_l = loss_d(D, reals_embed, fake_out.detach())
                d_loss.backward()

                D_opt.step()

                # Optimize generator
                fake_out = G(z)

                G_opt.zero_grad()

                g_loss = loss_g(D, reals_embed, fake_out)
                g_loss.backward()

                G_opt.step()

                if args.use_ema:
                    update_average(G_shadow, G_orig, beta=0.999)

                g_loss_sum += float(g_loss)
                d_loss_sum += float(d_loss)

                p_fake_g_sum += float(p_fake_g)
                p_fake_l_sum += float(p_fake_l)

                p_real_g_sum += float(p_real_g)
                p_real_l_sum += float(p_real_l)

                if global_step % args.log_every == 0:
                    cur_step = min(step + 1, args.log_every)
                    batches_per_sec = cur_step / (time.time() - start_time)

                    logging.info(f'[EPOCH {epoch + 1:03d}] [{step:05d} / {len(batches):05d}] ' +
                                 #f'grow_index: {current_grow_index}/{depth - 1}, ' +
                                 f'loss_d: {d_loss_sum / cur_step:.5f}, loss_g: {g_loss_sum / cur_step:.5f}, ' +
                                 f'p_fake_g: {p_fake_g_sum / cur_step:.5f}, p_fake_l: {p_fake_l_sum / cur_step:.5f}, ' +
                                 f'p_real_g: {p_real_g_sum / cur_step:.5f}, p_real_l: {p_real_l_sum / cur_step:.5f}, ' +
                                 f'batches/s: {batches_per_sec:02.2f}')

                    g_loss_sum = d_loss_sum = 0
                    p_fake_sum = p_real_sum = 0

                    p_fake_g_sum = 0
                    p_fake_l_sum = 0

                    p_real_g_sum = 0
                    p_real_l_sum = 0

                    start_time = time.time()

                if global_step % args.sample_every == 0:
                    samples_embeds = G_sample(z_sample)
                    samples = decode(samples_embeds)

                    reals_decode = decode(reals_embed)

                    (sample_dir / f'fakes_{global_step:06d}.txt').write_text('\n'.join(dataset.seq_to_text(samples)))
                    (sample_dir / f'reals_{global_step:06d}.txt').write_text('\n'.join(dataset.seq_to_text(reals_decode)))

                cur_step += 1
                global_step += 1

            torch.save(G, str(checkpoint_dir / f'G_{global_step:06d}.pth'))
            torch.save(D, str(checkpoint_dir / f'D_{global_step:06d}.pth'))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='char-gan')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--device', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--g-lr', type=float, default=1e-5)
    parser.add_argument('--d-lr', type=float, default=4e-5)
    parser.add_argument('--latent-size', type=int, default=128)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--sample-every', type=int, default=250)
    parser.add_argument('--n-sample', type=int, default=32)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--attn', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--data-parallel', action='store_true')
    args = parser.parse_args()

    main(args)
