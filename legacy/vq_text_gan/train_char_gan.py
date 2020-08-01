import argparse
import logging
import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from operator import attrgetter
from pathlib import Path
from torch.utils.data import DataLoader

from vq_text_gan.utils import update_average, get_default_device, setup_logging, setup_run, repr_list
from vq_text_gan.datasets import LMDBDataset, BPEDataset
from vq_text_gan.models.char_gan import Generator, UnetDiscriminator
from vq_text_gan.models.gan_base import SelfAttention, DiscriminatorBlock, conv
from vq_text_gan.losses import WGAN_GP, RelativisticAverageHingeLoss

from vq_vae_text.datasets import ByteLevelTextDataset


adversarial_loss = torch.nn.BCELoss()


class GANLoss:
    def __init__(self, D):
        self.D = D

    def loss_d(self, reals, fakes):
        real_global_scores, real_pixel_scores = self.D(reals)
        fake_global_scores, fake_pixel_scores = self.D(fakes)

        real_loss_g = F.binary_cross_entropy_with_logits(real_global_scores, torch.full_like(real_global_scores, 0.9))
        fake_loss_g = F.binary_cross_entropy_with_logits(fake_global_scores, torch.full_like(fake_global_scores, 0.1))

        real_loss_l = F.binary_cross_entropy_with_logits(real_pixel_scores, torch.full_like(real_pixel_scores, 0.9))
        fake_loss_l = F.binary_cross_entropy_with_logits(fake_pixel_scores, torch.full_like(fake_pixel_scores, 0.1))

        real_loss = real_loss_g + real_loss_l
        fake_loss = fake_loss_g + fake_loss_l

        loss = (real_loss + fake_loss) / 2

        p_real_g = (real_global_scores >= 0.0).type(torch.float).mean()
        p_real_l = (real_pixel_scores >= 0.0).type(torch.float).mean()

        p_fake_g = (fake_global_scores <= 0.0).type(torch.float).mean()
        p_fake_l = (fake_pixel_scores <= 0.0).type(torch.float).mean()

        return loss, p_real_g, p_real_l, p_fake_g, p_fake_l

    def loss_g(self, reals, fakes):
        fake_global_scores, fake_pixel_scores = self.D(fakes)

        fake_loss_g = F.binary_cross_entropy_with_logits(fake_global_scores, torch.full_like(fake_global_scores, 1.0))
        fake_loss_l = F.binary_cross_entropy_with_logits(fake_pixel_scores, torch.full_like(fake_pixel_scores, 1.0))

        return fake_loss_l + fake_loss_g


class RelativisticAverageHingeLoss:
    def __init__(self, D):
        self.D = D

    def loss_d(self, reals, fakes):
        real_global_scores, real_pixel_scores = self.D(reals)
        fake_global_scores, fake_pixel_scores = self.D(fakes)

        rf_diff_g = real_global_scores - torch.mean(fake_global_scores)
        fr_diff_g = fake_global_scores - torch.mean(real_global_scores)

        rf_diff_l = real_pixel_scores - torch.mean(fake_pixel_scores)
        fr_diff_l = fake_pixel_scores - torch.mean(real_pixel_scores)

        loss_g = F.relu(1 - rf_diff_g).mean() + F.relu(1 + fr_diff_g).mean()
        loss_l = F.relu(1 - rf_diff_l).mean() + F.relu(1 + fr_diff_l).mean()

        loss = loss_g + loss_l

        p_real_g = (real_global_scores >= 0.5).type(torch.float).mean()
        p_real_l = (real_pixel_scores >= 0.5).type(torch.float).mean()

        p_fake_g = (fake_global_scores <= 0.5).type(torch.float).mean()
        p_fake_l = (fake_pixel_scores <= 0.5).type(torch.float).mean()

        return loss, p_real_g, p_real_l, p_fake_g, p_fake_l

    def loss_g(self, reals, fakes):
        real_global_scores, real_pixel_scores = self.D(reals)
        fake_global_scores, fake_pixel_scores = self.D(fakes)

        rf_diff_g = real_global_scores - torch.mean(fake_global_scores)
        fr_diff_g = fake_global_scores - torch.mean(real_global_scores)

        rf_diff_l = real_pixel_scores - torch.mean(fake_pixel_scores)
        fr_diff_l = fake_pixel_scores - torch.mean(real_pixel_scores)

        loss_g = F.relu(1 + rf_diff_g).mean() + F.relu(1 - fr_diff_g).mean()
        loss_l = F.relu(1 + rf_diff_l).mean() + F.relu(1 - fr_diff_l).mean()

        return loss_g + loss_l


class WGAN_GP:
    def __init__(self, D, drift=0.001, use_gp=True, reg_lambda=10):
        self.D = D
        self.drift = drift
        self.use_gp = use_gp
        self.reg_lambda = reg_lambda

    def gradient_penalty(self, reals, fakes):
        batch_size = reals.size(0)

        # generate random epsilon
        eps = torch.rand((batch_size, 1, 1)).to(reals.device)

        # create the merge of both real and fake samples
        merged = eps * reals + (1 - eps) * fakes
        merged.requires_grad = True

        # forward pass
        global_score, local_scores = self.D(merged)
        local_score = local_scores.mean()

        # perform backward pass from op to merged for obtaining the gradients
        gradients = torch.autograd.grad(outputs=[global_score, local_score], inputs=merged,
                                        grad_outputs=[torch.ones_like(global_score), torch.ones_like(local_score)],
                                        create_graph=True, retain_graph=True, only_inputs=True)

        # calculate the penalty using these gradients
        penalties = [(gradient.reshape(gradient.size(0), -1).norm(p=2, dim=1) ** 2).mean().unsqueeze(0)
                     for gradient in gradients]
        penalty = self.reg_lambda * torch.cat(penalties).mean()

        # return the calculated penalty:
        return penalty

    def loss_d(self, reals, fakes):
        real_global_scores, real_pixel_scores = self.D(reals)
        fake_global_scores, fake_pixel_scores = self.D(fakes)

        loss_global = fake_global_scores.mean() - real_global_scores.mean() + self.drift * (real_global_scores ** 2).mean()
        loss_local = fake_pixel_scores.mean() - real_pixel_scores.mean() + self.drift * (real_pixel_scores ** 2).mean()

        loss = loss_global + loss_local

        if self.use_gp and self.reg_lambda:
            loss += self.gradient_penalty(reals, fakes)

        p_real_g = (real_global_scores >= 0.0).type(torch.float).mean()
        p_real_l = (real_pixel_scores >= 0.0).type(torch.float).mean()

        p_fake_g = (fake_global_scores <= 0.0).type(torch.float).mean()
        p_fake_l = (fake_pixel_scores <= 0.0).type(torch.float).mean()

        return loss, p_real_g, p_real_l, p_fake_g, p_fake_l

    def loss_g(self, reals, fakes):
        fake_global_scores, fake_pixel_scores = self.D(fakes)

        return -fake_global_scores.mean() - fake_pixel_scores.mean()



def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.orthogonal_(m.weight)


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

    dataset = ByteLevelTextDataset(args.dataset, seq_len=seq_length)

    vocab_size = dataset.vocab_size

    batches = DataLoader(dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    inter_dim = vocab_size

    # spiegel model
    # G = Generator(args.latent_size, [256, 256, 128, 64, 64], out_dim=inter_dim).to(device)
    # D = UnetDiscriminator(64, max_channel=256, depth=5, in_dim=inter_dim).to(device)

    G = Generator(args.latent_size, [512, 512, 512, 512], out_dim=vocab_size).to(device)
    D = UnetDiscriminator(512, max_channel=512, depth=4, in_dim=vocab_size).to(device)

    G.apply(apply_spectral_norm)
    D.apply(apply_spectral_norm)

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
    D_orig = D

    if args.data_parallel:
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    G_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    D_opt = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    z_sample = torch.randn(args.n_sample, args.latent_size, 1).to(device)

    #loss_f = RelativisticAverageHingeLoss(D)
    #loss_f = GANLoss(D)
    loss_f = WGAN_GP(D)

    try:
        global_step = 0
        for epoch in range(args.epochs):
            g_loss_sum = 0
            d_loss_sum = 0

            p_fake_g_sum = 0
            p_fake_l_sum = 0

            p_real_g_sum = 0
            p_real_l_sum = 0

            G_attn_sum = 0
            D_attn_sum = 0

            start_time = time.time()

            cur_step = 0

            for step, reals in enumerate(batches):
                reals = reals.to(device)
                reals_one_hot = F.one_hot(reals, num_classes=vocab_size).transpose(1, 2).to(torch.float)

                batch_size = reals.size(0)

                z = torch.randn(batch_size, args.latent_size, 1).to(device)

                tau = 0.1

                # Optimize the discriminator
                fake_out = G(z)
                fake_out = torch.softmax(fake_out / tau, dim=1)

                D_opt.zero_grad()

                d_loss, p_real_g, p_real_l, p_fake_g, p_fake_l = loss_f.loss_d(reals_one_hot, fake_out.detach())
                d_loss.backward()

                D_opt.step()

                # Optimize generator
                fake_out = G(z)
                fake_out = torch.softmax(fake_out / tau, dim=1)

                G_opt.zero_grad()

                g_loss = loss_f.loss_g(reals_one_hot, fake_out)
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

                #G_attn_sum += float(G_orig.attn.gamma)
                #D_attn_sum += float(D_orig.attn.gamma)

                if global_step % args.log_every == 0:
                    cur_step = min(step + 1, args.log_every)
                    batches_per_sec = cur_step / (time.time() - start_time)

                    logging.info(f'[EPOCH {epoch + 1:03d}] [{step:05d} / {len(batches):05d}] ' +
                                 #f'grow_index: {current_grow_index}/{depth - 1}, ' +
                                 f'loss_d: {d_loss_sum / cur_step:.5f}, loss_g: {g_loss_sum / cur_step:.5f}, ' +
                                 f'p_fake_g: {p_fake_g_sum / cur_step:.5f}, p_fake_l: {p_fake_l_sum / cur_step:.5f}, ' +
                                 f'p_real_g: {p_real_g_sum / cur_step:.5f}, p_real_l: {p_real_l_sum / cur_step:.5f}, ' +
                                 #f'G_attn_gamma: {G_attn_sum / cur_step:.2f}, D_attn_gamma: {D_attn_sum / cur_step:.2f}, '
                                 f'batches/s: {batches_per_sec:02.2f}')

                    g_loss_sum = d_loss_sum = 0

                    p_fake_g_sum = 0
                    p_fake_l_sum = 0

                    p_real_g_sum = 0
                    p_real_l_sum = 0

                    G_attn_sum = 0
                    D_attn_sum = 0

                    start_time = time.time()

                if global_step % args.sample_every == 0:
                    samples = G_sample(z_sample).argmax(1)

                    (sample_dir / f'fakes_{global_step:06d}.txt').write_text('\n'.join(dataset.seq_to_text(samples)))
                    (sample_dir / f'reals_{global_step:06d}.txt').write_text('\n'.join(dataset.seq_to_text(reals[:args.n_sample])))

                    # (sample_dir / f'fakes_{global_step:06d}.txt').write_text('\n'.join(dataset.seq_to_text(samples)))
                    # (sample_dir / f'reals_{global_step:06d}.txt').write_text('\n'.join(dataset.seq_to_text(reals_decode)))

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
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--g-lr', type=float, default=5e-5)
    parser.add_argument('--d-lr', type=float, default=2e-4)
    parser.add_argument('--latent-size', type=int, default=256)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--sample-every', type=int, default=250)
    parser.add_argument('--n-sample', type=int, default=32)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--attn', action='store_true')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--data-parallel', action='store_true')
    args = parser.parse_args()

    main(args)
