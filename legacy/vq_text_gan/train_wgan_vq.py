import argparse
import logging
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from operator import attrgetter
from torch.utils.data import DataLoader

from vq_text_gan.utils import update_average, get_default_device, setup_logging, setup_run, repr_list
from vq_text_gan.datasets import LMDBDataset, BPEDataset
from vq_text_gan.models.progan import Generator, Discriminator
from vq_text_gan.models.gan_base import SelfAttention
from vq_text_gan.losses import WGAN_GP, RelativisticAverageHingeLoss
from vq_text_gan.modules import CategoricalNoise, Quantize

def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight, 0.0, 0.02)


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

    decode = BPEDataset(args.original_dataset).seq_to_text

    vq_model = torch.load(args.vq_model).to(device)

    quantizers = vq_model.quantize
    for q in quantizers:
        q.eval()

    depth = vq_model.depth
    num_classes = vq_model.quantize[0].n_embed
    quant_dim = vq_model.quantize[0].dim

    dataset = LMDBDataset(args.vq_dataset)
    batches = DataLoader(dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    G = Generator(args.latent_size, [256, 256, 256, 256], quant_dim, attn=args.attn).to(device)
    D = Discriminator([256, 256, 256, 256], quant_dim, attn=args.attn).to(device)

    if args.attn:
        D_gammas = list(map(attrgetter('gamma'), filter(lambda m: isinstance(m, SelfAttention), D.modules())))
        G_gammas = list(map(attrgetter('gamma'), filter(lambda m: isinstance(m, SelfAttention), G.modules())))

    G.apply(init_weights)
    #G.apply(apply_spectral_norm)

    D.apply(init_weights)
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

    G_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    D_opt = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    z_sample = torch.randn(args.n_sample, args.latent_size, 1).to(device)

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

            for step, codes in enumerate(batches):
                current_grow_index = min(global_step // args.steps_per_stage, depth - 1)
                interpol = (global_step % args.steps_per_stage) / args.steps_per_stage

                quantize = quantizers[current_grow_index]

                code = codes[-(current_grow_index + 1)]
                code = code.to(device)
                code_embed = quantize.embed_code(code).transpose(1, 2)

                z = torch.randn(code.size(0), args.latent_size, 1).to(device)

                # Optimize the discriminator
                fake_out = G(z, extract_at_grow_index=current_grow_index)
                #fake_embed, fake_diff, fake_code = quantize(fake_out.transpose(1, 2)).transpose(1, 2)

                D_opt.zero_grad()

                loss_d, p_fake, p_real = loss_f.d_loss(code_embed, fake_out.detach())
                loss_d.backward()

                D_opt.step()

                # Optimize generator
                fake_out = G(z, extract_at_grow_index=current_grow_index)
                fake_embed, fake_diff, fake_code = quantize(fake_out.transpose(1, 2))

                G_opt.zero_grad()

                loss_g = loss_f.g_loss(code_embed, fake_out)
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
                                 f'grow_index: {current_grow_index}/{depth - 1}, ' +
                                 f'loss_d: {d_loss_sum / cur_step:.5f}, loss_g: {g_loss_sum / cur_step:.5f}, ' +
                                 f'p_fake: {p_fake_sum / cur_step:.5f}, p_real: {p_real_sum / cur_step:.5f}, ' +
                                 f'vq_diff: {fake_diff:.2f}, ' +
                                 (f'd_attn_gammas: [{D_gammas_avg}], g_attn_gammas: [{G_gammas_avg}], ' if args.attn else '') +
                                 f'batches/s: {batches_per_sec:02.2f}')

                    g_loss_sum = d_loss_sum = 0
                    p_fake_sum = p_real_sum = 0

                    D_gammas_sum = OrderedDict()
                    G_gammas_sum = OrderedDict()

                    start_time = time.time()

                if global_step % args.sample_every == 0:
                    current_depth = depth - 1 - current_grow_index

                    samples_out = G_sample(z_sample, extract_at_grow_index=current_grow_index).argmax(1)
                    _, _, samples_codes = quantize(fake_out.transpose(1, 2))
                    samples_logits = vq_model.decode_code(samples_codes, current_depth)
                    samples_decoded = decode(samples_logits.argmax(-1))

                    reals_logits = vq_model.decode_code(code[:args.n_sample], current_depth)
                    reals_decoded = decode(reals_logits.argmax(-1))

                    (sample_dir / f'fakes_{current_grow_index}_{global_step:06d}.txt').write_text(
                        '\n'.join(samples_decoded))
                    (sample_dir / f'reals_{current_grow_index}_{global_step:06d}.txt').write_text(
                        '\n'.join(reals_decoded))

                cur_step += 1
                global_step += 1

            torch.save(G, str(checkpoint_dir / f'G_{global_step:06d}.pth'))
            torch.save(D, str(checkpoint_dir / f'D_{global_step:06d}.pth'))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='wgan-vq')
    parser.add_argument('--device', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--g-lr', type=float, default=0.0001)
    parser.add_argument('--d-lr', type=float, default=0.0001)
    parser.add_argument('--latent-size', type=int, default=128)
    parser.add_argument('--log-every', type=int, default=250)
    parser.add_argument('--sample-every', type=int, default=2500)
    parser.add_argument('--n-sample', type=int, default=32)
    parser.add_argument('--n-print-samples', type=int, default=8)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--steps-per-stage', type=int, default=25_000)
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
