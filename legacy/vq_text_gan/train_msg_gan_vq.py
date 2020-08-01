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
from vq_text_gan.models.msg_gan import Generator, Discriminator
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

    if args.attn:
        G_attn = [False, False, True, False]
        D_attn = [False, True, False, False]
    else:
        G_attn = False
        D_attn = False


    G = Generator(args.latent_size, [512, 512, 256, 128], quant_dim, attn=G_attn).to(device)
    D = Discriminator([128, 256, 512, 512], quant_dim, attn=D_attn).to(device)

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

    G_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    D_opt = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    z_sample = torch.randn(args.n_sample, args.latent_size, 1).to(device)

    loss_f = WGAN_GP(D)
    #loss_f = RelativisticAverageHingeLoss(D)

    try:
        global_step = 0
        for epoch in range(args.epochs):
            g_loss_sum = 0
            d_loss_sum = 0

            p_fake_sum = 0
            p_real_sum = 0

            vq_diffs_sum = [0] * depth

            D_gammas_sum = OrderedDict()
            G_gammas_sum = OrderedDict()

            start_time = time.time()

            cur_step = 0

            for step, reals in enumerate(batches):
                #reals_code = [code.to(device) for code in reals_code]
                #reals_embed = [q.embed_code(c).transpose(1, 2) for q, c in zip(quantizers, reals_code)]

                reals = [real.to(device) for real in reals]

                batch_size = reals[0].size(0)

                z = torch.randn(batch_size, args.latent_size, 1).to(device)

                # Optimize the discriminator
                fake_out = G(z)
                fake_out = [t.detach() for t in fake_out]
                # fake_embeds = [q(
                #     o.transpose(1, 2)
                # )[0].transpose(1, 2).detach() for q, o in zip(quantizers, fake_out)]

                D_opt.zero_grad()

                loss_d, p_fake, p_real = loss_f.d_loss(reals, fake_out[::-1])
                loss_d.backward()

                D_opt.step()

                # Optimize generator
                fake_out = G(z)
                _, vq_diffs, fake_codes = list(zip(*[q(
                    o.transpose(1, 2))
                    for q, o in zip(quantizers, fake_out)
                ]))
                #fake_out = [t.transpose(1, 2) for t in fake_out]

                G_opt.zero_grad()

                loss_g = loss_f.g_loss(reals, fake_out[::-1])
                #loss_g += 0.01 * sum(vq_diffs)
                loss_g.backward()

                G_opt.step()

                if args.use_ema:
                    update_average(G_shadow, G_orig, beta=0.999)

                g_loss_sum += float(loss_g)
                d_loss_sum += float(loss_d)

                p_fake_sum += float(p_fake)
                p_real_sum += float(p_real)

                vq_diffs_sum = [v_old + float(v_new) for v_old, v_new in zip(vq_diffs_sum, vq_diffs)]

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

                    vq_diffs_avg = repr_list([diff / cur_step for diff in vq_diffs_sum])

                    logging.info(f'[EPOCH {epoch + 1:03d}] [{step:05d} / {len(batches):05d}] ' +
                                 # f'grow_index: {current_grow_index}/{depth - 1}, ' +
                                 f'loss_d: {d_loss_sum / cur_step:.5f}, loss_g: {g_loss_sum / cur_step:.5f}, ' +
                                 f'p_fake: {p_fake_sum / cur_step:.5f}, p_real: {p_real_sum / cur_step:.5f}, ' +
                                 (
                                     f'd_attn_gammas: [{D_gammas_avg}], g_attn_gammas: [{G_gammas_avg}], ' if args.attn else '') +
                                 f'vq_diffs: [{vq_diffs_avg}], ' +
                                 f'batches/s: {batches_per_sec:02.2f}')

                    g_loss_sum = d_loss_sum = 0
                    p_fake_sum = p_real_sum = 0

                    vq_diffs_sum = [0] * depth

                    D_gammas_sum = OrderedDict()
                    G_gammas_sum = OrderedDict()

                    start_time = time.time()

                if global_step % args.sample_every == 0:
                    sample_out = G_sample(z_sample)
                    sample_codes= [q(
                        o.transpose(1, 2)
                    )[2] for q, o in zip(quantizers, sample_out)]
                    sample_logits = [vq_model.decode_code(sample_code, depth - 1 - i) for i, sample_code in
                                     enumerate(sample_codes)]
                    samples_decoded = [decode(logits.argmax(-1)) for logits in sample_logits]

                    real_codes = [q(
                        o.transpose(1, 2)
                    )[2] for q, o in zip(quantizers, reals)]
                    reals_logits = [vq_model.decode_code(code[:args.n_sample], i) for i, code in enumerate(real_codes)]
                    reals_decoded = [decode(logits.argmax(-1)) for logits in reals_logits]

                    (sample_dir / f'fakes_{global_step:06d}.txt').write_text(
                        '\n\n'.join(map(lambda g: '\n'.join(g), zip(*samples_decoded))))
                    (sample_dir / f'reals_{global_step:06d}.txt').write_text(
                        '\n\n'.join(map(lambda g: '\n'.join(g), zip(*reals_decoded))))

                cur_step += 1
                global_step += 1

            torch.save(G, str(checkpoint_dir / f'G_{global_step:06d}.pth'))
            torch.save(D, str(checkpoint_dir / f'D_{global_step:06d}.pth'))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='msg-gan-vq')
    parser.add_argument('--device', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--g-lr', type=float, default=0.0003)
    parser.add_argument('--d-lr', type=float, default=0.0003)
    parser.add_argument('--latent-size', type=int, default=256)
    parser.add_argument('--log-every', type=int, default=200)
    parser.add_argument('--sample-every', type=int, default=2000)
    parser.add_argument('--n-sample', type=int, default=16)
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
