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
from vq_text_gan.losses import WGAN_GP, RelativisticAverageHingeLoss
from vq_text_gan.modules import SelfAttention, CategoricalNoise


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

    depth = vq_model.depth
    num_classes = vq_model.quantize[0].n_embed

    dataset = LMDBDataset(args.vq_dataset)
    batches = DataLoader(dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    G = Generator(args.latent_size, [128, 128, 128, 128], num_classes, attn=args.attn).to(device)
    D = Discriminator([128, 128, 128, 128], num_classes, attn=args.attn).to(device)

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
    D_opt = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.9))

    z_sample = torch.randn(args.n_sample, args.latent_size, 1).to(device)

    #loss_f = RelativisticAverageHingeLoss(D)
    loss_f = WGAN_GP(D)

    try:
        global_step = 0
        for epoch in range(args.epochs):
            g_loss_sum = 0
            d_loss_sum = 0

            D_gammas_sum = OrderedDict()
            G_gammas_sum = OrderedDict()

            start_time = time.time()

            cur_step = 0

            for step, codes in enumerate(batches):
                codes = [code.to(device) for code in codes]
                codes_one_hot = [F.one_hot(code, num_classes=num_classes).transpose(1, 2).type(torch.float) for code in codes]

                batch_size = codes[0].size(0)

                # code_noise.p = 0.3 * (1.0 - min(1.0, interpol * 2))
                # code = code_noise(code)

                z = torch.randn(batch_size, args.latent_size, 1).to(device)

                # Optimize the discriminator
                fake_logits = G(z)
                fake_probs = [torch.softmax(logits, dim=1).detach() for logits in fake_logits]

                D_opt.zero_grad()

                loss_d = loss_f.d_loss(codes_one_hot, fake_probs[::-1])
                loss_d.backward()

                D_opt.step()

                # Optimize generator
                fake_logits = G(z)
                fake_probs = [torch.softmax(logits, dim=1) for logits in fake_logits]

                G_opt.zero_grad()

                loss_g = loss_f.g_loss(codes_one_hot, fake_probs[::-1])
                loss_g.backward()

                G_opt.step()

                if args.use_ema:
                    update_average(G_shadow, G_orig, beta=0.999)

                g_loss_sum += float(loss_g)
                d_loss_sum += float(loss_d)

                # p_fake_sum += float(p_fake)
                # p_real_sum += float(p_real)

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
                                 #f'p_fake: {p_fake_sum / cur_step:.5f}, p_real: {p_real_sum / cur_step:.5f}, ' +
                                 (f'd_attn_gammas: [{D_gammas_avg}], g_attn_gammas: [{G_gammas_avg}], ' if args.attn else '') +
                                 f'batches/s: {batches_per_sec:02.2f}')

                    g_loss_sum = d_loss_sum = 0

                    D_gammas_sum = OrderedDict()
                    G_gammas_sum = OrderedDict()

                    start_time = time.time()

                if global_step % args.sample_every == 0:
                    sample_codes = [logits.argmax(1) for logits in G_sample(z_sample)]
                    sample_logits = [vq_model.decode_code(sample_code, depth - 1 - i) for i, sample_code in enumerate(sample_codes)]
                    samples_decoded = [decode(logits.argmax(-1)) for logits in sample_logits]

                    reals_logits = [vq_model.decode_code(code[:args.n_sample], i) for i, code in enumerate(codes)]
                    reals_decoded = [decode(logits.argmax(-1)) for logits in reals_logits]

                    (sample_dir / f'fakes_{global_step:06d}.txt').write_text('\n\n'.join(map(lambda g: '\n'.join(g), zip(*samples_decoded))))
                    (sample_dir / f'reals_{global_step:06d}.txt').write_text('\n\n'.join(map(lambda g: '\n'.join(g), zip(*reals_decoded))))

                cur_step += 1
                global_step += 1

            torch.save(G, str(checkpoint_dir / f'G_{global_step:06d}.pth'))
            torch.save(D, str(checkpoint_dir / f'D_{global_step:06d}.pth'))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='msg-gan')
    parser.add_argument('--device', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--g-lr', type=float, default=0.0001)
    parser.add_argument('--d-lr', type=float, default=0.0004)
    parser.add_argument('--latent-size', type=int, default=128)
    parser.add_argument('--log-every', type=int, default=250)
    parser.add_argument('--sample-every', type=int, default=2500)
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
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--data-parallel', action='store_true')
    args = parser.parse_args()

    main(args)
