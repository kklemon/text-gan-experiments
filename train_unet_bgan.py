import argparse
import math
import time
import torch

from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader

from text_gans.nets.gan import Generator, UNetDiscriminator
from text_gans.bgan import multinomial_bgan_loss_unet
from text_gans.sampling import get_sampling_model
from text_gans.utils import init_weights, get_log2_seq_length, get_default_device
from text_gans.datasets import ByteLevelTextDataset


def main(dataset,
         device=None,
         batch_size=64,
         latent_size=256,
         seq_length=64,
         use_ema=False,
         n_sample=16,
         epochs=100,
         n_mc_samples=20,
         log_every=25,
         sample_every=250):
    device = get_default_device(device)

    log2_seq_length = get_log2_seq_length(seq_length)

    dataset = ByteLevelTextDataset(dataset, seq_length)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    result_dir = Path(f'results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-bgan-unet-text')
    result_dir.mkdir(parents=True)

    samples_dir = result_dir / 'samples'
    samples_dir.mkdir()

    checkpoint_dir = result_dir / 'checkpoints'
    checkpoint_dir.mkdir()

    model_depth = log2_seq_length - 1

    G = Generator(latent_size, [256] * model_depth, attn_at=3, out_dim=dataset.vocab_size)
    D = UNetDiscriminator(256, max_channel=256, depth=model_depth, attn_at=3, in_dim=dataset.vocab_size)

    G.apply(init_weights)
    D.apply(init_weights)

    G = G.to(device)
    D = D.to(device)

    sampling_model = get_sampling_model(G, use_ema)

    G.train()
    D.train()

    G_opt = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    D_opt = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))

    z_sample = torch.randn(n_sample, latent_size, 1).to(device)

    for epoch in range(epochs):
        g_loss_sum = 0
        d_loss_sum = 0

        p_fake_sum = 0
        p_real_sum = 0

        g_attn_sum = 0
        d_attn_sum = 0

        start_time = time.time()

        for step, reals in enumerate(dataloader):
            reals = reals.to(device)

            z = torch.randn(reals.size(0), latent_size, 1).to(device)

            fake_logits = G(z)

            d_loss, g_loss, p_fake, p_real = multinomial_bgan_loss_unet(
                D, fake_logits, reals, n_samples=n_mc_samples, tau=0.8
            )

            D_opt.zero_grad()
            G_opt.zero_grad()

            torch.autograd.backward([d_loss, g_loss])

            # penalty = gradient_penalty(D, fake_logits, reals_one_hot)
            # penalty.backward()

            D_opt.step()
            G_opt.step()

            sampling_model.update(G)

            g_loss_sum += float(g_loss)
            d_loss_sum += float(d_loss)

            p_fake_sum += float(p_fake)
            p_real_sum += float(p_real)

            g_attn_sum += float(G.attn.gamma)
            d_attn_sum += float(D.attn.gamma)

            if step % log_every == 0:
                cur_step = min(step + 1, log_every)
                batches_per_sec = cur_step / (time.time() - start_time)
                print(f'[EPOCH {epoch + 1:03d}] [{step:04d} / {len(dataloader):04d}] '
                      f'loss_d: {d_loss_sum / cur_step:.5f}, loss_g: {g_loss_sum / cur_step:.5f}, '
                      f'p_fake: {p_fake_sum / cur_step:.5f}, p_real: {p_real_sum / cur_step:.5f}, '
                      f'g_attn: {g_attn_sum / cur_step:.2f}, d_attn: {d_attn_sum / cur_step:.2f}, '
                      f'batches/s: {batches_per_sec:02.2f}')
                g_loss_sum = d_loss_sum = 0
                p_fake_sum = p_real_sum = 0

                g_attn_sum = 0
                d_attn_sum = 0

                start_time = time.time()

            if step % sample_every == 0:
                global_step = epoch * len(dataloader) + step

                samples = dataset.seq_to_text(sampling_model.model(z_sample).argmax(1))
                reals_decoded = dataset.seq_to_text(reals)

                (samples_dir / f'fakes_{global_step:06d}.txt').write_text('\n'.join(samples))
                (samples_dir / f'reals_{global_step:06d}.txt').write_text('\n'.join(reals_decoded))

        torch.save(G, str(checkpoint_dir / f'G_{epoch}.p'))
        torch.save(D, str(checkpoint_dir / f'D_{epoch}.p'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--latent-size', type=int, default=256)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--sample-every', type=int, default=250)
    parser.add_argument('--n-sample', type=int, default=16)
    parser.add_argument('--n-mc-samples', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--seq-length', type=int, default=64)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--device')
    args = parser.parse_args()

    main(**vars(args))
