import logging
import time

from collections import OrderedDict
from pathlib import Path
from typing import List

import torch


class BaseModel:
    def __init__(self, result_folder):
        self.result_folder = Path(result_folder)

        self.sample_folder = self.result_folder / 'samples'
        self.sample_folder.mkdir(exist_ok=True)

        self.checkpoint_folder = self.result_folder / 'checkpoints'
        self.checkpoint_folder.mkdir(exist_ok=True)

    def train(self, epochs, log_every=100, sample_every=1000):
        raise NotImplementedError

    def save_samples(self, samples: List[str], prefix: str, step: int, ext='.txt'):
        dump = '\n'.join(samples)

        (self.result_folder / f'latest_{prefix}{ext}').write_text(dump)
        (self.sample_folder / f'{prefix}_{step:06d}{ext}').write_text(dump)


class TextGANModel(BaseModel):
    def __init__(self,
                 G, D,
                 G_opt, D_opt,
                 loss_f,
                 batches,
                 latent_size,
                 device,
                 result_folder,
                 sampling_model,
                 sample_decoder,
                 z_sample):
        super().__init__(result_folder)

        self.G = G
        self.D = D
        self.G_opt = G_opt
        self.D_opt = D_opt
        self.loss_f = loss_f
        self.batches = batches
        self.latent_size = latent_size
        self.device = device
        self.sampling_model = sampling_model
        self.sample_decoder = sample_decoder
        self.z_sample = z_sample

    @property
    def uses_attn(self):
        return self.G.attn and self.D.attn

    def build_log_string(self):
        log_string_base = '[EPOCH {epoch:03d}] [{step:05d} / {num_batches:05d}]'
        log_string_parts = OrderedDict(loss_d='{d_loss:.5f}', loss_g='{g_loss:.5f}')
        if self.uses_attn:
            log_string_parts.update(G_attn_gamma='{G_attn:.2f}', D_attn_gamma='{D_attn:.2f}')
        log_string_parts['batches/s'] = '{batches_per_sec:02.2f}'

        log_string = ', '.join(map(': '.join, log_string_parts.items()))

        return ' '.join([log_string_base, log_string])

    def train(self, epochs, log_every=10, sample_every=10):
        log_string = self.build_log_string()

        try:
            training_step = 0
            for epoch in range(epochs):
                g_loss_sum = 0
                d_loss_sum = 0

                G_attn_sum = 0
                D_attn_sum = 0

                start_time = time.time()

                cur_step = 0

                for step, reals in enumerate(self.batches):
                    reals = reals.to(self.device)

                    batch_size = reals.size(0)

                    z = torch.randn(batch_size, self.latent_size, 1).to(self.device)

                    # Optimize discriminator
                    fake_out = self.G(z)

                    self.D_opt.zero_grad()

                    d_loss = self.loss_f.loss_d(reals, fake_out.detach())
                    d_loss.backward()

                    self.D_opt.step()

                    # Optimize generator
                    fake_out = self.G(z)

                    self.G_opt.zero_grad()

                    g_loss = self.loss_f.loss_g(reals, fake_out)
                    g_loss.backward()

                    self.G_opt.step()

                    self.sampling_model.update(self.G)

                    g_loss_sum += float(g_loss)
                    d_loss_sum += float(d_loss)

                    if self.uses_attn:
                        G_attn_sum += float(self.G.attn.gamma)
                        D_attn_sum += float(self.D.attn.gamma)

                    if training_step % log_every == 0:
                        cur_step = min(step + 1, log_every)
                        batches_per_sec = cur_step / (time.time() - start_time)

                        log_msg = log_string.format(epoch=epoch + 1, step=step, num_batches=len(self.batches),
                                                    d_loss=d_loss_sum / cur_step, g_loss=g_loss_sum / cur_step,
                                                    G_attn=G_attn_sum / cur_step, D_attn=D_attn_sum / cur_step,
                                                    batches_per_sec=batches_per_sec)
                        logging.info(log_msg)

                        g_loss_sum = d_loss_sum = 0

                        G_attn_sum = 0
                        D_attn_sum = 0

                        start_time = time.time()

                    if training_step % sample_every == 0:
                        samples_embeds = self.sampling_model.model(self.z_sample)
                        samples = self.sample_decoder(samples_embeds)

                        reals_decode = self.sample_decoder(reals[:len(samples)])

                        self.save_samples(samples, 'fakes', training_step)
                        self.save_samples(reals_decode, 'reals', training_step)

                    cur_step += 1
                    training_step += 1

                torch.save(self.G, str(self.checkpoint_folder / f'G_{training_step:06d}.pth'))
                torch.save(self.D, str(self.checkpoint_folder / f'D_{training_step:06d}.pth'))
        except KeyboardInterrupt:
            pass
