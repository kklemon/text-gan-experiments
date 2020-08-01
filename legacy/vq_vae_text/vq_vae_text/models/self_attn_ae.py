import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from typing import Union

from vq_vae_text.modules import Quantize, CategoricalNoise


def wn_conv1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))


def wn_conv_transpose1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class BaseBlock(nn.Module):
    def __init__(self, dim, dim_feedforward, mode, n_head=8, dropout=0.1):
        super().__init__()

        if mode == 'shrink':
            out_dim = dim // 2
        elif mode == 'expand':
            out_dim = dim * 2
        else:
            raise ValueError('mode argument must be either shrink or expand')

        self.encoder = nn.TransformerEncoderLayer(dim, n_head, dim_feedforward, dropout)
        self.shrink = nn.Linear(dim, out_dim)

    def forward(self, src):
        l, bs, e = src.size()

        out = self.encoder(src)
        out = out.permute(1, 0, 2)
        out = F.relu(self.shrink(out))
        out = out.view(bs, -1, e)
        out = out.permute(1, 0, 2)

        return out


class SelfAttentionAE(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 dim: int,
                 dim_feedforward: int,
                 n_fold: int,
                 tau: float,
                 pad_idx: Union[None, int],
                 input_noise=0.0,
                 embed_dropout=0.1,
                 num_vq_embeds: int = 512,
                 vq_embeds_dim: int = None,
                 vq_loss_alpha=1.0,
                 ignore_quant=False):
        super().__init__()

        self.pad_idx = pad_idx
        self.tau = tau
        self.vq_loss_alpha = vq_loss_alpha
        self.ignore_quant = ignore_quant

        self.vq_loss = 0
        self.nll_loss = 0
        self.acc = 0

        self.vq_blend = 0.0
        self.blend_steps = 10000
        self.blend_step = 0

        self.vq_embeds_dim = vq_embeds_dim

        self.input_noise = CategoricalNoise(vocab_size, input_noise)
        self.embed_dropout = nn.Dropout(embed_dropout)

        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)
        self.quantize = Quantize(vq_embeds_dim, num_vq_embeds, decay=0.95)

        self.encode_to_quants = nn.Linear(dim, vq_embeds_dim)
        self.quants_to_decode = nn.Linear(vq_embeds_dim, dim)

        self.encoder = nn.Sequential(*[BaseBlock(dim, dim_feedforward, mode='shrink') for _ in range(n_fold)])
        self.decoder = nn.Sequential(*[BaseBlock(dim, dim_feedforward, mode='expand') for _ in range(n_fold)])

        self.nll = nn.NLLLoss(reduction='none', ignore_index=self.pad_idx)

    def get_vq_blend(self):
        blend = self.blend_step / self.blend_steps
        return (np.sin(np.pi * blend - np.pi * 0.5) + 1.0) / 2.0 if blend < 1.0 else 1.0

    def encode(self, x):
        # x ∈ [B, S]
        x = self.input_noise(x)
        x = self.embed(x)
        x = self.embed_dropout(x)

        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = self.encode_to_quants(x)
        x = x.transpose(0, 1)

        z, diff, ids = self.quantize(x)  # z ∈ [B, S / 2^k, Q]

        blend = self.get_vq_blend()
        z = z * blend + x * (1 - blend)
        z += torch.rand_like(z) * 0.5

        if self.ignore_quant:
            z = x
            diff = diff.detach()

        return z, diff, ids

    def decode(self, z):
        x = z.transpose(0, 1)
        x = self.quants_to_decode(x)
        x = self.decoder(x)
        x = x.transpose(0, 1)

        logits = x / self.tau
        logp_probs = F.log_softmax(logits, dim=-1)

        return logp_probs

    def decode_codes(self, code):
        quant = self.quantize.embed_code(code)
        x = self.decode(quant)
        return x

    def forward(self, x):
        z, diff, ids = self.encode(x)
        logp = self.decode(z)

        if self.training:
            self.blend_step += 1

        return logp, z, diff, ids

    def compute_accuracy(self, recon_probs, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target)

        lens = mask.sum(-1).float()
        corr = ((recon_probs.argmax(-1) == target) * mask).sum(-1) / lens

        acc = corr.double().mean().item()

        return acc

    def loss_function(self, inputs, target):
        logp, z, diff, ids = inputs

        if self.pad_idx is not None:
            mask = target != self.pad_idx
        else:
            mask = torch.ones_like(target)
        lens = mask.sum(-1).float()

        acc = self.compute_accuracy(logp, target, mask=mask)

        bs = logp.size(0)

        logp = logp.view(-1, logp.size(-1))
        target = target.reshape(-1)

        nll_loss = self.nll(logp, target).view(bs, -1) * mask
        nll_loss = (nll_loss.sum(-1) / lens).mean()

        self.vq_loss = diff
        self.nll_loss = nll_loss
        self.acc = acc

        return self.nll_loss + self.vq_loss_alpha * self.get_vq_blend() * self.vq_loss

    def latest_losses(self):
        return {
            'nll': self.nll_loss,
            'vq': self.vq_loss,
            'acc': self.acc,
            'vq_blend': self.get_vq_blend()
        }


if __name__ == '__main__':
    enc = BaseBlock(128, 512, mode='shrink')
    dec = BaseBlock(128, 512, mode='expand')
    x = torch.rand(256, 32, 128)
    out = dec(enc(x))

    assert out.shape == x.shape
