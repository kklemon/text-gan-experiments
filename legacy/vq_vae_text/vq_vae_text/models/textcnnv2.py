import torch
import math
import numpy as np

from torch import nn
from torch.nn import functional as F

from typing import Union

from vq_vae_text.modules import Quantize, GeometricCategoricalDropout, CategoricalNoise, SlicedQuantize, ChannelWiseLayerNorm, wn_conv_transpose1d, wn_conv1d, wn_linear, Attention
from vq_vae_text.models.transformer import PositionalEncoding


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(in_channel, channel, kernel_size=kernel_size, padding=padding + 2, dilation=3),

            nn.ELU(),
            nn.Conv1d(channel, in_channel, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, input):
        return self.conv(input) + input


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, res_channel, n_res_blocks=2):
        super().__init__()

        self.first = nn.Conv1d(in_channel, channel, kernel_size=4, stride=2, padding=1)

        blocks = []
        for i in range(n_res_blocks):
            blocks.append(ResBlock(channel, res_channel))

        self.blocks = nn.Sequential(*blocks)

        self.attention = Attention(channel, channel // 4, channel, n_heads=1)

    def forward(self, input):
        out = keys = self.first(input)
        out = self.blocks(out)
        out = out + self.attention(out, out)
        return out


class Decoder(nn.Module):
    def __init__(self, channel, out_channel, res_channel, n_res_blocks=2):
        super().__init__()

        blocks = []

        for i in range(n_res_blocks):
            blocks.append(ResBlock(channel, res_channel))

        self.blocks = nn.Sequential(*blocks)

        self.attention = Attention(channel, channel // 4, channel, n_heads=1)

        self.final = nn.ConvTranspose1d(channel, out_channel, kernel_size=4, stride=2, padding=1)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        out = self.blocks(input)
        out = out + self.attention(out, out)
        out = self.final(out)
        return out


class TextCNNV2(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 channel: int,
                 res_channel: int,
                 n_res_blocks: int,
                 n_encoders: int,
                 tau: float,
                 pad_idx: Union[None, int],
                 input_noise=0.0,
                 embed_dropout=0.0,
                 num_vq_embeds: int = 512,
                 vq_embeds_dim: int = None,
                 vq_loss_alpha=0.25,
                 vq_decay=0.99,
                 ignore_quant=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.tau = tau
        self.vq_loss_alpha = vq_loss_alpha
        self.ignore_quant = ignore_quant

        self.vq_loss = 0
        self.nll_loss = 0
        self.acc = 0

        self.vq_blend = 0.0
        self.blend_steps = 5000
        self.blend_step = 0

        self.vq_embeds_dim = vq_embeds_dim

        self.input_noise = CategoricalNoise(vocab_size, input_noise)
        self.embed_dropout = nn.Dropout(embed_dropout)

        self.embed = nn.Embedding(vocab_size, channel, padding_idx=pad_idx, max_norm=1.0)

        self.encoder = nn.Sequential(*[Encoder(channel, channel, res_channel, n_res_blocks)
                                       for i in range(n_encoders)])
        self.decoder = nn.Sequential(*[Decoder(channel, channel, res_channel, n_res_blocks)
                                       for i in range(n_encoders)[::-1]])

        self.conv_to_quant = nn.Conv1d(channel, vq_embeds_dim, kernel_size=1)
        self.quant_to_conv = nn.Conv1d(vq_embeds_dim, channel, kernel_size=1)

        self.quantize = Quantize(dim=vq_embeds_dim, n_embed=num_vq_embeds, decay=vq_decay)

        self.conv_to_logits = nn.Conv1d(channel, vocab_size, kernel_size=1)

        self.nll = nn.NLLLoss(reduction='none', ignore_index=self.pad_idx)

    def get_quantization_layers(self):
        return [self.quantize]

    def encode(self, input):
        out = self.embed(input)
        out = out.permute(0, 2, 1)

        out = self.encoder(out)
        out = self.conv_to_quant(out).permute(0, 2, 1)

        quant, diff, code = self.quantize(out)
        quant = quant.permute(0, 2, 1)

        return [quant], diff, [code]

    def decode(self, quants):
        quant = quants[0]
        quant = self.quant_to_conv(quant)
        out = self.decoder(quant)
        logits = self.conv_to_logits(out)
        logits = logits.permute(0, 2, 1)

        logits = logits / self.tau
        logp_probs = F.log_softmax(logits, dim=-1)

        return logp_probs

    def decode_code(self, codes):
        code = codes[0]
        quant = self.quantize.embed_code(code)
        quant = quant.permute(0, 2, 1)
        x = self.decode([quant])
        return x

    def forward(self, x):
        z, diff, ids = self.encode(x)
        logp = self.decode(z)

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

        return self.nll_loss + self.vq_loss_alpha * self.vq_loss

    def latest_losses(self):
        return {
            'nll': self.nll_loss,
            'ppl': math.exp(self.nll_loss),
            'bpc': self.nll_loss / math.log(2),
            'vq': self.vq_loss,
            'acc': self.acc,
        }
