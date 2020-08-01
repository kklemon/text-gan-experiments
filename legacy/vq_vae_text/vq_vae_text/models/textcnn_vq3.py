import torch
import math
import numpy as np

from torch import nn
from torch.nn import functional as F

from typing import Union, List, Tuple

from vq_vae_text.modules import Quantize, CategoricalNoise, SlicedQuantize, ChannelWiseLayerNorm, wn_conv_transpose1d, wn_conv1d, wn_linear, Noise
from vq_vae_text.models.transformer import PositionalEncoding
from .textcnn_vq2 import EncoderAlt2, DecoderAlt2


# class ResBlock(nn.Module):
#     def __init__(self, in_channel, channel):
#         super().__init__()
#
#         self.conv = nn.Sequential(
#             nn.ELU(),
#             nn.Conv1d(in_channel, channel, 3, padding=1),
#             ChannelWiseLayerNorm(channel),
#
#             nn.ELU(),
#             nn.Conv1d(channel, in_channel, 1),
#         )
#
#         self.norm = ChannelWiseLayerNorm(in_channel)
#
#     def forward(self, input):
#         return self.norm(self.conv(input) + input)
#
#
# class Encoder(nn.Module):
#     def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
#         super().__init__()
#
#         blocks = [
#             nn.Conv1d(in_channel, channel, kernel_size=1),
#
#             ResBlock(channel, n_res_channel),
#             nn.Conv1d(channel, channel, 4, stride=2, padding=1),
#
#             ResBlock(channel, n_res_channel),
#             nn.Conv1d(channel, channel, 4, stride=2, padding=1),
#             nn.ELU(),
#
#             nn.Conv1d(channel, channel, kernel_size=1),
#         ]
#
#         self.blocks = nn.Sequential(*blocks)
#
#     def forward(self, input):
#         return self.blocks(input)
#
#
# class Decoder(nn.Module):
#     def __init__(
#         self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
#     ):
#         super().__init__()
#
#         blocks = [
#             nn.Conv1d(in_channel, channel, 3, padding=1),
#
#             ResBlock(channel, n_res_channel),
#             nn.ConvTranspose1d(channel, channel, 4, stride=2, padding=1),
#
#             ResBlock(channel, n_res_channel),
#             nn.ConvTranspose1d(channel, channel, 4, stride=2, padding=1),
#             nn.ELU(),
#
#             nn.Conv1d(channel, out_channel, 1)
#         ]
#
#         self.blocks = nn.Sequential(*blocks)
#
#     def forward(self, input):
#         return self.blocks(input)


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(in_channel, channel, kernel_size=kernel_size, padding=padding, dilation=dilation),

            nn.ELU(),
            nn.Conv1d(channel, in_channel, kernel_size=kernel_size, padding=padding, dilation=dilation),
        )

        self.norm = ChannelWiseLayerNorm(in_channel)

    def forward(self, input):
        return self.norm(self.conv(input) + input)


class MultiheadedSelfAttention(nn.MultiheadAttention):
    def forward(self, input):
        out = input.permute(2, 0, 1)
        out = super().forward(out, out, out)[0]
        out = out.permute(1, 2, 0)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, res_channel, n_res_blocks=2, down=2, attn=False):
        super().__init__()

        blocks = [
            nn.Conv1d(in_channel, channel, kernel_size=1),
            nn.ELU()
        ]

        for i in range(down):
            for j in range(n_res_blocks):
                blocks.append(ResBlock(channel, res_channel, kernel_size=3, padding=1))

            blocks += [
                nn.Conv1d(channel, channel, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
            ]

        self.blocks = nn.Sequential(*blocks)

        if attn:
            self.attn = MultiheadedSelfAttention(channel, 8, dropout=0.1)
        else:
            self.attn = None

    def forward(self, input):
        out = skip = self.blocks(input)
        if self.attn:
            out = self.attn(out)
            out += skip
        return out


Encoder = EncoderAlt2

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, res_channel, n_res_blocks=2, up=2, attn=False):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv1d(in_channel, channel, kernel_size=1),
            nn.ELU()
        )

        if attn:
            self.attn = MultiheadedSelfAttention(channel, 8, dropout=0.1)
        else:
            self.attn = None

        blocks = []
        for i in range(up):
            blocks += [
                nn.ConvTranspose1d(channel, channel, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
            ]

            for j in range(n_res_blocks):
                blocks.append(ResBlock(channel, res_channel, kernel_size=3, padding=1))

        blocks.append(nn.Conv1d(channel, out_channel, kernel_size=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        out = skip = self.first(input)
        if self.attn:
            out = self.attn(out)
            out += skip
        out = self.blocks(out)
        return out

Decoder = DecoderAlt2


class TextCNNVQ3(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 channel: int,
                 res_channel: int,
                 n_res_block: int,
                 tau: float,
                 pad_idx: Union[None, int],
                 config: List[Tuple[int, int]],
                 input_noise=0.0,
                 embed_dropout=0.0,
                 vq_embed_dim: int = 8,
                 vq_loss_alpha=1.0,
                 attn=False,
                 d_slice=1,
                 ignore_quant=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.tau = tau
        self.vq_loss_alpha = vq_loss_alpha
        self.d_slice = d_slice
        self.ignore_quant = ignore_quant

        self.vq_loss = 0
        self.nll_loss = 0
        self.acc = 0

        self.vq_blend = 0.0
        self.blend_steps = 5000
        self.blend_step = 0

        self.vq_embed_dim = vq_embed_dim

        self.input_noise = CategoricalNoise(vocab_size, input_noise)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx, max_norm=1.0)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.cond_decoders = nn.ModuleList()
        self.quant_conv = nn.ModuleList()
        self.quantize = nn.ModuleList()

        for i, (factor, n_embeds) in enumerate(config):
            self.encoders.append(
                Encoder(embed_dim if i == 0 else channel, channel, res_channel, down=factor)
            )

            dec_in_channels = vq_embed_dim * (1 if i + 1 == len(config) else 2)
            dec_out_channels = vocab_size if i == 0 else vq_embed_dim

            self.decoders.append(
                Decoder(dec_in_channels, dec_out_channels, channel, res_channel, up=factor)
            )

            if i + 1 != len(config):
                self.cond_decoders.append(
                    Decoder(vq_embed_dim, vq_embed_dim, channel, res_channel, up=config[i + 1][0], n_res_blocks=1)
                )
            else:
                self.cond_decoders.append(None)

            self.quant_conv.append(nn.Conv1d(channel + (0 if i + 1 == len(config) else vq_embed_dim), vq_embed_dim, kernel_size=1))
            #self.quant_conv.append(nn.Conv1d(channel, vq_embed_dim, kernel_size=1))
            self.quantize.append(Quantize(vq_embed_dim, n_embeds))

        self.upsample_t = nn.Sequential(
            nn.ConvTranspose1d(vq_embed_dim, channel, 4, stride=2, padding=1),
            nn.ELU(),

            nn.ConvTranspose1d(
                channel, vq_embed_dim, 4, stride=2, padding=1
            ),
        )

        self.noise = Noise(alpha=0.05)

        self.nll = nn.NLLLoss(reduction='none', ignore_index=self.pad_idx)

    def get_quantization_layers(self):
        return list(self.quantize)

    def get_vq_blend(self):
        blend = self.blend_step / self.blend_steps
        #return 1
        return (np.sin(np.pi * blend - np.pi * 0.5) + 1.0) / 2.0 if blend < 1.0 else 1.0

    def encode(self, x):
        # x ∈ [B, S]
        x = self.embed(x)
        x = x.permute(0, 2, 1)

        # Encode
        encs = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encs.append(x)

        # Quantize in opposite direction to condition each latent on each other from top to bottom
        quants = []
        codes = []
        diffs = []
        for i, (encoding, conv, quantize, cond_decoder) in enumerate(list(zip(encs, self.quant_conv, self.quantize, self.cond_decoders))[::-1]):
            if i > 0:
                cond = cond_decoder(quants[i - 1])
                encoding = torch.cat([encoding, cond], dim=1)
            quant = conv(encoding).transpose(1, 2)
            quant_, diff, code = quantize(torch.tanh(quant))

            if self.blend_step >= 10_000:
                quant = quant_

            quant = quant.transpose(1, 2)

            quants.append(quant)
            codes.append(code)
            diffs.append(diff.unsqueeze(0))

        diff = torch.cat(diffs).sum()

        return quants[::-1], diff, codes[::-1]

    def decode(self, z):
        x = None
        for i, (quant, decoder) in enumerate(zip(z[::-1], self.decoders[::-1])):
            if i > 0:
                x = torch.cat([quant, x], dim=1)
            else:
                x = quant
            x = decoder(x)

        x = x.transpose(1, 2)

        # quant_b, quant_t = z
        # quant_t = self.upsample_t(quant_t)
        #
        # x = self.decoders[0](torch.cat([quant_t, quant_b], dim=1))
        # x = x.transpose(1, 2)

        logits = x / self.tau
        logp_probs = F.log_softmax(logits, dim=-1)

        return logp_probs

    def decode_code(self, codes):
        quants = [quantize.embed_code(code).transpose(1, 2) for code, quantize in zip(codes, self.quantize)]
        dec = self.decode(quants)
        return dec

    def forward(self, x):
        codes, diff, ids, = self.encode(x)
        logp = self.decode(codes)

        if self.training:
            self.blend_step += 1

        return logp, codes, diff, ids

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
            'vq_blend': self.get_vq_blend()
        }
