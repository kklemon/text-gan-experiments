import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from typing import Union

from vq_vae_text.modules import Quantize, ResBlock, CategoricalNoise



class ChannelShrink(nn.Module):
    def forward(self, input):
        bs = input.size(0)
        ch = input.size(1)

        return input.view(bs, ch * 2, -1)


class ChannelExpand(nn.Module):
    def forward(self, input):
        bs = input.size(0)
        ch = input.size(1)

        return input.view(bs, ch // 2, -1)



class Encoder(nn.Module):
    def __init__(self, channel, res_channel, n_res_blocks):
        super().__init__()

        blocks = [
            nn.Conv1d(channel, channel // 2, kernel_size=4, stride=2, padding=1),
            nn.ELU(),

            nn.Conv1d(channel // 2, channel, kernel_size=4, stride=2, padding=1),
            nn.ELU(),

            nn.Conv1d(channel, channel, kernel_size=3, padding=1),
        ]

        for i in range(n_res_blocks):
            dilation = 2 ** i
            blocks.append(ResBlock(channel, res_channel, padding=dilation, dilation=dilation))

        blocks.append(nn.ELU())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


# class Encoder(nn.Module):
#     def __init__(self, channel, res_channel, n_res_blocks):
#         super().__init__()
#
#         blocks = []
#         #
#         # for i in range(2):
#         #     dilation = 2 ** i
#         #     blocks += [
#         #         nn.Conv1d(channel, channel // 2, kernel_size=4, stride=2, padding=1),
#         #         nn.ELU(),
#         #
#         #         nn.Conv1d(channel // 2, channel, kernel_size=3, padding=1),
#         #     ]
#         #
#         #     for j in range(n_res_blocks):
#         #         blocks.append(ResBlock(channel, res_channel, padding=dilation, dilation=dilation),)
#
#         for i in range(n_res_blocks):
#             dilation = 2 ** i
#             blocks += [
#                 nn.Conv1d(channel, channel // 2, kernel_size=4, stride=2, padding=1),
#                 nn.ELU(),
#
#                 nn.Conv1d(channel // 2, channel, kernel_size=3, padding=1),
#
#                 ResBlock(channel, res_channel, padding=dilation, dilation=dilation),
#             ]
#
#         blocks.append(nn.ELU())
#
#         self.blocks = nn.Sequential(*blocks)
#
#     def forward(self, input):
#         return self.blocks(input)



class Decoder(nn.Module):
    def __init__(self, channel, out_channel, res_channel, n_res_blocks):
        super().__init__()

        blocks = []
        for i in range(n_res_blocks):
            dilation = 1 #2 ** (n_res_blocks - i - 1)
            blocks.append(ResBlock(channel, res_channel, padding=dilation, dilation=dilation))

        blocks.append(nn.ELU())

        blocks += [
            nn.ConvTranspose1d(channel, channel // 2, kernel_size=4, stride=2, padding=1),
            nn.ELU(),

            nn.ConvTranspose1d(channel // 2,  out_channel, kernel_size=4, stride=2, padding=1),
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


# class Decoder(nn.Module):
#     def __init__(self, channel, out_channel, res_channel, n_res_blocks):
#         super().__init__()
#
#         blocks = []
#
#         # for i in range(2):
#         #     dilation = 2 ** i
#         #
#         #     for j in range(n_res_blocks):
#         #         blocks.append(ResBlock(channel, res_channel, padding=dilation, dilation=dilation))
#         #
#         #     blocks += [
#         #         nn.ConvTranspose1d(channel, channel // 2, kernel_size=4, stride=2, padding=1),
#         #         nn.ELU(),
#         #
#         #         nn.Conv1d(channel // 2, channel if i + 1 < n_res_blocks else out_channel, kernel_size=3, padding=1),
#         #     ]
#
#         for i in range(n_res_blocks):
#             dilation = 2 ** i
#             blocks += [
#                 ResBlock(channel, res_channel, padding=dilation, dilation=dilation),
#
#                 nn.ConvTranspose1d(channel, channel // 2, kernel_size=4, stride=2, padding=1),
#                 nn.ELU(),
#
#                 nn.Conv1d(channel // 2, channel if i + 1 < n_res_blocks else out_channel, kernel_size=3, padding=1),
#             ]
#
#         blocks.append(nn.ELU())
#
#         self.blocks = nn.Sequential(*blocks)
#
#     def forward(self, input):
#         return self.blocks(input)


class TextCNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 channel: int,
                 res_channel: int,
                 n_res_block,
                 tau: float,
                 pad_idx: Union[None, int],
                 input_embed_dim: int,
                 input_noise=0.0,
                 embed_dropout=0.1,
                 num_vq_embeds: int = 512,
                 vq_embeds_dim: int = None,
                 vq_loss_alpha=1.0,
                 ignore_quant=False,
                 **kwargs):
        super().__init__()

        self.pad_idx = pad_idx
        self.tau = tau
        self.vq_loss_alpha = vq_loss_alpha

        self.vq_loss = 0
        self.nll_loss = 0
        self.acc = 0

        self.vq_blend = 0.0
        self.blend_steps = 10000
        self.blend_step = 0

        self.vq_embeds_dim = vq_embeds_dim
        self.ignore_quant = ignore_quant

        self.input_noise = CategoricalNoise(vocab_size, input_noise)
        self.embed_dropout = nn.Dropout(embed_dropout)

        self.embed = nn.Embedding(vocab_size, input_embed_dim, padding_idx=pad_idx)
        self.quantize = Quantize(vq_embeds_dim, num_vq_embeds, decay=0.90)

        self.embeds_to_encode = nn.Conv1d(input_embed_dim, channel, kernel_size=3, padding=1)
        self.encode_to_quants = nn.Conv1d(channel, vq_embeds_dim, kernel_size=3, padding=1)
        self.quants_to_decode = nn.Conv1d(vq_embeds_dim, channel, kernel_size=3, padding=1)

        self.encoder = Encoder(channel, res_channel, n_res_block)
        self.decoder = Decoder(channel, vocab_size, res_channel, n_res_block)

        self.nll = nn.NLLLoss(reduction='none', ignore_index=self.pad_idx)

    def get_vq_blend(self):
        blend = self.blend_step / self.blend_steps
        return np.sin(np.pi * 0.5 * blend) if blend < 1.0 else 1.0

    def encode(self, x):
        x = self.input_noise(x)
        x = self.embed(x)
        x = self.embed_dropout(x)
        x = x.permute(0, 2, 1)
        x = self.embeds_to_encode(x)
        x = self.encoder(x)
        x = self.encode_to_quants(x)
        x = x.permute(0, 2, 1)

        z, diff, ids = self.quantize(x)

        if self.ignore_quant:
            z = x
            diff = diff.detach()

        blend = self.get_vq_blend()
        z = z * blend + x * (1 - blend)

        return z, diff, ids

    def decode(self, z):
        x = z.permute(0, 2, 1)
        x = self.quants_to_decode(x)
        logits = self.decoder(x).permute(0, 2, 1)
        logits = logits / self.tau

        logp_probs = F.log_softmax(logits, dim=-1)

        return logp_probs

    def decode_codes(self, code):
        quant = self.quantize.embed_code(code)
        x = self.decode(quant)
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

        return self.nll_loss + self.vq_loss_alpha * self.get_vq_blend() * self.vq_loss

    def latest_losses(self):
        return {
            'nll': self.nll_loss,
            'vq': self.vq_loss,
            'acc': self.acc,
            'vq_blend': self.get_vq_blend()
        }
