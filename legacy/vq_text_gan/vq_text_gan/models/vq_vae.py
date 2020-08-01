import torch
import math
import functools
import operator
import numpy as np

from torch import nn
from torch.nn import functional as F

from typing import Union

from vq_text_gan.modules import Attention, Quantize, CategoricalNoise, ChannelWiseLayerNorm


def activation():
    return nn.ELU()


class GatedResBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            activation(),
            nn.Conv1d(in_channel, channel, kernel_size=kernel_size, padding=padding + dilation - 1, dilation=dilation),

            activation(),
        )

        self.conv2 = nn.Conv1d(channel, in_channel, kernel_size=kernel_size, padding=padding)
        self.gate = nn.Conv1d(channel, in_channel, kernel_size=kernel_size, padding=padding)

    def forward(self, input):
        out = self.conv1(input)
        return input + self.conv2(out) * torch.sigmoid(self.gate(out))


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, res_channel, n_res_blocks=2, n_heads=1):
        super().__init__()

        self.first = nn.Conv1d(in_channel, channel, kernel_size=4, stride=2, padding=1)
        self.res_blocks = nn.Sequential(*[GatedResBlock(channel, res_channel, dilation=3) for _ in range(n_res_blocks)])
        self.attention = Attention(channel, channel // 4, channel, n_heads=n_heads)
        self.final = nn.Conv1d(channel, channel, kernel_size=1)

    def forward(self, input):
        out = self.first(input)
        out = self.res_blocks(out)
        out = out + self.attention(out, out)
        out = self.final(out)
        return out


class Decoder(nn.Module):
    def __init__(self, channel, out_channel, res_channel, n_res_blocks=2, n_heads=1):
        super().__init__()

        self.res_blocks = nn.Sequential(*[GatedResBlock(channel, res_channel, dilation=3) for _ in range(n_res_blocks)])
        self.attention = Attention(channel, channel // 4, channel, n_heads=n_heads)
        self.final = nn.ConvTranspose1d(channel, out_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, input):
        out = self.res_blocks(input)
        out = out + self.attention(out, out)
        out = self.final(out)
        return out


class TextVQVAE(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 channel: int,
                 res_channel: int,
                 n_res_blocks: int,
                 depth: int,
                 tau: float,
                 pad_idx: Union[None, int],
                 n_heads=1,
                 input_noise=0.0,
                 embed_dropout=0.0,
                 num_vq_embeds: int = 512,
                 vq_embeds_dim: int = None,
                 vq_loss_alpha=0.25,
                 vq_decay=0.99):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.tau = tau
        self.depth = depth
        self.num_vq_embeds = num_vq_embeds
        self.vq_embeds_dim = vq_embeds_dim
        self.vq_loss_alpha = vq_loss_alpha

        self.vq_loss = 0
        self.nll_loss = 0
        self.acc = 0

        self.input_noise = CategoricalNoise(vocab_size, input_noise)
        self.embed_dropout = nn.Dropout(embed_dropout)

        self.embed = nn.Embedding(vocab_size, channel, padding_idx=pad_idx, max_norm=1.0)

        if not isinstance(n_heads, (tuple, list)):
            self.n_heads = [n_heads] * self.depth
        else:
            assert len(n_heads) == depth
            self.n_heads = n_heads

        if not isinstance(n_res_blocks, (tuple, list)):
            self.n_res_blocks = [n_res_blocks] * self.depth
        else:
            assert len(n_res_blocks) == depth
            self.n_res_blocks = n_res_blocks

        self.encoders = nn.ModuleList(Encoder(channel, channel, res_channel, n_res_blocks, n_heads)
                                      for n_res_blocks, n_heads in zip(self.n_res_blocks, self.n_heads))
        self.decoders = nn.ModuleList(Decoder(channel, channel, res_channel, n_res_blocks, n_heads)
                                      for n_res_blocks, n_heads in list(zip(self.n_res_blocks, self.n_heads))[::-1])

        self.conv_to_quant = nn.ModuleList(nn.Conv1d(channel, vq_embeds_dim, kernel_size=1) for _ in range(depth))
        self.quant_to_conv = nn.ModuleList(nn.Conv1d(vq_embeds_dim, channel, kernel_size=1) for _ in range(depth))

        self.quantize = nn.ModuleList(Quantize(dim=vq_embeds_dim, n_embed=num_vq_embeds, decay=vq_decay)
                                      for _ in range(depth))

        self.conv_to_logits = nn.Conv1d(channel, vocab_size, kernel_size=1)

        self.nll = nn.NLLLoss(reduction='none', ignore_index=self.pad_idx)

    def get_quantization_layers(self):
        return [self.quantize]

    def _check_depths(self, depths):
        if depths is None:
            depths = list(range(self.depth))
        elif isinstance(depths, int):
            depths = [depths]
        else:
            assert isinstance(depths, (tuple, list))
            assert all(map(lambda x: isinstance(x, int) and x >= 0, depths))
        return depths

    def encode(self, input, to_depths=None):
        to_depths = self._check_depths(to_depths)

        max_depth = max(to_depths)

        out = self.embed(input)
        out = out.permute(0, 2, 1)

        quants = []
        codes = []
        diffs = []

        for i, (encoder, conv_to_quant, quantize) in enumerate(zip(self.encoders, self.conv_to_quant, self.quantize)):
            out = encoder(out)
            if i in to_depths:
                quant = conv_to_quant(out).transpose(1, 2)
                quant, diff, code = quantize(
                    torch.tanh(quant)
                )
                quant = quant.transpose(1, 2)

                quants.append(quant)
                codes.append(code)
                diffs.append(diff)

            if i >= max_depth:
                break

        return quants, diffs, codes

    def decode(self, quant, from_depth):
        out = self.quant_to_conv[from_depth](quant)

        for i, decoder in list(enumerate(self.decoders))[:from_depth + 1]:
            out = decoder(out)

        logits = self.conv_to_logits(out).transpose(1, 2)
        logits = logits / self.tau
        logp = F.log_softmax(logits, dim=-1)

        return logp

    def decode_code(self, code, from_depth):
        quant = self.quantize[from_depth].embed_code(code)
        quant = quant.transpose(1, 2)
        logp = self.decode(quant, from_depth)
        return logp

    def forward(self, input):
        quants, diffs, codes = self.encode(input)
        logps = [self.decode(quant, from_depth=i) for i, quant in enumerate(quants)]

        return logps, quants, diffs, codes

    def compute_accuracy(self, prediction, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target)

        lengths = mask.sum(-1).float()
        correct = ((prediction == target) * mask).sum(-1) / lengths

        accuracy = correct.double().mean().item()

        return accuracy

    def loss_function(self, inputs, target):
        logps, quants, diffs, codes = inputs

        batch_size = target.size(0)

        if getattr(self, 'pad_idx'):
            mask = target != self.pad_idx
        else:
            mask = torch.ones_like(target)
        lengths = mask.sum(-1).float()

        stats = {}

        accuracies = []
        nlls = []
        for i, logp in enumerate(logps):
            accuracy = self.compute_accuracy(logp.argmax(-1), target, mask=mask)

            logp = logp.view(-1, self.vocab_size)

            nll = F.nll_loss(logp, target.view(-1), reduction='none').view(batch_size, -1) * mask
            nll = (nll.sum(-1) / lengths).mean()

            accuracies.append(accuracy)
            nlls.append(nll)

            stats[f'acc{i}'] = accuracy
            stats[f'nll{i}'] = nll
            stats[f'vq_diff{i}'] = nll

        nll_loss = functools.reduce(operator.add, nlls, 0)
        diff_loss = functools.reduce(operator.add, diffs, 0)

        loss = nll_loss + self.vq_loss_alpha * diff_loss

        acc_avg = np.mean(accuracies)

        stats['acc_avg'] = acc_avg
        stats['nll_sum'] = nll_loss

        return loss, stats


class SingleStageTextVQVAE(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 channel: int,
                 res_channel: int,
                 n_res_blocks: int,
                 depth: int,
                 tau: float,
                 pad_idx: Union[None, int],
                 n_heads=1,
                 input_noise=0.0,
                 embed_dropout=0.0,
                 num_vq_embeds: int = 512,
                 vq_embeds_dim: int = None,
                 vq_loss_alpha=0.25,
                 vq_decay=0.99):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.tau = tau
        self.depth = depth
        self.num_vq_embeds = num_vq_embeds
        self.vq_embeds_dim = vq_embeds_dim
        self.vq_loss_alpha = vq_loss_alpha

        self.vq_loss = 0
        self.nll_loss = 0
        self.acc = 0

        self.input_noise = CategoricalNoise(vocab_size, input_noise)
        self.embed_dropout = nn.Dropout(embed_dropout)

        self.embed = nn.Embedding(vocab_size, channel, padding_idx=pad_idx, max_norm=1.0)

        if not isinstance(n_heads, (tuple, list)):
            self.n_heads = [n_heads] * self.depth
        else:
            assert len(n_heads) == depth
            self.n_heads = n_heads

        if not isinstance(n_res_blocks, (tuple, list)):
            self.n_res_blocks = [n_res_blocks] * self.depth
        else:
            assert len(n_res_blocks) == depth
            self.n_res_blocks = n_res_blocks

        self.encoder = nn.Sequential(
            *[Encoder(channel, channel, res_channel, n_res_blocks, n_heads) for _ in range(depth)]
        )

        self.decoder = nn.Sequential(
            *[Decoder(channel, channel, res_channel, n_res_blocks, n_heads) for _ in range(depth)[::-1]]
        )

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
        out = self.conv_to_quant(out).transpose(1, 2)
        quant, diff, code = self.quantize(
            torch.tanh(out)
        )
        quant = quant.transpose(1, 2)

        return quant, diff, code

    def decode(self, quant):
        out = self.quant_to_conv(quant)
        out = self.decoder(out)

        logits = self.conv_to_logits(out).transpose(1, 2)
        logits = logits / self.tau
        logp = F.log_softmax(logits, dim=-1)

        return logp

    def decode_code(self, code):
        quant = self.quantize.embed_code(code)
        quant = quant.transpose(1, 2)
        logp = self.decode(quant)
        return logp

    def forward(self, input):
        quant, diff, code = self.encode(input)
        logp = self.decode(quant)

        return logp, quant, diff, code

    def compute_accuracy(self, prediction, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target)

        lengths = mask.sum(-1).float()
        correct = ((prediction == target) * mask).sum(-1) / lengths

        accuracy = correct.double().mean().item()

        return accuracy

    def loss_function(self, inputs, target):
        logp, z, diff, ids = inputs

        if self.pad_idx is not None:
            mask = target != self.pad_idx
        else:
            mask = torch.ones_like(target)
        lens = mask.sum(-1).float()

        acc = self.compute_accuracy(logp.argmax(-1), target, mask=mask)

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