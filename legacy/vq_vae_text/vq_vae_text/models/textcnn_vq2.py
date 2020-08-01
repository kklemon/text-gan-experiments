import torch
import math
import numpy as np

from torch import nn
from torch.nn import functional as F

from typing import Union

from vq_vae_text.modules import Quantize, CategoricalNoise, SlicedQuantize, ChannelWiseLayerNorm, wn_conv_transpose1d, wn_conv1d, wn_linear, Noise, EqualizedConv1d, EqualizedConvTranspose1d, Attention, DecomposedQuantize
from vq_vae_text.models.transformer import PositionalEncoding


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

def activation():
    return nn.ELU()


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.conv = nn.Sequential(
            activation(),
            nn.Conv1d(in_channel, channel, kernel_size=kernel_size, padding=padding),
            #PixelNorm(),

            #nn.ELU(),
            #nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=padding, dilation=dilation),

            activation(),
            nn.Conv1d(channel, in_channel, kernel_size=1),
            #PixelNorm(),
        )

        #self.norm = ChannelWiseLayerNorm(in_channel)

    def forward(self, input):
        return self.conv(input) + input


class GatesResBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            activation(),
            nn.Conv1d(in_channel, channel, kernel_size=kernel_size, padding=padding),
            ChannelWiseLayerNorm(channel),

            activation()
        )

        self.conv2 = nn.Conv1d(channel, in_channel, kernel_size=kernel_size, padding=padding)
        self.gate = nn.Sequential(
            nn.Conv1d(channel, in_channel, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid()
        )

        self.norm = ChannelWiseLayerNorm(in_channel)

    def forward(self, input):
        out = self.conv1(input)
        return self.norm(input + self.conv2(out) * self.gate(out))


class MultiheadedSelfAttention(nn.MultiheadAttention):
    def forward(self, input):
        out = input.permute(2, 0, 1)
        out = super().forward(out, out, out)[0]
        out = out.permute(1, 2, 0)
        return out


class ResStack(nn.Module):
    def __init__(self, n_blocks, *args, **kwargs):
        super().__init__()

        self.res_blocks = nn.Sequential(*[ResBlock(*args, **kwargs) for i in range(n_blocks)])

    def forward(self, input):
        return self.res_blocks(input) + input


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, res_channel, n_res_blocks=2, attention=True):
        super().__init__()

        blocks = [
            nn.Conv1d(in_channel, channel, kernel_size=1),
            nn.ELU()
        ]

        for i in range(2):
            for j in range(n_res_blocks):
                blocks.append(ResBlock(channel, res_channel, kernel_size=3, padding=1))

            blocks += [
                nn.Conv1d(channel, channel, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
            ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class EncoderAlt(nn.Module):
    def __init__(self, in_channel, channel, res_channel, n_res_blocks=2, attention=True):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv1d(in_channel, channel // 2, 4, stride=2, padding=1),
            activation(),

            nn.Conv1d(channel // 2, channel, 4, stride=2, padding=1),
            activation(),

            nn.Conv1d(channel, channel, 3, padding=1),
        )

        self.query_res_block = ResBlock(channel, res_channel)
        self.key_res_block = ResBlock(channel, res_channel)

        self.attention = Attention(channel, channel, channel, n_heads=8)
        self.out_res_block = ResBlock(channel, res_channel)

    def forward(self, input):
        out = self.down(input)
        out = self.attention(out)
        out = self.out_res_block(out)
        return out


class EncoderAlt2(nn.Module):
    def __init__(self, in_channel, channel, res_channel, n_res_blocks=2, attention=True):
        super().__init__()

        # self.down = nn.Sequential(
        #     nn.Conv1d(in_channel, channel // 2, 4, stride=2, padding=1),
        #     activation(),
        #
        #     nn.Conv1d(channel // 2, channel, 4, stride=2, padding=1),
        #     activation(),
        #
        #     nn.Conv1d(channel, channel, 3, padding=1),
        # )
        #
        # self.res_blocks = nn.Sequential(*[GatesResBlock(channel, res_channel) for _ in range(n_res_blocks)])

        self.first = nn.Sequential(
            nn.Conv1d(in_channel, channel, kernel_size=1),
        )

        self.down = nn.Sequential(
            GatesResBlock(channel, res_channel),

            activation(),
            nn.Conv1d(channel, channel, 2, stride=2),

            GatesResBlock(channel, res_channel),

            activation(),
            nn.Conv1d(channel, channel, 2, stride=2),
        )

        self.res = GatesResBlock(channel, res_channel)

        self.attention = Attention(channel, channel // 4, channel, n_heads=1)

        self.post_res = nn.Sequential(
            activation(),
            nn.Conv1d(channel, channel, kernel_size=1),

            activation()
        )
        self.post_attn = nn.Sequential(
            activation(),
            nn.Conv1d(channel, channel, kernel_size=1),
            ChannelWiseLayerNorm(channel),

            activation()
        )

        self.final = nn.Sequential(
            activation(),
            nn.Conv1d(channel, channel, kernel_size=1),
            ChannelWiseLayerNorm(channel),

            activation()
        )

        self.attn_norm = ChannelWiseLayerNorm(channel)
        self.out_norm = ChannelWiseLayerNorm(channel)

    def forward(self, input):
        # out = self.down(input)
        # res = self.res_blocks(out)

        out = self.first(input)
        out = self.down(out)
        res = self.res(out)

        att = self.post_attn(self.attn_norm(self.attention(res, out)))
        res = self.post_res(out)

        return self.out_norm(self.final(att + res))


Encoder = EncoderAlt2


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, res_channel, n_res_blocks=2):
        super().__init__()

        blocks = [
            nn.Conv1d(in_channel, channel, kernel_size=1),
            activation()
        ]

        for i in range(2):
            blocks += [
                nn.ConvTranspose1d(channel, channel, kernel_size=2, stride=2, padding=1),
                activation(),
            ]

            for j in range(n_res_blocks):
                blocks.append(ResBlock(channel, res_channel, kernel_size=3, padding=1))

        blocks.append(nn.Conv1d(channel, out_channel, kernel_size=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class DecoderAlt(nn.Module):
    def __init__(self, in_channel, out_channel, channel, res_channel, n_res_blocks=2):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv1d(in_channel, channel, kernel_size=1),
            ResBlock(channel, res_channel)
        )

        self.attention = Attention(channel, channel // 4, channel, n_heads=1)
        self.res_block = ResBlock(channel, res_channel)
        self.up = nn.Sequential(
            nn.ConvTranspose1d(channel, channel // 2, 4, stride=2, padding=1),
            activation(),

            nn.ConvTranspose1d(
                channel // 2, channel, 4, stride=2, padding=1
            )
        )
        self.out_res_block = nn.Sequential(
            ResBlock(channel, res_channel),
            activation(),
            nn.Conv1d(channel, out_channel, kernel_size=1)
        )

    def forward(self, input):
        out = self.first(input)
        out = self.attention(out, out)
        out = self.res_block(out)
        out = self.up(out)
        out = self.out_res_block(out)
        return out


class DecoderAlt2(nn.Module):
    def __init__(self, in_channel, out_channel, channel, res_channel, n_res_blocks=2):
        super().__init__()

        self.first = nn.Conv1d(in_channel, channel, kernel_size=1)

        self.res_blocks = nn.Sequential(*[GatesResBlock(channel, res_channel) for _ in range(n_res_blocks)])

        self.attention = Attention(channel, channel, channel, n_heads=16)

        self.post_res = nn.Sequential(
            activation(),
            nn.Conv1d(channel, channel, kernel_size=1),
            ChannelWiseLayerNorm(channel),

            activation()
        )
        self.post_attn = nn.Sequential(
            activation(),
            nn.Conv1d(channel, channel, kernel_size=1),
            ChannelWiseLayerNorm(channel),

            activation()
        )

        self.combine = nn.Sequential(
            activation(),
            nn.Conv1d(channel, channel, kernel_size=1),
            ChannelWiseLayerNorm(channel),

            activation()
        )

        # self.up = nn.Sequential(
        #     nn.ConvTranspose1d(channel, channel // 2, 4, stride=2, padding=1),
        #     activation(),
        #
        #     nn.ConvTranspose1d(
        #         channel // 2, out_channel, 4, stride=2, padding=1
        #     )
        # )

        self.up = nn.Sequential(
            nn.ConvTranspose1d(channel, channel, 2, stride=2),

            GatesResBlock(channel, res_channel),

            activation(),
            nn.ConvTranspose1d(channel, channel, 2, stride=2),

            GatesResBlock(channel, res_channel),
            activation(),

            nn.Conv1d(channel, out_channel, kernel_size=1),
        )

        self.attn_norm = ChannelWiseLayerNorm(channel)
        self.out_norm = ChannelWiseLayerNorm(channel)

    def forward(self, input):
        out = self.first(input)
        res = self.res_blocks(out)

        att = self.post_attn(self.attn_norm(self.attention(res, out)))
        res = self.post_res(out)

        out = self.out_norm(self.combine(att + res))

        return self.up(out)



Decoder = DecoderAlt2


class TextCNNVQ2(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 channel: int,
                 res_channel: int,
                 n_res_block: int,
                 tau: float,
                 pad_idx: Union[None, int],
                 input_noise=0.0,
                 embed_dropout=0.0,
                 num_vq_embeds: int = 512,
                 vq_embed_dim: int = 64,
                 vq_loss_alpha=1.0,
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
        #self.pos_encoder = PositionalEncoding(embed_dim, max_len=1024)
        #self.quant_pos_encoder = PositionalEncoding(vq_embed_dim, max_len=256)

        self.enc_a = Encoder(embed_dim, channel, res_channel, n_res_block)
        self.enc_b = Encoder(channel, channel, res_channel, n_res_block)

        self.quantize_conv_b = nn.Conv1d(channel, vq_embed_dim, 1)
        #self.quantize_b = SlicedQuantize(d_slice, dim=vq_embed_dim, n_embed=num_vq_embeds)
        self.quantize_b = Quantize(vq_embed_dim, num_vq_embeds)
        #self.quantize_b = DecomposedQuantize(16, vq_embed_dim, num_vq_embeds)
        self.dec_b = Decoder(vq_embed_dim, vq_embed_dim, channel, res_channel, n_res_block)

        self.quantize_conv_a = nn.Conv1d(vq_embed_dim + channel, vq_embed_dim, 1)
        #self.quantize_a = SlicedQuantize(d_slice, dim=vq_embed_dim, n_embed=num_vq_embeds)
        self.quantize_a = Quantize(vq_embed_dim, num_vq_embeds)
        #self.quantize_a = DecomposedQuantize(64, vq_embed_dim, num_vq_embeds)
        self.upsample_b = nn.Sequential(
            nn.ConvTranspose1d(vq_embed_dim, channel, 4, stride=2, padding=1),
            nn.ELU(),

            nn.ConvTranspose1d(
                channel, vq_embed_dim, 4, stride=2, padding=1
            ),
        )
        self.dec = Decoder(
            vq_embed_dim + vq_embed_dim,
            vocab_size,
            channel,
            res_channel,
            n_res_block,
        )

        self.noise = Noise(alpha=0.05)

        self.nll = nn.NLLLoss(reduction='none', ignore_index=self.pad_idx)

    def get_vq_blend(self):
        blend = self.blend_step / self.blend_steps
        #return 1
        return (np.sin(np.pi * blend - np.pi * 0.5) + 1.0) / 2.0 if blend < 1.0 else 1.0

    def get_quantization_layers(self):
        return [self.quantize_a, self.quantize_b]

    def encode(self, x):
        # x ∈ [B, S]
        x = self.embed(x)
        #x = self.pos_encoder(x.permute(1, 0, 2)).permute(1, 2, 0)
        x = x.transpose(1, 2)

        enc_a = self.enc_a(x)
        enc_b = self.enc_b(enc_a)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 1)
        quant_b_, diff_b, ids_b = self.quantize_b(
            self.noise(torch.tanh(quant_b))
        )
        if self.blend_step >= 10000:
            quant_b = quant_b_

        quant_b = self.noise(quant_b).permute(0, 2, 1)

        dec_b = self.dec_b(quant_b)
        enc_a = torch.cat([dec_b, enc_a], 1)

        quant_a = self.quantize_conv_a(enc_a).permute(0, 2, 1)
        quant_a_, diff_a, ids_a = self.quantize_a(
            self.noise(torch.tanh(quant_a))
        )
        if self.blend_step >= 10000:
            quant_a = quant_a_
        quant_a = self.noise(quant_a).permute(0, 2, 1)


        return (quant_a, quant_b), diff_b + diff_a, (ids_a, ids_b)

    def decode(self, z):
        quant_a, quant_b = z

        upsample_t = self.upsample_b(quant_b)
        quant = torch.cat([upsample_t, quant_a], 1)
        dec = self.dec(quant)
        dec = dec.permute(0, 2, 1)          # x ∈ [B, S, C]

        logits = dec / self.tau
        logp_probs = F.log_softmax(logits, dim=-1)

        return logp_probs

    def decode_code(self, codes):
        code_a, code_b = codes

        quant_a = self.quantize_a.embed_code(code_a)
        quant_a = quant_a.transpose(1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.transpose(1, 2)

        dec = self.decode((quant_a, quant_b))

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
