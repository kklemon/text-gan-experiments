import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from typing import Union

from vq_vae_text.modules import Quantize, CategoricalNoise, SlicedQuantize, ResBlock, ChannelWiseLayerNorm, wn_conv_transpose1d, wn_conv1d, wn_linear


class EncoderBlock(nn.Module):
    def __init__(self, channel, n_repeat=3):
        super().__init__()

        blocks = [
            nn.ELU(),
            wn_conv1d(channel, channel, kernel_size=3, padding=1),

            nn.ELU(),
            wn_conv1d(channel, channel, kernel_size=1),
        ]

        self.blocks = nn.Sequential(*blocks)
        self.down = nn.Conv1d(channel, channel, kernel_size=2, stride=2)

    def forward(self, input):
        out = self.blocks(input) + input
        out = self.down(out)
        return out


# class Encoder(nn.Module):
#     def __init__(self, channel, res_channel, n_res_blocks):
#         super().__init__()
#
#         blocks = [
#             nn.Conv1d(channel, channel // 2, kernel_size=2, stride=2),
#             nn.ELU(),
#
#             nn.Conv1d(channel // 2, channel, kernel_size=2, stride=2),
#             nn.ELU(),
#
#             nn.Conv1d(channel, channel, kernel_size=3, padding=1),
#         ]
#
#         for i in range(n_res_blocks):
#             blocks.append(ResBlock(channel, res_channel, padding=1))
#
#         blocks.append(nn.ELU())
#
#         self.blocks = nn.Sequential(*blocks)
#
#     def forward(self, input):
#         return self.blocks(input)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = wn_linear(style_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class ConstantInput(nn.Module):
    def __init__(self, shape):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, *shape), requires_grad=True)
        self.rep_dims = [1] * (self.input.ndim - 1)

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, *self.rep_dims)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, num_channel, style_dim, is_first_block=False):
        super().__init__()

        if is_first_block:
            self.first = ConstantInput([num_channel, 16])
        else:
            self.first = wn_conv_transpose1d(num_channel, num_channel, kernel_size=2, stride=2)

        self.conv1 = nn.Conv1d(num_channel, num_channel, kernel_size=3, padding=1)
        self.adain1 = AdaptiveInstanceNorm(num_channel, style_dim)
        self.activation1 = nn.ELU()

        self.conv2 = nn.Conv1d(num_channel, num_channel, kernel_size=3, padding=1)
        self.adain2 = AdaptiveInstanceNorm(num_channel, style_dim)
        self.activation2 = nn.ELU()

    def forward(self, input, style):
        out = self.first(input)

        skip = out

        out = self.conv1(out)
        out = self.adain1(out + skip, style)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.adain2(out, style)
        out = self.activation2(out)

        return out + skip


class Decoder(nn.Module):
    def __init__(self, channel, out_channel, style_dim, n_blocks):
        super().__init__()

        self.n_blocks = n_blocks - 2

        self.attn = nn.MultiheadAttention(style_dim, num_heads=1)
        self.query = ConstantInput([self.n_blocks * 2, style_dim])

        self.blocks = nn.ModuleList([
            DecoderBlock(channel, 16 * 64, is_first_block=i == 0) for i in range(self.n_blocks)
        ])

        self.up = nn.Sequential(
            nn.ConvTranspose1d(channel, channel, kernel_size=2, stride=2),
            nn.ELU(),

            nn.ConvTranspose1d(channel, channel, kernel_size=2, stride=2),
            nn.ELU(),
        )

        self.out_conv = nn.Conv1d(channel, out_channel, kernel_size=1)

    def forward(self, input):
        #query = self.query(input).permute(1, 0, 2)
        #key_value = input.permute(1, 0, 2)
        #styles = self.attn(query=query, key=key_value, value=key_value)

        #print(input.shape)
        styles = input.reshape(input.size(0), -1)

        #assert styles.size(0) == self.n_blocks

        #styles = styles.view(-1, 2, styles.size(1), styles.size(2))

        # out = input
        # for block, (style1, style2) in zip(self.blocks, styles):
        #     out = block(out, style1, style2)

        out = input
        for block in self.blocks:
            out = block(out, styles)

        out = self.up(out)
        out = self.out_conv(out)

        #print(out)

        return out


class TextCNNAdaIN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 channel: int,
                 n_fold: int,
                 tau: float,
                 pad_idx: Union[None, int],
                 input_noise=0.0,
                 embed_dropout=0.1,
                 num_vq_embeds: int = 512,
                 vq_embeds_dim: int = None,
                 vq_loss_alpha=1.0,
                 d_slice=1,
                 ignore_quant=False):
        super().__init__()

        self.pad_idx = pad_idx
        self.tau = tau
        self.vq_loss_alpha = vq_loss_alpha
        self.d_slice = d_slice
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

        self.embed = nn.Embedding(vocab_size, channel, padding_idx=pad_idx)
        self.quantize = SlicedQuantize(d_slice, dim=vq_embeds_dim, n_embed=num_vq_embeds, decay=0.99)

        #self.embeds_to_encode = nn.Conv1d(input_embed_dim, channel, kernel_size=3, padding=1)
        self.encode_to_quants = nn.Conv1d(channel, vq_embeds_dim, kernel_size=1)
        #self.quants_to_decode = nn.Conv1d(vq_embeds_dim, channel, kernel_size=1)

        #self.encoder = Encoder(channel, 64, 2)
        self.encoder = nn.Sequential(*[EncoderBlock(channel) for i in range(n_fold)])
        self.decoder = Decoder(channel, vocab_size, vq_embeds_dim, n_blocks=int(np.log2(256) - np.log2(16) + 1))

        self.nll = nn.NLLLoss(reduction='none', ignore_index=self.pad_idx)

    def get_vq_blend(self):
        blend = self.blend_step / self.blend_steps
        return 0
        return (np.sin(np.pi * blend - np.pi * 0.5) + 1.0) / 2.0 if blend < 1.0 else 1.0

    def encode(self, x):
        # x ∈ [B, S]
        #x = self.input_noise(x)
        x = self.embed(x)                # x ∈ [B, S, C]
        #x = self.embed_dropout(x)

        x = x.permute(0, 2, 1)           # x ∈ [B, C, S]
        x = self.encoder(x)              # x ∈ [B, C, S / 2^k]
        x = self.encode_to_quants(x)     # x ∈ [B, Q, S / 2^k]
        x = x.permute(0, 2, 1)           # x ∈ [B, S / 2^k, Q]

        z, diff, ids = self.quantize(x)  # z ∈ [B, S / 2^k, Q]

        blend = self.get_vq_blend()
        z = z * blend + x * (1 - blend)
        z += torch.rand_like(z) * 1.0

        if self.ignore_quant:
            z = x
            diff = diff.detach()

        z = x

        return z, diff, ids

    def decode(self, z):
        x = self.decoder(z)
        #x = self.quants_to_decode(x)    # x ∈ [B, C, S / 2^k]
        x = x.permute(0, 2, 1)          # x ∈ [B, S, C]

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
    decoder = Decoder(128, 64, 7)
    styles = torch.rand(32, 256, 64)

    res = decoder(styles)

    pass