import torch

from torch import nn
from torch.nn import functional as F

from typing import Union

from vq_vae_text.modules import Quantize, CategoricalNoise, Attention, wn_conv1d, wn_conv_transpose1d


class ResBlock(nn.Module):
    def __init__(self, channel, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(channel, channel, kernel_size=1),

            nn.ELU(),
            nn.Conv1d(channel, channel, kernel_size=1),

            nn.ELU(),
            nn.Conv1d(channel, channel, kernel_size=1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, channel_dim, n_folds):
        super().__init__()

        self.res = ResBlock(channel_dim, channel_dim)
        self.attn = nn.MultiheadAttention(channel_dim, num_heads=8, dropout=0.1)
        self.norm = nn.LayerNorm(channel_dim)
        self.down = nn.Sequential(*[nn.Conv1d(channel_dim, channel_dim, kernel_size=2, stride=2)
                                    for _ in range(n_folds)])

    def forward(self, input):
        out = self.res(input)

        out = out.permute(2, 0, 1)
        out = self.norm(self.attn(out, out, out)[0] + out)

        out = out.permute(1, 2, 0)
        out = self.down(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, channel_dim):
        super().__init__()

        self.res = ResBlock(channel_dim, channel_dim)
        self.attn = nn.MultiheadAttention(channel_dim, num_heads=8, dropout=0.1)
        self.norm = nn.LayerNorm(channel_dim)
        self.up = nn.ConvTranspose1d(channel_dim, channel_dim, kernel_size=2, stride=2)

    def forward(self, input):
        out = self.res(input)

        out = out.permute(2, 0, 1)
        out = self.norm(self.attn(out, out, out)[0] + out)

        out = out.permute(1, 2, 0)
        out = self.up(out)

        return out


class Decoder(nn.Module):
    def __init__(self, channel_dim, n_fold):
        super().__init__()

        self.blocks = nn.Sequential(*[DecoderBlock(channel_dim) for _ in range(n_fold)])

    def forward(self, input):
        return self.blocks(input)


class TextCNNVAttnV2(nn.Module):
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
                 d_slice: int = 1,
                 **kwargs):
        super().__init__()

        self.pad_idx = pad_idx
        self.tau = tau
        self.vq_loss_alpha = vq_loss_alpha
        self.d_slice = d_slice

        self.vq_loss = 0
        self.nll_loss = 0
        self.acc = 0

        self.vq_embeds_dim = vq_embeds_dim

        self.input_noise = CategoricalNoise(vocab_size, input_noise)
        self.embed_dropout = nn.Dropout(embed_dropout)

        self.embed = nn.Embedding(vocab_size, channel, padding_idx=pad_idx)
        self.quantize = Quantize(vq_embeds_dim // self.d_slice, num_vq_embeds, decay=0.99)

        #self.embeds_to_encode = nn.Conv1d(input_embed_dim, channel, kernel_size=3, padding=1)
        self.encode_to_quants = nn.Conv1d(channel, vq_embeds_dim, kernel_size=3, padding=1)
        self.quants_to_decode = nn.Conv1d(vq_embeds_dim, channel, kernel_size=3, padding=1)

        self.encoder = Encoder(channel, n_fold)
        self.decoder = Decoder(channel, n_fold)

        self.nll = nn.NLLLoss(reduction='none', ignore_index=self.pad_idx)

    def encode(self, x):
        # x ∈ [B, S]
        x = self.input_noise(x)
        x = self.embed(x)                # x ∈ [B, S, C]
        x = self.embed_dropout(x)

        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.encode_to_quants(x)
        x = x.permute(0, 2, 1)

        shape = x.size()
        x = x.reshape(shape[0], shape[1] * self.d_slice, -1)
        z, diff, ids = self.quantize(x)  # z ∈ [B, S / 2^k, Q]

        return z, diff, ids

    def decode(self, z):
        shape = z.size()
        z = z.view(shape[0], shape[1] // self.d_slice, -1)
        x = z.permute(0, 2, 1)          # x ∈ [B, Q, S / 2^k]
        x = self.quants_to_decode(x)    # x ∈ [B, C, S / 2^k]
        x = self.decoder(x)             # x ∈ [B, C, S]
        x = x.permute(0, 2, 1)          # x ∈ [B, S, C]

        logits = x / self.tau
        logp_probs = F.log_softmax(logits, dim=-1)

        return logp_probs

    def decode_codes(self, code):
        quant = self.quantize.embed_code(code)
        quant = quant.view(quant.size(0), quant.size(1) // self.d_slice, -1)
        x = self.decode(quant)
        return x

    def forward(self, x):
        z, diff, ids = self.encode(x)
        logp = self.decode(z)
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
            'vq': self.vq_loss,
            'acc': self.acc
        }
