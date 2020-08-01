import torch
import math

from torch import nn
from torch.nn import functional as F

from typing import Union

from vq_vae_text.modules import Quantize, CategoricalNoise, Attention, ChannelWiseLayerNorm, wn_conv1d, wn_conv_transpose1d
from vq_vae_text.models.transformer import PositionalEncoding


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(in_channel, channel, kernel_size=kernel_size, padding=padding, dilation=dilation),

            nn.ELU(),
            nn.Conv1d(channel, in_channel, kernel_size=1),
        )

        self.norm = ChannelWiseLayerNorm(in_channel)

    def forward(self, input):
        return self.norm(self.conv(input) + input)


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, res_channel, n_res_blocks=2, down=2, attn_heads=1, attn_feedfoward=256):
        super().__init__()

        blocks = [
            nn.Conv1d(in_channel, channel, kernel_size=1),
            nn.ELU()
        ]

        for _ in range(down):
            for _ in range(n_res_blocks):
                blocks.append(ResBlock(channel, res_channel))

            blocks += [
                nn.Conv1d(channel, channel, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
            ]

        self.blocks = nn.Sequential(*blocks)
        self.attn = nn.TransformerEncoderLayer(channel, attn_heads, attn_feedfoward)

    def forward(self, input):
        out = skip = self.blocks(input)
        out = out.permute(2, 0, 1)
        out = self.attn(out).permute(1, 2, 0)
        return out + skip


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, res_channel, n_res_blocks=2):
        super().__init__()

        blocks = [
            nn.Conv1d(in_channel, channel, kernel_size=1),
            nn.ELU()
        ]

        for i in range(2):
            blocks += [
                nn.ConvTranspose1d(channel, channel, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
            ]

            for j in range(n_res_blocks):
                blocks.append(ResBlock(channel, res_channel, kernel_size=3, padding=1))

        blocks.append(nn.Conv1d(channel, out_channel, kernel_size=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, dim_feedforward, num_layers, dropout=0.1):
        super().__init__()
        self.trg_mask = None
        self.pos_encoder = PositionalEncoding(embed_dim)

        self.decoder_layer = nn.TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward)
        self.transformer = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.trg_embed = nn.Embedding(vocab_size, embed_dim)

        self.feature_dim = embed_dim
        self.decoder = nn.Linear(embed_dim, vocab_size)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, trg, mem):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            device = trg.device
            mask = self.generate_square_subsequent_mask(len(trg)).to(device)
            self.trg_mask = mask

        trg = self.trg_embed(trg) * math.sqrt(self.feature_dim)
        trg = self.pos_encoder(trg)

        output = self.transformer(trg, mem, tgt_mask=self.trg_mask)
        output = self.decoder(output)
        return output


class TextCNNV2Attn(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 channel: int,
                 res_channel: int,
                 n_res_block: int,
                 tau: float,
                 pad_idx: Union[None, int],
                 eos_idx: int,
                 input_noise=0.0,
                 embed_dropout=0.0,
                 num_vq_embeds: int = 512,
                 vq_embed_dim: int = 64,
                 vq_loss_alpha=1.0,
                 d_slice=1,
                 ignore_quant=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx
        self.eox_idx = eos_idx
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
        self.pos_encoder = PositionalEncoding(embed_dim)

        self.enc1 = Encoder(embed_dim, channel, res_channel, n_res_block, down=2)
        self.enc2 = Encoder(channel, channel, res_channel, n_res_block, down=2)
        self.enc3 = Encoder(channel, channel, res_channel, n_res_block, down=1)

        self.quant_conv_1 = nn.Conv1d(channel, vq_embed_dim, kernel_size=1)
        self.quant_conv_2 = nn.Conv1d(channel, vq_embed_dim, kernel_size=1)
        self.quant_conv_3 = nn.Conv1d(channel, vq_embed_dim, kernel_size=1)

        self.quant_1 = Quantize(vq_embed_dim, num_vq_embeds)
        self.quant_2 = Quantize(vq_embed_dim, num_vq_embeds)
        self.quant_3 = Quantize(vq_embed_dim, num_vq_embeds)

        self.conv_quant_1 = nn.Conv1d(vq_embed_dim, 256, kernel_size=1)
        self.conv_quant_2 = nn.Conv1d(vq_embed_dim, 256, kernel_size=1)
        self.conv_quant_3 = nn.Conv1d(vq_embed_dim, 256, kernel_size=1)

        self.decoder = TransformerDecoder(vocab_size + 1, 256, 4, 1024, 6)

        #self.noise = Noise(alpha=0.05)

        self.nll = nn.NLLLoss(reduction='none', ignore_index=self.pad_idx)

    def encode(self, x):
        # x ∈ [B, S]
        x = self.input_noise(x)
        x = self.embed(x) * math.sqrt(self.embed_dim)
        x = self.embed_dropout(x)
        x = self.pos_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

        x = x.permute(0, 2, 1)

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        quant1 = self.quant_conv_1(enc1).permute(0, 2, 1)
        quant2 = self.quant_conv_2(enc2).permute(0, 2, 1)
        quant3 = self.quant_conv_3(enc3).permute(0, 2, 1)

        quant1, diff1, id1 = self.quant_1(quant1)
        quant2, diff2, id2 = self.quant_2(quant2)
        quant3, diff3, id3 = self.quant_3(quant3)

        return (quant1, quant2, quant3), diff1 + diff2 + diff3, (id1, id2, id3)

    def decode(self, z, x=None, max_length=256):
        z1 = self.conv_quant_1(z[0].transpose(1, 2))
        z2 = self.conv_quant_2(z[1].transpose(1, 2))
        z3 = self.conv_quant_3(z[2].transpose(1, 2))

        mem = torch.cat([z1, z2, z3], dim=2).permute(2, 0, 1)

        # FIXME: adjust length properly
        trg = torch.full_like(x, self.pad_idx, dtype=x.dtype, device=x.device)

        trg[:, 0] = self.vocab_size
        trg[:, 1:] = x[:, :-1]

        logits = self.decoder(trg.T, mem).permute(1, 0, 2)
        logits = logits / self.tau

        logp_probs = F.log_softmax(logits, dim=-1)

        return logp_probs

    def decode_codes(self, code):
        code = code.view(code.size(0), self.d_slice, -1)
        quant = self.quantize.embed_code(code)
        quant = quant.view(quant.size(0), quant.size(1) // self.d_slice, -1)
        x = self.decode(quant)
        return x

    def forward(self, x):
        z, diff, ids = self.encode(x)
        logp = self.decode(z, x)
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
