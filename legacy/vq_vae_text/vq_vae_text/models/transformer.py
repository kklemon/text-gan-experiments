import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext

from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, Transformer
from torchtext.data.utils import get_tokenizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        self.encoder_layers = TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_layers)

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.feature_dim = embed_dim
        self.decoder = nn.Linear(embed_dim, vocab_size)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self.generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.embed(src) * math.sqrt(self.feature_dim)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super().__init__()
        self.trg_mask = None
        self.pos_encoder = PositionalEncoding(embed_dim)

        self.transformer = Transformer(embed_dim, num_heads, num_encoder_layers, num_decoder_layers, hidden_dim, dropout=dropout)

        self.src_embed = nn.Embedding(vocab_size, embed_dim)
        self.trg_embed = nn.Embedding(vocab_size, embed_dim)

        self.feature_dim = embed_dim
        self.decoder = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            device = trg.device
            mask = self.transformer.generate_square_subsequent_mask(len(trg)).to(device)
            self.trg_mask = mask

        src = self.src_embed(src) * math.sqrt(self.feature_dim)
        src = self.pos_encoder(src)

        trg = self.trg_embed(trg) * math.sqrt(self.feature_dim)
        trg = self.pos_encoder(trg)

        output = self.transformer(src, trg, tgt_mask=self.trg_mask)
        output = self.decoder(output)
        return output
