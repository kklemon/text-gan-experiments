import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from vq_text_gan.modules import MinibatchStdDev
from vq_text_gan.models.gan_base import GeneratorBlock, DiscriminatorBlock, GeneratorBlockResidual, DiscriminatorBlockResidual, conv, activation


class Generator(nn.Module):
    def __init__(self, latent_size, block_channels, extract_dim, attn=False):
        super().__init__()

        channels = [latent_size] + block_channels

        self.blocks = nn.ModuleList()
        self.extract_layers = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            self.blocks.append(GeneratorBlockResidual(in_channels, out_channels, is_initial=i == 0, attn=attn))
            self.extract_layers.append(conv(out_channels, extract_dim, kernel_size=1))

    def forward(self, input, extract_at_grow_index):
        out = input
        for i, (block, extract_layer) in enumerate(zip(self.blocks, self.extract_layers)):
            out = block(out)
            if extract_at_grow_index == i:
                return extract_layer(out)
        raise ValueError(f'extract_at_grow_index must be < than the number of blocks ({len(self.blocks)})')


class InjectLayer(nn.Module):
    def __init__(self, num_clases, out_channels, use_embedding=False, embedding_dropout=0.3):
        super().__init__()

        self.num_classes = num_clases
        self.out_channels = out_channels

        if use_embedding:
            self.embed = nn.Sequential(
                nn.Embedding(num_clases, out_channels, max_norm=1.0),
                nn.Dropout(embedding_dropout)
            )
            self.conv = conv(out_channels, out_channels, kernel_size=1)
        else:
            self.embed = None
            self.conv = conv(num_clases, out_channels, kernel_size=1)

        self.activation = activation()

    def forward(self, input):
        if self.embed is not None:
            embed = self.embed(input)
            embed = embed.transpose(1, 2)
            embed = self.conv(embed)
        else:
            embed = F.one_hot(input, num_classes=self.num_classes).type(torch.float)
            embed = embed.transpose(1, 2)
            embed = self.conv(embed)
        embed = self.activation(embed)
        return embed


# class InjectLayer(nn.Module):
#     def __init__(self, num_clases, out_channels, use_embedding=False, embedding_dropout=0.3):
#         super().__init__()
#
#         self.num_classes = num_clases
#         self.out_channels = out_channels
#
#         self.conv = conv(num_clases, out_channels, kernel_size=1)
#
#         self.activation = activation()
#
#     def forward(self, input):
#         return self.activation(self.conv(input))


class Discriminator(nn.Module):
    def __init__(self, block_channels, num_input_classes, use_minibatch_std_dev=True, use_embeddings=False, attn=False):
        super().__init__()

        channels = block_channels + [block_channels[-1]]

        num_blocks = len(block_channels)

        self.blocks = nn.ModuleList()
        self.inject_layers = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            self.inject_layers.append(InjectLayer(num_input_classes, in_channels, use_embedding=use_embeddings))

            if i + 1 == num_blocks:
                self.blocks.append(nn.Sequential(
                    MinibatchStdDev(),
                    DiscriminatorBlockResidual(in_channels + 1, out_channels, is_final=i + 1 == len(block_channels), attn=attn)
                ))
            else:
                self.blocks.append(
                    DiscriminatorBlockResidual(in_channels, out_channels, is_final=i + 1 == len(block_channels), attn=attn)
                )

        self.final = conv(channels[-1], 1, kernel_size=1)

    def forward(self, input, inject_at_grow_index=None):
        if inject_at_grow_index is None:
            inject_at_grow_index = int(math.log2(input.size(1))) - 2

        real_inject_index = len(self.blocks) - inject_at_grow_index - 1
        out = self.inject_layers[real_inject_index](input)
        for block in self.blocks[real_inject_index:]:
            out = block(out)

        out = self.final(out)
        out = out.view(-1)
        return out
