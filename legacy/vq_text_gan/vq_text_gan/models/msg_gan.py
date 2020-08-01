import torch
import torch.nn as nn

from vq_text_gan.models.progan import GeneratorBlock, DiscriminatorBlock, GeneratorBlockResidual, DiscriminatorBlockResidual, MinibatchStdDev, activation, conv


class Generator(nn.Module):
    def __init__(self, latent_size, block_channels, extract_dims, extract_at_indices=None, attn=False):
        super().__init__()

        if not extract_at_indices:
            extract_at_indices = list(range(len(block_channels)))

        if not isinstance(extract_dims, (list, tuple)):
            extract_dims = [extract_dims] * len(block_channels)

        if not isinstance(attn, (list, tuple)):
            attn = [attn] * len(block_channels)

        channels = [latent_size] + block_channels

        self.blocks = nn.ModuleList()
        self.extract_layers = nn.ModuleList()

        for i, (in_channels, out_channels, extract_dim, _attn) in enumerate(zip(channels[:-1], channels[1:], extract_dims, attn)):
            self.blocks.append(GeneratorBlock(in_channels, out_channels, is_initial=i == 0, attn=_attn))
            if i in extract_at_indices:
                self.extract_layers.append(conv(channels[i + 1], extract_dim, kernel_size=1))
            else:
                self.extract_layers.append(None)

    def forward(self, input):
        x = input
        outputs = []
        for block, extract_layer in zip(self.blocks, self.extract_layers):
            x = block(x)
            if extract_layer is not None:
                outputs.append(extract_layer(x))
        return outputs


class Discriminator(nn.Module):
    def __init__(self, block_channels, inject_dims, inject_at_indices=None, attn=False):
        super().__init__()

        if not inject_at_indices:
            inject_at_indices = list(range(len(block_channels)))

        if not isinstance(inject_dims, (list, tuple)):
            inject_dims = [inject_dims] * len(block_channels)

        if not isinstance(attn, (list, tuple)):
            attn = [attn] * len(block_channels)

        self.inject_at_indices = inject_at_indices

        channels = block_channels + [block_channels[-1]]

        self.blocks = nn.ModuleList()
        self.inject_layers = nn.ModuleList()

        for i, (in_channels, out_channels, inject_dim, _attn) in enumerate(zip(channels[:-1], channels[1:], inject_dims, attn)):
            if i in inject_at_indices:
                self.inject_layers.append(conv(inject_dim, in_channels // 2, kernel_size=1))
                in_channels = (0 if not i else in_channels) + in_channels // 2
            else:
                self.inject_layers.append(None)

            self.blocks.append(DiscriminatorBlock(in_channels, out_channels, attn=_attn))

        self.final = nn.Sequential(
            MinibatchStdDev(),

            conv(channels[-1] + 1, 32, kernel_size=2),
            activation(),

            conv(32, 1, kernel_size=1)
        )

    def forward(self, inputs):
        assert len(inputs) == len(self.inject_at_indices)

        inputs = list(inputs)

        x = None
        for i, (block, inject_layer) in enumerate(zip(self.blocks, self.inject_layers)):
            if inject_layer is not None:
                input = inject_layer(inputs.pop(0))
                if x is not None:
                    input = torch.cat([x, input], dim=1)
                x = input
            else:
                assert x is not None

            x = block(x)

        assert len(inputs) == 0
        #assert x.size(1) == self.num_channels and x.size(2) == 1

        x = self.final(x)
        x = x.view(-1)

        return x


if __name__ == '__main__':
    num_blocks = 4

    G = Generator(128, [64] * num_blocks, 32)
    D = Discriminator([64] * num_blocks, 32)

    x = [torch.randn(32, 32, 2 ** (i + 2)) for i in range(num_blocks)]
    D(x[::-1])

    D(G(torch.randn(32, 128, 1))[::-1])