import torch
import torch.nn as nn

from text_gans.nets.base import conv, conv_transpose, activation, norm, SelfAttention, MinibatchStdDev


class ResBlock(nn.Module):
    def __init__(self, channels, bn=False):
        super().__init__()

        _norm = lambda *args, **kwargs: nn.Identity()
        if bn:
            _norm = norm

        self.blocks = nn.Sequential(
            # conv(channels, channels, kernel_size=3, padding=1),
            # # _norm(channels),
            # activation(),
            # _norm(channels),
            #
            # conv(channels, channels, kernel_size=3, padding=1),
            # # _norm(channels),
            # activation(),
            # _norm(channels)

            _norm(channels),
            activation(),
            conv(channels, channels, kernel_size=3, padding=1),

            _norm(channels),
            activation(),
            conv(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, input):
        return input + self.blocks(input)


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_initial=False):
        super().__init__()

        blocks = []
        if is_initial:
            blocks.append(conv_transpose(in_channels, out_channels, kernel_size=4))
        else:
            blocks.append(conv_transpose(in_channels, out_channels, kernel_size=4, stride=2, padding=1))

        blocks += [
            activation(),
            norm(out_channels),

            ResBlock(out_channels, bn=True),
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        out = self.blocks(input)
        return out


class Generator(nn.Module):
    def __init__(self, latent_size, block_channels, out_dim, attn_at=None):
        super().__init__()

        #channels = [latent_size] + block_channels
        channels = [32] * len(block_channels)

        if attn_at is not None:
            self.attn = SelfAttention(channels[attn_at])
            self.attn_at = attn_at
        else:
            self.attn = None
            self.attn_at = None

        self.linear = nn.Linear(latent_size, 4 * 32)

        self.blocks = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            self.blocks.append(GeneratorBlock(in_channels, out_channels))

        self.final = conv(channels[-1], out_dim, kernel_size=1)

    def forward(self, input):
        out = self.linear(input.squeeze())
        out = out.reshape(-1, 32, 4)

        for i, block in enumerate(self.blocks):
            if self.attn_at == i:
                out = self.attn(out)
            out = block(out)

        return self.final(out)


class UpBlock(nn.Module):
    def __init__(self, x1_channels, x2_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            activation(),
            conv_transpose(x1_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )
        self.merge = nn.Sequential(
            activation(),
            conv(out_channels + x2_channels, out_channels, kernel_size=1),
        )
        self.res_block = nn.Sequential(
            ResBlock(out_channels),
            # ResBlock(out_channels)
        )

    def forward(self, x1, x2):
        up = self.up(x1)
        cat = torch.cat([up, x2], dim=1)
        merge = self.merge(cat)
        out = self.res_block(merge)
        return out


class UNetDiscriminator(nn.Module):
    def __init__(self, base_channel, max_channel, depth, in_dim, attn_at=2):
        super().__init__()

        self.num_classes = in_dim

        self.encoder_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        self.up_blocks = nn.ModuleList()

        self.first = conv(in_dim, base_channel, kernel_size=1)
        self.final = conv(base_channel, 1, kernel_size=1)

        def ch_for_depth(d):
            return min(max_channel, base_channel * 2 ** d)

        if attn_at is not None:
            self.attn = SelfAttention(ch_for_depth(attn_at))
            self.attn_at = attn_at
        else:
            self.attn = None
            self.attn_at = None

        final_ch = ch_for_depth(depth - 1)

        for i in range(depth):
            ch_in = ch_for_depth(i)
            ch_out = ch_for_depth(i + 1)

            self.encoder_blocks.append(nn.Sequential(
                ResBlock(ch_in)
            ))
            if i + 1 != depth:
                self.down_blocks.append(nn.Sequential(
                    conv(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                    activation(),
                ))

        for i in range(depth)[:0:-1]:
            self.up_blocks.append(UpBlock(ch_for_depth(i), ch_for_depth(i - 1), ch_for_depth(i - 1)))

        self.minibatch_std_dev = MinibatchStdDev()
        # self.conv = conv(final_ch + 1, final_ch, kernel_size=1)

        self.activation = activation()
        self.linear = nn.Linear(final_ch + 1, 1)

    def forward(self, input):
        out = self.first(input)

        outputs = []

        for i, encoder_block in enumerate(self.encoder_blocks):
            out = encoder_block(out)

            if self.attn_at == i:
                out = self.attn(out)

            if i + 1 != len(self.encoder_blocks):
                outputs.append(out)
                out = self.down_blocks[i](out)

        out = self.activation(out)
        global_out = self.minibatch_std_dev(out)
        # out = self.conv(out)
        # out = self.activation(out)

        global_out = self.linear(
            global_out.sum([-1])
        ).squeeze()

        outputs = outputs[::-1]

        for skip, up_block in zip(outputs, self.up_blocks):
            out = up_block(out, skip)

        local_out = self.final(out).squeeze()

        return global_out, local_out
