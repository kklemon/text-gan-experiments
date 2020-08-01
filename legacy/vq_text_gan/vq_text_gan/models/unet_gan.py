import torch
import torch.nn as nn
import torch.nn.functional as F

from vq_text_gan.modules import MinibatchStdDev


def norm(*args, **kwargs):
    return nn.BatchNorm1d(*args, **kwargs)


def conv(*args, **kwargs):
    return nn.Conv2d(*args, **kwargs)


def conv_transpose(*args, **kwargs):
    return nn.ConvTranspose2d(*args, **kwargs)


def activation():
    return nn.LeakyReLU(0.2)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False, sn=False):
        super().__init__()

        _norm = lambda *args, **kwargs: nn.Identity()
        if bn:
            _norm = norm

        self.skip = conv(in_channels, out_channels, kernel_size=1)

        # self.res_block = nn.Sequential(
        #     _norm(in_channels),
        #     activation(),
        #     conv(in_channels, out_channels, kernel_size=3, padding=1),
        #
        #     _norm(out_channels),
        #     activation(),
        #     conv(out_channels, out_channels, kernel_size=3, padding=1),
        # )

        self.blocks = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=5, padding=2),
            activation(),
            _norm(out_channels),

            conv(out_channels, out_channels, kernel_size=5, padding=2),
            activation(),
            _norm(out_channels)
        )

    def forward(self, input):
        return self.blocks(input)

class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super().__init__()

        self.chanel_in = in_dim

        self.query_conv = conv(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = conv(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = conv(in_dim, in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.tensor(0.05), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width)

        out = self.gamma * out + x
        return out


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_initial=False):
        super().__init__()

        blocks = []
        if is_initial:
            blocks.append(conv_transpose(in_channels, out_channels, kernel_size=4))
        else:
            blocks.append(conv_transpose(in_channels, out_channels, kernel_size=4, stride=2, padding=1))

        blocks += [
            ResBlock(out_channels, out_channels, bn=True),

            norm(out_channels),
            activation(),
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        out = self.blocks(input)
        return out


class Generator(nn.Module):
    def __init__(self, latent_size, block_channels, out_dim, attn_at=None):
        super().__init__()

        channels = [latent_size] + block_channels

        if attn_at is not None:
            self.attn = SelfAttention(channels[attn_at + 1])
            self.attn_at = attn_at
        else:
            self.attn_at = None

        self.blocks = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            self.blocks.append(GeneratorBlock(in_channels, out_channels, is_initial=i == 0))

        self.final = conv(channels[-1], out_dim, kernel_size=1)

    def forward(self, input):
        out = input

        for i, block in enumerate(self.blocks):
            if self.attn_at == i:
                out = self.attn(out)
            out = block(out)

        return self.final(out)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = conv_transpose(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.res_block = ResBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        up = self.up(x1)
        merge = torch.cat([up, x2], dim=1)
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

        self.attn_at = attn_at
        self.attn = SelfAttention(ch_for_depth(attn_at))

        final_ch = ch_for_depth(depth - 1)

        for i in range(depth):
            ch_in = ch_for_depth(i)
            ch_out = ch_for_depth(i + 1)

            self.encoder_blocks.append(ResBlock(ch_in, activation=activation()))
            if i + 1 != depth:
                self.down_blocks.append(conv(ch_in, ch_out, kernel_size=4, stride=2, padding=1))

        for i in range(depth)[:0:-1]:
            self.up_blocks.append(UpBlock(ch_for_depth(i), ch_for_depth(i - 1)))

        self.minibatch_std_dev = MinibatchStdDev()
        self.conv = conv(final_ch + 1, final_ch, kernel_size=1)

        self.activation = activation()
        self.linear = nn.Linear(final_ch, 1)

    def forward(self, input):
        out = self.first(input)

        outputs = []

        for i, encoder_block in enumerate(self.encoder_blocks):
            out = encoder_block(out)

            if i + 1 != len(self.encoder_blocks):
                outputs.append(out)
                out = self.down_blocks[i](out)

        out = self.minibatch_std_dev(out)
        out = self.conv(out)

        global_out = self.linear(
            self.activation(out).sum([-1, -2])
        ).squeeze()

        outputs = outputs[::-1]

        for skip, up_block in zip(outputs, self.up_blocks):
            out = up_block(out, skip)

        pixel_out = self.final(out).squeeze()

        return global_out, pixel_out

        # for i, (decoder_block, up, merge, skip) in enumerate(zip(self.decoder_blocks, inter_outs[::-1])):
        #     skip = up(skip)
        #     cat = merge(torch.cat([out, skip], dim=1))
        #     out = decoder_block(cat)

