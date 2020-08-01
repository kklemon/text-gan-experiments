import torch
import torch.nn as nn


def activation():
    return nn.ReLU()


def conv(*args, **kwargs):
    return nn.Conv1d(*args, **kwargs)


def conv_transpose(*args, **kwargs):
    return nn.ConvTranspose1d(*args, **kwargs)


def norm(*args, **kwargs):
    return nn.BatchNorm1d(*args, **kwargs)


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.chanel_in = in_dim

        self.query_conv = conv(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = conv(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = conv(in_dim, in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.tensor(0.05), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
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


class MinibatchStdDev(nn.Module):
    def forward(self, x, alpha=1e-8):
        batch_size, _, width = x.shape

        y = x - x.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)
        y = y.mean().view(1, 1, 1)
        y = y.repeat(batch_size, 1, width)
        y = torch.cat([x, y], 1)

        return y


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attn=False, is_initial=False):
        super().__init__()

        blocks = []
        if is_initial:
            blocks.append(conv_transpose(in_channels, out_channels, kernel_size=4))
        else:
            blocks.append(conv_transpose(in_channels, out_channels, kernel_size=4, stride=2, padding=1))

        blocks += [
            activation(),
            nn.BatchNorm1d(out_channels),

            conv(out_channels, out_channels, kernel_size=3, padding=1),
            activation(),
            nn.BatchNorm1d(out_channels),

            conv(out_channels, out_channels, kernel_size=3, padding=1),
            activation(),
            nn.BatchNorm1d(out_channels),
        ]

        self.blocks = nn.Sequential(*blocks)
        if attn:
            self.attn = SelfAttention(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, input):
        out = self.blocks(input)
        out = self.attn(out)
        return out


class GeneratorBlockResidual(nn.Module):
    def __init__(self, in_channels, out_channels, attn=False, is_initial=False):
        super().__init__()

        if is_initial:
            self.first = nn.Sequential(
                conv_transpose(in_channels, out_channels, kernel_size=4),
                activation(),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.first = nn.Sequential(
                conv_transpose(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                activation(),
                nn.BatchNorm1d(out_channels),
            )

        self.res_block = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            activation(),
            conv(out_channels, out_channels, kernel_size=3, padding=1),

            nn.BatchNorm1d(out_channels),
            activation(),
            conv(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if attn:
            self.attn = SelfAttention(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, input):
        out = self.first(input)
        out = self.res_block(out) + out
        out = self.attn(out)
        return out


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attn=False, is_final=False):
        super().__init__()

        blocks = [
            conv(in_channels, out_channels, kernel_size=3, padding=1),
            activation(),

            conv(out_channels, out_channels, kernel_size=3, padding=1),
            activation(),
        ]

        self.blocks = nn.Sequential(*blocks)

        if attn:
            self.attn = SelfAttention(out_channels)
        else:
            self.attn = nn.Identity()

        if is_final:
            self.down = nn.Sequential(
                conv(out_channels, out_channels, kernel_size=4),
                activation(),
            )
        else:
            self.down = nn.Sequential(
                conv(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
                activation(),
            )

        self.activation = activation()

    def forward(self, input):
        out = self.blocks(input)
        out = self.attn(out)
        out = self.down(out)
        return out


class DiscriminatorBlockResidual(nn.Module):
    def __init__(self, in_channels, out_channels, attn=False, is_final=False):
        super().__init__()

        self.first = nn.Sequential(
            activation(),
            conv(in_channels, out_channels, kernel_size=1),
        )

        self.res_block = nn.Sequential(
            activation(),
            conv(out_channels, out_channels, kernel_size=3, padding=1),

            activation(),
            conv(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if attn:
            self.attn = SelfAttention(out_channels)
        else:
            self.attn = nn.Identity()

        if is_final:
            self.down = nn.Sequential(
                conv(out_channels, out_channels, kernel_size=4),
                activation(),
            )
        else:
            self.down = nn.Sequential(
                conv(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
                activation(),
            )

        self.activation = activation()

    def forward(self, input):
        out = self.first(input)
        out = self.res_block(out) + out
        out = self.attn(out)
        out = self.down(out)
        return out
