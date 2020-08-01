import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


def wn_conv1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))


def wn_conv_transpose1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def wn_linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


class BaseQuantize(nn.Module):
    def embed_code(self, embed_id):
        raise NotImplementedError


class Quantize(BaseQuantize):
    def __init__(self, dim, n_embed, decay=0.999, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class DecomposedQuantize(BaseQuantize):
    def __init__(self, length, dim, n_embed, decay=0.999, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.quantizations = nn.ModuleList([Quantize(dim, n_embed, decay, eps) for _ in range(length)])

    def forward(self, input):
        out = torch.empty_like(input)
        diff = None
        ids = torch.empty(*input.shape[:-1], dtype=torch.long, device=input.device)
        for i in range(input.size(1)):
            quant, diff, code = self.quantizations[i](input[:, i])
            out[:, i] = quant
            ids[:, i] = code
            if diff is None:
                diff = diff
            else:
                diff += diff
        return out, diff / len(self.quantizations), ids

    def embed_code(self, embed_id):
        out = torch.empty(*embed_id.size(), self.dim, dtype=torch.float, device=embed_id.device)
        for i in range(embed_id.size(1)):
            out[:, i] = self.quantizations[i].embed_code(embed_id[:, i])
        return out


class SlicedQuantize(nn.Module):
    def __init__(self, d_slice, dim, **kwargs):
        super().__init__()

        self.dim = dim // d_slice
        self.quantize = Quantize(dim=self.dim, **kwargs)
        self.d_slice = d_slice

    def forward(self, input):
        shape = input.size()
        input = input.reshape(*input.shape[:-2], -1, self.dim)
        z, diff, ids = self.quantize(input)
        z = z.view(shape)
        return z, diff, ids


class CategoricalNoise(nn.Module):
    def __init__(self, n_classes, p):
        super().__init__()

        self.n_classes = n_classes
        self.p = p

    def forward(self, input):
        if self.training:
            mask = (torch.rand(input.shape, device=input.device) > self.p).type(input.dtype)
            noise = torch.randint_like(input, 0, self.n_classes)

            return input * mask + (1 - mask) * noise
        else:
            return input


class GeometricCategoricalDropout(nn.Module):
    def __init__(self, n, q, alpha):
        super().__init__()

        if not (0 < q < 1):
            raise ValueError('q must be a value 0 < ... < 1')

        self.a = 1 / (((q ** (n + 1) - 1) / (q - 1)) - 1)
        self.n = n
        self.q = q
        self.alpha = alpha
        #
        # self.probs = 1 - alpha * (1 - torch.full([n], self.a) * torch.pow(self.q, (torch.arange(n) + 1).type(torch.float)))
        # self.m = torch.distributions.Bernoulli(self.probs)

    def forward(self, input):
        assert input.max() <= self.n

        if self.training:
            probs = 1 - torch.pow(self.q, input.type(torch.float))
        else:
            probs = torch.zeros_like(input)

        mask = (torch.rand(input.shape, device=input.device) <= probs).type(input.dtype)
        return mask


class Noise(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        if not self.training:
            return input
        return input + self.alpha * torch.randn_like(input)


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(in_channel, channel, kernel_size=kernel_size, padding=padding, dilation=dilation),

            nn.ELU(),
            nn.Conv1d(channel, in_channel, kernel_size=1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class ChannelWiseLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, input):
        shape = input.size()
        input = input.view(shape[0], shape[1], -1).transpose(1, 2)
        input = self.ln(input)
        input = input.transpose(1, 2).view(shape)
        return input


class Attention(nn.Module):
    def __init__(self, in_dim, key_query_dim, value_dim, n_heads=1, tau=1.0):
        super().__init__()

        self.query_w = wn_linear(in_dim, key_query_dim)
        self.key_w = wn_linear(in_dim, key_query_dim)
        self.value_w = wn_linear(in_dim, value_dim)

        self.n_heads = n_heads
        self.kq_head_dim = key_query_dim // n_heads
        self.val_head_dim = value_dim // n_heads

        self.tau = tau

    def forward(self, query, key):
        bs, _, l = query.size()

        query_ = query.transpose(1, 2)
        key_ = key.transpose(1, 2)

        def reshape(x, head_dim):
            return x.view(bs, -1, self.n_heads, head_dim).transpose(1, 2)

        query = reshape(self.query_w(query_), self.kq_head_dim)
        key = reshape(self.key_w(key_), self.kq_head_dim).transpose(2, 3)
        value = reshape(self.value_w(key_), self.val_head_dim)

        attn = (query @ key) / sqrt(self.kq_head_dim)
        attn = attn / self.tau
        attn = F.softmax(attn, dim=-1)

        out = attn @ value
        out = out.transpose(1, 2).reshape(
            bs, l, self.n_heads * self.val_head_dim
        )
        out = out.permute(0, 2, 1)

        return out



class EqualizedConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_out, c_in, *nn.modules.utils._single(kernel_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = padding

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = np.prod(nn.modules.utils._single(kernel_size)) * c_in  # value of fan_in
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        return F.conv1d(input=x,
                        weight=self.weight * self.scale,  # scale the weight on runtime
                        bias=self.bias if self.use_bias else None,
                        stride=self.stride,
                        padding=self.pad)


class EqualizedConvTranspose1d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.nn.init.normal_(
            torch.empty(c_in, c_out, *nn.modules.utils._single(kernel_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = padding

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in  # value of fan_in for deconv
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        return F.conv_transpose1d(input=x,
                                  weight=self.weight * self.scale,  # scale the weight on runtime
                                  bias=self.bias if self.use_bias else None,
                                  stride=self.stride,
                                  padding=self.pad)


class EqualizedLinear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        """
        Linear layer modified for equalized learning rate
        """
        from numpy import sqrt

        super().__init__()

        self.weight = nn.Parameter(torch.nn.init.normal_(
            torch.empty(c_out, c_in)
        ))

        self.use_bias = bias

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale,
                        self.bias if self.use_bias else None)


if __name__ == '__main__':
    n = 4096

    do = GeometricCategoricalDropout(n, 0.998, 1.0)

    i = 10000
    res = torch.zeros(n)
    for j in range(i):
        sample = torch.randint(0, n, [128])
        out = do(sample)
        for k, m in enumerate(out):
            if m:
                res[sample[k]] += 1

    res /= i

    import numpy as np
    import matplotlib.pyplot as plt
    plt.plot(np.arange(n), res.numpy())
    plt.ylim(bottom=0)
    plt.show()