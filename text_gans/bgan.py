import torch
import torch.nn.functional as F

from bgan.losses import compute_norm_weights


def multinomial_bgan_loss_unet(D, fake_logits, reals, n_samples=8, tau=1.0):
    batch_size = reals.size(0)
    n_channels = fake_logits.size(1)
    spatial_dims = reals.shape[1:]

    fake_logits /= tau

    fake_p = torch.softmax(
        fake_logits.view(batch_size, n_channels, -1).transpose(1, 2),
        dim=-1
    ).view(1, -1, n_channels)

    samples = torch.multinomial(fake_p.repeat(n_samples, 1, 1).view(-1, n_channels), num_samples=1)
    samples = samples.view(n_samples, batch_size, *spatial_dims)
    samples_one_hot = F.one_hot(samples, num_classes=n_channels).type(torch.float)
    samples_one_hot = samples_one_hot.permute(0, 1, 3, 2)

    reals_one_hot = F.one_hot(reals, num_classes=n_channels).type(torch.float).transpose(1, 2)

    real_global_out, real_local_out = D(reals_one_hot)
    fake_global_out, fake_local_out = D(samples_one_hot.view(-1, *samples_one_hot.shape[2:]))

    log_w_local = fake_local_out.view(n_samples, batch_size, *spatial_dims)
    log_w_global = fake_global_out.view(n_samples, batch_size)

    log_g = -(samples_one_hot * (fake_logits - torch.logsumexp(fake_logits, dim=1, keepdim=True)).unsqueeze(0)).sum(2)

    w_tilde_local = compute_norm_weights(log_w_local).detach()
    w_tilde_global = compute_norm_weights(log_w_global).detach()

    d_loss = F.softplus(-real_local_out).mean() + F.softplus(-fake_local_out).mean() + fake_local_out.mean() + \
             F.softplus(-real_global_out).mean() + F.softplus(-fake_global_out).mean() + fake_global_out.mean()
    g_loss = (w_tilde_local * log_g).sum(0).mean() + (w_tilde_global * log_g.mean([-1])).sum(0).mean()

    p_fake = (fake_local_out < 0).type(torch.float).mean().detach()
    p_real = (real_local_out > 0).type(torch.float).mean().detach()

    return d_loss, g_loss, p_fake, p_real


def gradient_penalty(D, fake_logits, reals):
    batch_size = reals.size(0)

    fakes = fake_logits.softmax(1).detach()

    # generate random epsilon
    eps = torch.rand((batch_size, 1, 1, 1)).to(reals.device)

    # create the merge of both real and fake samples
    merged = eps * reals + (1 - eps) * fakes
    merged.requires_grad = True

    # forward pass
    op = D(merged)

    # perform backward pass from op to merged for obtaining the gradients
    gradient = torch.autograd.grad(outputs=op, inputs=merged,
                                   grad_outputs=torch.ones_like(op), create_graph=True,
                                   retain_graph=True, only_inputs=True)[0]

    # calculate the penalty using these gradients
    penalty = 6 * (gradient.reshape(gradient.size(0), -1).norm(p=2, dim=1) ** 2).mean().unsqueeze(0)

    # return the calculated penalty:
    return penalty
