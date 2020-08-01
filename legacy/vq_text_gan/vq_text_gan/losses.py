import torch
import torch.nn.functional as F


def compute_norm_weights(log_w):
    log_n = torch.log(torch.tensor(float(log_w.shape[0])))
    log_z_est = torch.logsumexp(log_w - log_n, dim=0)
    log_w_tilde = log_w - log_z_est - log_n

    w_tilde = torch.exp(log_w_tilde)

    return w_tilde


def binary_bgan_loss(D, fake_logits, reals, n_samples=8):
    batch_size, num_channels = fake_logits.shape[:2]
    spatial_dims = fake_logits.shape[2:]

    fake_logits = fake_logits.unsqueeze(0)
    samples = torch.rand(n_samples, batch_size, num_channels, *spatial_dims, device=fake_logits.device)
    samples = (samples <= torch.sigmoid(fake_logits)).type(torch.float)

    real_out = D(reals)
    fake_out = D(samples.view(-1, 1, *spatial_dims))

    log_w = fake_out.view(n_samples, batch_size)

    log_g = -((1.0 - samples) * fake_logits + F.softplus(-fake_logits)).mean(dim=(2, 3, 4))

    w_tilde = compute_norm_weights(log_w).detach()

    d_loss = F.softplus(-real_out).mean() + F.softplus(-fake_out).mean() + fake_out.mean()
    g_loss = -(w_tilde * log_g).sum(0).mean()

    p_fake = (fake_out < 0).type(torch.float).mean().detach()
    p_real = (real_out > 0).type(torch.float).mean().detach()

    return d_loss, g_loss, p_fake, p_real


def multinomial_bgan_loss(D, fake_logits, reals, n_samples=8, tau=1.0):
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

    real_out = D(reals)
    fake_out = D(samples.view(-1, *samples.shape[2:]))

    log_w = fake_out.view(n_samples, batch_size)

    log_g = -(samples_one_hot * (fake_logits - torch.logsumexp(fake_logits, dim=1, keepdim=True)).unsqueeze(0)).sum(dim=(2, 3))

    w_tilde = compute_norm_weights(log_w).detach()

    d_loss = F.softplus(-real_out).mean() + F.softplus(-fake_out).mean() + fake_out.mean()
    g_loss = (w_tilde * log_g).sum(0).mean()

    p_fake = (fake_out < 0).type(torch.float).mean().detach()
    p_real = (real_out > 0).type(torch.float).mean().detach()

    return d_loss, g_loss, p_fake, p_real


class WGAN_GP:
    def __init__(self, D, drift=0.001, use_gp=True, reg_lambda=10):
        self.D = D
        self.drift = drift
        self.use_gp = use_gp
        self.reg_lambda = reg_lambda

    def gradient_penalty(self, reals, fakes):
        is_list = True
        if not isinstance(reals, (list, tuple)):
            assert not isinstance(fakes, (list, tuple))
            reals = [reals]
            fakes = [fakes]
            is_list = False

        reals = [real.detach() for real in reals]
        fakes = [fake.detach() for fake in fakes]

        batch_size = reals[0].size(0)

        # generate random epsilon
        eps = torch.rand((batch_size, 1, 1)).to(reals[0].device)

        # create the merge of both real and fake samples
        merged = [eps * real + (1 - eps) * fake for real, fake in zip(reals, fakes)]
        for sample in merged:
            sample.requires_grad = True

        # forward pass
        op = self.D(merged if is_list else merged[0])

        # perform backward pass from op to merged for obtaining the gradients
        gradients = torch.autograd.grad(outputs=op, inputs=merged,
                                        grad_outputs=torch.ones_like(op), create_graph=True,
                                        retain_graph=True, only_inputs=True)

        # calculate the penalty using these gradients
        penalties = [(gradient.reshape(gradient.size(0), -1).norm(p=2, dim=1) ** 2).mean().unsqueeze(0)
                     for gradient in gradients]
        penalty = self.reg_lambda * torch.cat(penalties).mean()

        # return the calculated penalty:
        return penalty

    def d_loss(self, reals, fakes):
        fake_out = self.D(fakes)
        real_out = self.D(reals)

        loss = fake_out.mean() - real_out.mean() + self.drift * (real_out ** 2).mean()

        # rf_diff = real_out - torch.mean(fake_out)
        # fr_diff = fake_out - torch.mean(real_out)
        #
        # loss = F.relu(1 - rf_diff).mean() + F.relu(1 + fr_diff).mean()

        if self.use_gp and self.reg_lambda:
            loss += self.gradient_penalty(reals, fakes)

        p_fake = (fake_out < 0.0).type(torch.float).mean().detach()
        p_real = (real_out > 0.0).type(torch.float).mean().detach()

        return loss, p_fake, p_real

    def g_loss(self, reals, fakes):
        return -self.D(fakes).mean()

        # real_out = self.D(reals)
        # fake_out = self.D(fakes)
        #
        # rf_diff = real_out - torch.mean(fake_out)
        # fr_diff = fake_out - torch.mean(real_out)
        #
        # return F.relu(1 + rf_diff).mean() + F.relu(1 - fr_diff).mean()


class RelativisticAverageHingeLoss:
    def __init__(self, D):
        self.D = D

    def d_loss(self, reals, fakes):
        real_out = self.D(reals)
        fake_out = self.D(fakes)

        rf_diff = real_out - torch.mean(fake_out)
        fr_diff = fake_out - torch.mean(real_out)

        p_fake = (fake_out < 0.5).type(torch.float).mean().detach()
        p_real = (real_out > 0.5).type(torch.float).mean().detach()

        return F.relu(1 - rf_diff).mean() + F.relu(1 + fr_diff).mean(), p_fake, p_real

    def g_loss(self, reals, fakes):
        real_out = self.D(reals)
        fake_out = self.D(fakes)

        rf_diff = real_out - torch.mean(fake_out)
        fr_diff = fake_out - torch.mean(real_out)

        return F.relu(1 + rf_diff).mean() + F.relu(1 - fr_diff).mean()
