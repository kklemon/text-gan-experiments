import torch
import torch.nn.functional as F


class BaseGANLoss:
    def loss_d(self, reals, fakes):
        raise NotImplementedError

    def loss_g(self, reals, fakes):
        raise NotImplementedError


class GANLoss(BaseGANLoss):
    def __init__(self, D):
        self.D = D

    def loss_d(self, reals, fakes):
        real_global_scores, real_pixel_scores = self.D(reals)
        fake_global_scores, fake_pixel_scores = self.D(fakes)

        real_loss_g = F.binary_cross_entropy_with_logits(real_global_scores, torch.full_like(real_global_scores, 1))
        fake_loss_g = F.binary_cross_entropy_with_logits(fake_global_scores, torch.full_like(fake_global_scores, 0))

        real_loss_l = F.binary_cross_entropy_with_logits(real_pixel_scores, torch.full_like(real_pixel_scores, 1))
        fake_loss_l = F.binary_cross_entropy_with_logits(fake_pixel_scores, torch.full_like(fake_pixel_scores, 0))

        real_loss = real_loss_g + real_loss_l
        fake_loss = fake_loss_g + fake_loss_l

        loss = (real_loss + fake_loss) / 2

        # p_real_g = (real_global_scores >= 0.0).type(torch.float).mean()
        # p_real_l = (real_pixel_scores >= 0.0).type(torch.float).mean()
        #
        # p_fake_g = (fake_global_scores <= 0.0).type(torch.float).mean()
        # p_fake_l = (fake_pixel_scores <= 0.0).type(torch.float).mean()

        return loss

    def loss_g(self, reals, fakes):
        fake_global_scores, fake_pixel_scores = self.D(fakes)

        fake_loss_g = F.binary_cross_entropy_with_logits(fake_global_scores, torch.full_like(fake_global_scores, 1.0))
        fake_loss_l = F.binary_cross_entropy_with_logits(fake_pixel_scores, torch.full_like(fake_pixel_scores, 1.0))

        return fake_loss_l + fake_loss_g


class RelativisticAverageHingeLoss:
    def __init__(self, D):
        self.D = D

    def loss_d(self, reals, fakes):
        real_global_scores, real_pixel_scores = self.D(reals)
        fake_global_scores, fake_pixel_scores = self.D(fakes)

        rf_diff_g = real_global_scores - torch.mean(fake_global_scores)
        fr_diff_g = fake_global_scores - torch.mean(real_global_scores)

        rf_diff_l = real_pixel_scores - torch.mean(fake_pixel_scores, dim=0)
        fr_diff_l = fake_pixel_scores - torch.mean(real_pixel_scores, dim=0)

        loss_g = F.relu(1 - rf_diff_g).mean() + F.relu(1 + fr_diff_g).mean()
        loss_l = F.relu(1 - rf_diff_l).mean() + F.relu(1 + fr_diff_l).mean()

        loss = loss_g + loss_l

        # p_real_g = (real_global_scores >= 0.5).type(torch.float).mean()
        # p_real_l = (real_pixel_scores >= 0.5).type(torch.float).mean()
        #
        # p_fake_g = (fake_global_scores <= 0.5).type(torch.float).mean()
        # p_fake_l = (fake_pixel_scores <= 0.5).type(torch.float).mean()

        return loss

    def loss_g(self, reals, fakes):
        real_global_scores, real_pixel_scores = self.D(reals)
        fake_global_scores, fake_pixel_scores = self.D(fakes)

        rf_diff_g = real_global_scores - torch.mean(fake_global_scores)
        fr_diff_g = fake_global_scores - torch.mean(real_global_scores)

        rf_diff_l = real_pixel_scores - torch.mean(fake_pixel_scores, dim=0)
        fr_diff_l = fake_pixel_scores - torch.mean(real_pixel_scores, dim=0)

        loss_g = F.relu(1 + rf_diff_g).mean() + F.relu(1 - fr_diff_g).mean()
        loss_l = F.relu(1 + rf_diff_l).mean() + F.relu(1 - fr_diff_l).mean()

        return loss_g + loss_l


class WGAN_GP(BaseGANLoss):
    def __init__(self, D, drift=0.001, use_gp=True, reg_lambda=10):
        self.D = D
        self.drift = drift
        self.use_gp = use_gp
        self.reg_lambda = reg_lambda

    def gradient_penalty(self, reals, fakes):
        batch_size = reals.size(0)

        reals = reals.detach()
        fakes = fakes.detach()

        # generate random epsilon
        eps = torch.rand((batch_size, 1, 1)).to(reals.device)

        # create the merge of both real and fake samples
        merged = eps * reals + (1 - eps) * fakes
        merged.requires_grad = True

        # forward pass
        global_score, local_scores = self.D(merged)
        local_score = local_scores.mean(-1)

        # perform backward pass from op to merged for obtaining the gradients
        gradients = torch.autograd.grad(outputs=[global_score, local_score], inputs=merged,
                                        grad_outputs=[torch.ones_like(global_score), torch.ones_like(local_score)],
                                        create_graph=True, retain_graph=True, only_inputs=True)

        # calculate the penalty using these gradients
        penalties = [(gradient.reshape(gradient.size(0), -1).norm(p=2, dim=1) ** 2).mean().unsqueeze(0)
                     for gradient in gradients]
        penalty = self.reg_lambda * torch.cat(penalties).mean()

        # return the calculated penalty:
        return penalty

    def loss_d(self, reals, fakes):
        real_global_scores, real_pixel_scores = self.D(reals)
        fake_global_scores, fake_pixel_scores = self.D(fakes)

        loss_global = fake_global_scores.mean() - real_global_scores.mean() + self.drift * (real_global_scores ** 2).mean()
        loss_local = fake_pixel_scores.mean() - real_pixel_scores.mean() + self.drift * (real_pixel_scores ** 2).mean()

        loss = loss_global + loss_local

        if self.use_gp and self.reg_lambda:
            loss += self.gradient_penalty(reals, fakes)

        # p_real_g = (real_global_scores >= 0.0).type(torch.float).mean()
        # p_real_l = (real_pixel_scores >= 0.0).type(torch.float).mean()
        #
        # p_fake_g = (fake_global_scores <= 0.0).type(torch.float).mean()
        # p_fake_l = (fake_pixel_scores <= 0.0).type(torch.float).mean()

        return loss

    def loss_g(self, reals, fakes):
        fake_global_scores, fake_pixel_scores = self.D(fakes)

        return -fake_global_scores.mean() - fake_pixel_scores.mean()


name_to_loss_class = {
    'gan_loss': GANLoss,
    'relativistic_average_hinge_loss': RelativisticAverageHingeLoss,
    'wgan_gp': WGAN_GP
}


def get_loss_function_by_name(D, name: str):
    return name_to_loss_class[name](D)
