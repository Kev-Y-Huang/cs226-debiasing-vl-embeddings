import torch

import torch.nn as nn


class SimpleGenerator(nn.Module):
    """
    Simple generator model according to Zhang et al. (2018).
    Given the first three words in a gender-based analogy problem x_1, x_2, and x_3,
    the generator predicts the fourth word x_4.
    """
    def __init__(self, embed_dim):
        init = torch.randn([embed_dim, 1])
        unit_init = init / (torch.norm(init))
        self.w = torch.nn.Parameter(unit_init, requires_grad=True)

    def forward(self, x):
        v = x[:, 1, :] + x[:, 2, :] - x[:, 0, :]
        y_hat = v - torch.matmul(torch.matmul(v, self.w), self.w.T)
        return y_hat


class SimpleDiscriminator(nn.Module):
    """
    Simple discriminator model according to Zhang et al. (2018).
    Given a word embedding z, the discriminator predicts the value of the protected attribute.
    """
    def __init__(self, embed_dim):
        super(SimpleDiscriminator, self).__init__()
        init = torch.randn([embed_dim, 1])
        unit_init = init / (torch.norm(init))
        self.w = torch.nn.Parameter(unit_init, requires_grad=True)

    def forward(self, x):
        z_hat = torch.matmul(x, self.w)
        return z_hat

# Define the GAN
class SimpleGAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(SimpleGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        z = self.generator(x)
        y = self.discriminator(z)
        return z, y