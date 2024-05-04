import torch
import torch.nn as nn


class SimplePredictor(nn.Module):
    """
    Simple predictor model according to Zhang et al. (2018).
    Given the first three words in a gender-based analogy problem x_1, x_2, and x_3,
    the predictor predicts the fourth word x_4.
    """

    def __init__(self, embed_dim):
        super(SimplePredictor, self).__init__()
        init = torch.randn([embed_dim, 1])
        unit_init = init / (torch.norm(init))
        self.w = torch.nn.Parameter(unit_init)

    def forward(self, x):
        v = x[:, 1, :] + x[:, 2, :] - x[:, 0, :]
        y_hat = v - torch.matmul(torch.matmul(v, self.w), self.w.T)
        return y_hat

    def predict(self, x):
        with torch.no_grad():
            y_hat = x - torch.matmul(torch.matmul(x, self.w), self.w.T)
            return y_hat

    def predict_batch(self, xs):
        with torch.no_grad():
            y_hats = []
            # Split the input into chunks of size 32 to avoid OOM errors
            for embeds in torch.split(xs, 32):
                y_hats.append(self.predict(embeds))
            return torch.cat(y_hats, axis=0)


class SimpleAdversary(nn.Module):
    """
    Simple adversary model according to Zhang et al. (2018).
    Given a word embedding z, the adversary predicts the value of the protected attribute.
    """

    def __init__(self, embed_dim):
        super(SimpleAdversary, self).__init__()
        init = torch.randn(embed_dim)
        unit_init = init / (torch.norm(init))
        self.w = torch.nn.Parameter(unit_init)

    def forward(self, x):
        z_hat = torch.matmul(x, self.w)
        return z_hat
