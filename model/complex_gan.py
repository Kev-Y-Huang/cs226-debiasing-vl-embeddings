import torch
import torch.nn as nn


class ComplexPredictor(nn.Module):
    """
    Complex predictor model.
    Given the first three words in a gender-based analogy problem x_1, x_2, and x_3,
    the predictor predicts the fourth word x_4.
    """

    def __init__(self, embed_dim):
        super(ComplexPredictor, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 512)
        self.lrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, embed_dim)

    def forward(self, x):
        v = x[:, 1, :] + x[:, 2, :] - x[:, 0, :]
        output = self.lrelu(self.fc1(v))
        output = self.fc2(output)
        return v - output

    def predict(self, x):
        with torch.no_grad():
            output = self.lrelu(self.fc1(x))
            output = self.fc2(output)
            return x - output

    def predict_batch(self, xs):
        with torch.no_grad():
            y_hats = []
            # Split the input into chunks of size 32 to avoid OOM errors
            for embeds in torch.split(xs, 32):
                y_hats.append(self.predict(embeds))
            return torch.cat(y_hats, axis=0)


class ComplexAdversary(nn.Module):
    def __init__(self, embed_dim):
        super(ComplexAdversary, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, 1)

    def forward(self, embedding):
        x = torch.relu(self.fc1(embedding))
        output = self.fc2(x)
        return output
