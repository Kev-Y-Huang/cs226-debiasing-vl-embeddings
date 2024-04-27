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
        self.fc1 = nn.Linear(embed_dim * 3, 1024)
        self.fc2 = nn.Linear(1024, embed_dim)

    def forward(self, x):
        combined = torch.cat((x[:, 1, :], x[:, 2, :], x[:, 0, :]), dim=1)
        x = torch.relu(self.fc1(combined))
        output = self.fc2(x)
        return output


class ComplexAdversary(nn.Module):
    def __init__(self, embed_dim):
        super(ComplexAdversary, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embedding):
        x = torch.relu(self.fc1(embedding))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        output = self.sigmoid(x)
        return output