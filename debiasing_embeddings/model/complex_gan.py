import torch
import torch.nn as nn


class ComplexGenerator(nn.Module):
    """
    Complex generator model.
    Given the first three words in a gender-based analogy problem x_1, x_2, and x_3,
    the generator predicts the fourth word x_4.
    """
    def __init__(self, embed_dim):
        super(ComplexGenerator, self).__init__()
        self.fc1 = nn.Linear(embed_dim * 3, 1024)
        self.fc2 = nn.Linear(1024, embed_dim)

    def forward(self, word1, word2, word3):
        combined = torch.cat((word1, word2, word3), dim=1)
        x = torch.relu(self.fc1(combined))
        output = self.fc2(x)
        return output


class ComplexDiscriminator(nn.Module):
    def __init__(self, embed_dim):
        super(ComplexDiscriminator, self).__init__()
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