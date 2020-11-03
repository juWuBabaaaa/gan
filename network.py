from torch import nn


class Discriminator(nn.Module):

    def __init__(self, sample_dimension):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(sample_dimension, 100),
            nn.Sigmoid(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):

    def __init__(self, noise_dimension, sample_dimension):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dimension, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, sample_dimension)
        )

    def forward(self, x):
        return self.main(x)


