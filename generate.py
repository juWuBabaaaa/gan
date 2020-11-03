import network
import utils
import torch
from torch import nn


class Framework:

    def __init__(self, sample_dimension, noise_dimension, bs):
        self.sample_dimension = sample_dimension
        self.noise_dimension = noise_dimension
        self.bs = bs

        self.D = network.Discriminator(
            sample_dimension=sample_dimension
        )
        self.G = network.Generator(
            noise_dimension=noise_dimension,
            sample_dimension=sample_dimension
        )

        self.optimizer_d = torch.optim.RMSprop(self.D.parameters())
        self.optimizer_g = torch.optim.RMSprop(self.G.parameters())

        self.criterion = nn.BCELoss()

    def train(self, epochs, train_iter, d_steps, g_steps):
        true_labels = torch.ones(self.bs)
        fake_labels = torch.zeros(self.bs)
        for epoch in range(epochs):
            for iteration in range(train_iter):
                for p in self.D.parameters():
                    p.requires_grad = True

                for i in range(d_steps):
                    sample = torch.Tensor(utils.sampler(self.bs, self.sample_dimension))
                    d_real_decision = self.D(sample)
                    d_real_loss = self.criterion(d_real_decision.squeeze(), true_labels)

                    d_fake_data = self.G(torch.randn(self.bs, self.noise_dimension))
                    d_fake_decision = self.D(d_fake_data)
                    d_fake_loss = self.criterion(d_fake_decision.squeeze(), fake_labels)
                    d_loss = d_real_loss + d_fake_loss

                    self.optimizer_d.zero_grad()
                    d_loss.backward()
                    self.optimizer_d.step()

                for j in range(g_steps):
                    for p in self.D.parameters():
                        p.requires_grad = False

                    g_fake_data = self.G(torch.randn(self.bs, self.noise_dimension))
                    g_fake_decision = self.D(g_fake_data)
                    g_loss = self.criterion(g_fake_decision.squeeze(), true_labels)

                    self.optimizer_g.zero_grad()
                    g_loss.backward()
                    self.optimizer_g.step()


