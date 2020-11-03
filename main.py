from torch import nn
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import utils
import generate


sample_dimension = 2
noise_dimension = 2
bs = 1000

frame = generate.Framework(sample_dimension, noise_dimension, bs)
frame.train(1, 1000, 6, 1)

fakes = frame.G(torch.randn(200, noise_dimension)).detach().numpy()

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.set_xlim([-0.5, 0.5])
ax1.set_ylim([-0.5, 0.5])
ax1.plot(fakes[:, 0], fakes[:, 1], '.')
ax2 = fig.add_subplot(122)
data = utils.sampler(200, sample_dimension)
ax2.set_xlim([-0.5, 0.5])
ax2.set_ylim([-0.5, 0.5])
ax2.plot(data[:, 0], data[:, 1], '.')
plt.show()

