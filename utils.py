import numpy as np
import matplotlib.pyplot as plt


def sampler(bs, sample_dimension):

    return 0.1 * np.random.randn(bs, sample_dimension)


if __name__ == '__main__':

    x = sampler(
        bs=500,
        sample_dimension=2
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.plot(x[:, 0], x[:, 1], '.')

    plt.show()

