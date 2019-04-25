import gzip
import requests
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

def get_mnist(data_dir):
    """Get MNIST data.

    Parameters
    -----------
    data_dir : :obj:`str`
        Diretcory where to store the MNIST dataset.

    Returns
    --------
    :obj:`tuple(torch.FloatTensor)`
        Training and validation sets.

    """
    url = 'http://deeplearning.net/data/mnist/'
    filename = "mnist.pkl.gz"

    path = Path(data_dir, filename)

    if not path.exists():
        r = requests.get(url + filename)
        with open(path, 'wb') as h:
            h.write(r.content)

    with gzip.open(path, "rb") as h:
        (x_train, y_train), (x_valid, y_valid), _ = pickle.load(h, encoding="latin-1")

    return (torch.FloatTensor(x_train),
            torch.FloatTensor(y_train),
            torch.FloatTensor(x_valid),
            torch.FloatTensor(y_valid))

def show_random_samples(dataset, rows=5, cols=5, width=None, height=None, shuffle=True):
    """Show rows * cols random samples."""
    if width is None: width = 1.5*cols
    if height is None: height = 1.5*rows

    if rows * cols == 1:
        axes = [plt.subplots(rows, cols, figsize=(width, height))[1]]
    else:
        axes = plt.subplots(rows, cols, figsize=(width, height))[1].flatten()

    # by default batch_size=1 in DataLoader
    for ax, (x, y) in zip(axes, DataLoader(dataset, shuffle=shuffle)):
        ax.imshow(x.reshape((28, 28)), cmap="gray")
        ax.axis('off')
