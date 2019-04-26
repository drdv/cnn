import logging
import gzip
import requests
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

log = logging.getLogger(__name__)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    """Train model for for a given number of epochs.

    Parameters
    -----------
    epochs : :obj:`int`
        Number fo epochs .

    model : subclass of :obj:`nn.Module`
        The model.

    loss_func : :obj:`callable`
        Loss function accepting two parameters: (pred, true)

    opt : :obj:`torch.optim`
        Optimization routine.

    train_dl : :obj:`DataLoader`
        Data loader for the training set.

    valid_dl : :obj:`DataLoader`
        Data loader for the validation set.

    """
    def train():
        """Train model for one epoch."""
        model.train()
        for xb, yb in train_dl:
            loss = loss_func(model(xb), yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

    def evaluate():
        """Evaluate model using the validation set."""
        model.eval()
        with torch.no_grad():
            loss, n = 0, 0
            for xb, yb in valid_dl:
                n += len(xb)
                loss += loss_func(model(xb), yb) * len(xb)

        return loss/n

    for epoch in range(epochs):
        train()
        log.info('epoch {}: {}'.format(epoch, evaluate()))

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

    return map(torch.tensor, (x_train, y_train, x_valid, y_valid))

def show_random_samples(batch, rows=5, cols=5, width=None, height=None, shuffle=True):
    """Show rows * cols random samples from a batch (without labels)."""
    if width is None: width = 1.5*cols
    if height is None: height = 1.5*rows

    if rows * cols == 1:
        axes = [plt.subplots(rows, cols, figsize=(width, height))[1]]
    else:
        axes = plt.subplots(rows, cols, figsize=(width, height))[1].flatten()

    # by default batch_size=1 in DataLoader
    for ax, x in zip(axes, DataLoader(TensorDataset(batch), shuffle=shuffle)):
        ax.imshow(x[0].reshape(batch.shape[-2:]), cmap="gray")
        ax.axis('off')
