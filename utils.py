import logging
import zipfile
import gzip
import requests
import pickle
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim

# see as well: LambdaLR, StepLR, ReduceLROnPlateau, MultiStepLR, ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR

log = logging.getLogger(__name__)

class CustomDataLoader:
    """Preprocess batches in a dataloader."""
    def __init__(self, dl, func):
        self.dl, self.func = dl, func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl: yield (self.func(*b))

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

    Note
    -----
    Using model.train() in train() and model.eval() in evaluate() changes the
    behavior of some of the layers in the model. For example, Dropout and
    BatchNorm should be turned off during evaluation and on during training.

    """
    def train():
        """Train model for one epoch."""
        model.train()
        for batch_index, (xb, yb) in enumerate(train_dl):
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

def count_parameters(model):
    """Return the number of parameters in a model.

    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def get_hymenoptera(data_dir):
    """Get HYMENOPTERA data.

    I saw it in the following tutorial:
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    Parameters
    -----------
    data_dir : :obj:`str`
        Diretcory where to store the dataset.

    """
    url = 'https://download.pytorch.org/tutorial/'
    filename = "hymenoptera_data.zip"

    path = Path(data_dir, filename)

    if not path.exists():
        r = requests.get(url + filename)
        with open(path, 'wb') as h:
            h.write(r.content)

    with zipfile.ZipFile(path, "r") as h:
        h.extractall(data_dir)

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

def rosenbrock(x):
    """Rosenbrock function."""
    return (1-x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosenbrock_contour(iterates=None, **kwargs):
    """Plot contours of the Rosenbrock function."""
    n = 250
    X, Y = np.meshgrid(np.linspace(-2,2,n),
                       np.linspace(-1,3,n))
    fig = plt.figure(figsize=(14,8))
    plt.contour(X, Y, rosenbrock([X,Y]), np.logspace(-0.5, 3.5, 20, base=10), cmap='gray')

    if iterates is not None:
        if isinstance(iterates, dict):
            for key, value in iterates.items():
                plt.plot(*(zip(*value['iterates'])),
                         ls='--', marker='o',
                         label='lr: {}'.format(key))
        else:
            plt.plot(*(zip(*iterates)), 'bo--')

    plt.xlabel('x')
    plt.ylabel('y')
    if isinstance(iterates, dict): plt.legend()

    if 'lr_history' in kwargs and kwargs['lr_history']:
        plt.figure(figsize=(13,8))
        plt.plot(kwargs['lr_history'])
        plt.xlabel('iter')
        plt.ylabel('learning rate')
        plt.title('Cosine annealing')
        plt.grid(True)

def gd(x0, lr=1e-3, mu=0.0, nesterov=False,
       use_torch_opt=False, schedule_lr=False,
       N=500, model=rosenbrock, **kwargs):
    """Gradient descent.

    Parameters
    -----------
    x0 : :obj:`numpy.array`
        Initial iterate.

    lr : :obj:`float`
        Learning rate.

    mu : :obj:`float` [0, 1)
        Momentum, see [1] (1-2).

    nesterov : :obj:`bool`
        If `True` use Nesterov accelerated gradient,
        see [1] (3-4).

    use_torch_opt : :obj:`bool`
        If `True` use torch.optim.SGD. Note that there are
        slight differences in the interpretation of the
        `lr` and `mu` parameters (see the documentation of optim.SGD),
        hence the results would be different from the manual version.

    schedule_lr : :obj:`bool`
        Use scheduler for the learning rate
        (used only when use_torch_opt=`True`).

    N : :obj:`int`
        Number of iterations.

    model : :obj:`callable`
        The model.

    References
    -----------
    [1] http://proceedings.mlr.press/v28/sutskever13.pdf

    Returns
    --------
    :obj:`list(numpy.array)`
        The iterates.
    """
    if schedule_lr and not use_torch_opt: use_torch_opt = True

    x = torch.tensor(x0, requires_grad=True)

    lr_history = []
    opt = None
    if use_torch_opt:
        opt = optim.SGD([x], lr=lr, momentum=mu, nesterov=nesterov)

        if schedule_lr:
            if 'T_max' in kwargs: T_max = kwargs['T_max']
            else: T_max = N//5

            if 'eta_min' in kwargs: eta_min = kwargs['eta_min']
            else: eta_min = lr//3

            scheduler = CosineAnnealingLR(opt, T_max, eta_min)

    iterates = [x.clone().data.numpy()]
    v = torch.zeros(len(x)).double()
    for k in range(N):
        if schedule_lr:
            scheduler.step()
            lr_history.append(opt.param_groups[0]['lr'])

        if opt is None:  # perform step manually
            if nesterov: y = model(x + mu * v)
            else: y = model(x)

            y.backward()
            with torch.no_grad():
                v = mu * v - lr * x.grad
                x += v
                x.grad.zero_()
        else:
            y = model(x)
            y.backward()
            opt.step()
            opt.zero_grad()

        iterates.append(x.clone().data.numpy())

    return {'iterates': iterates,
            'lr_history': lr_history}
