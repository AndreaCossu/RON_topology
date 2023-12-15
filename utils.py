from scipy.integrate import odeint
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from esn import spectral_norm_scaling
import torch.utils.data as data
import torch.nn.functional as F


class datasetforRC(data.Dataset):
    """
    This class assumes mydata to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """

    def __init__(self, mydata):
        self.mydata = mydata

    def __getitem__(self, idx):
        sample = self.mydata[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding gives problems with scikit-learn LogisticRegression of RC models
        return idx_inp, idx_targ

    def __len__(self):
        return len(self.mydata)


class Adiac_dataset(data.Dataset):
    """
    This class assumes mydata to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """

    def __init__(self, mydata):
        self.mydata = mydata

    def __getitem__(self, idx):
        sample = self.mydata[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding targets
        idx_targ = F.one_hot(idx_targ.type(torch.int64), num_classes=37).float()
        # reshape target for torch (batch, classes)
        idx_targ = idx_targ.reshape(idx_targ.shape[1])
        return idx_inp, idx_targ

    def __len__(self):
        return len(self.mydata)


class LSTM(nn.Module):
    def __init__(self, n_inp, n_hid, n_out):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_inp, n_hid, batch_first=True,
                                  num_layers=1)
        self.readout = torch.nn.Linear(n_hid, n_out)

    def forward(self, x):
        out, h = self.lstm(x)
        out = self.readout(out[:, -1])
        return out


def get_hidden_topology(n_hid, topology, sparsity, scaler):
    if topology == 'full':
        h2h = 2 * (2 * torch.rand(n_hid, n_hid) - 1)
    elif topology == 'lower':
        h2h = torch.tril(2 * torch.rand(n_hid, n_hid) - 1)
        if sparsity > 0:
            n_zeroed_diagonals = int(sparsity * n_hid)
            for i in range(n_hid-1, n_hid - n_zeroed_diagonals - 1, -1):
                h2h.diagonal(-i).zero_()
    elif topology == 'orthogonal':
        rand = torch.rand(n_hid, n_hid)
        orth = torch.linalg.qr(rand)[0]
        identity = torch.eye(n_hid)
        if sparsity > 0:
            n_zeroed_rows = int(sparsity * n_hid)
            idxs = torch.randperm(n_hid)[:n_zeroed_rows].tolist()
            identity[idxs, idxs] = 0.
        h2h = torch.matmul(identity, orth)
    elif topology == 'band':
        bandwidth = int(scaler)  # 5
        h2h = torch.zeros(n_hid, n_hid)
        for i in range(bandwidth, n_hid - bandwidth):
            h2h[i, i - bandwidth: i + bandwidth + 1] = 2 * torch.rand(2 * bandwidth + 1) - 1
        for i in range(0, bandwidth):
            h2h[i, 0: i + bandwidth + 1] = 2 * torch.rand(i + bandwidth + 1) - 1
        for i in range(n_hid - bandwidth, n_hid):
            h2h[i, i - bandwidth: n_hid] = 2 * torch.rand(n_hid + bandwidth - i) - 1
    elif topology == 'ring':
        # scaler = 1
        h2h = torch.zeros(n_hid, n_hid)
        for i in range(1, n_hid):
            h2h[i, i - 1] = 1
        h2h[0, n_hid - 1] = 1
        h2h = scaler * h2h
    elif topology == 'toeplitz':
        from scipy.linalg import toeplitz
        bandwidth = int(scaler)  # 5
        upperdiagcoefs = np.zeros(n_hid)
        upperdiagcoefs[:bandwidth] = 2 * torch.rand(bandwidth) - 1
        lowerdiagcoefs = np.zeros(n_hid)
        lowerdiagcoefs[:bandwidth] = 2 * torch.rand(bandwidth) - 1
        lowerdiagcoefs[0] = upperdiagcoefs[0]  # diagonal coefficient
        h2h = toeplitz(list(lowerdiagcoefs), list(upperdiagcoefs))
        h2h = torch.Tensor(h2h)
    else:
        raise ValueError("Wrong topology choice.")
    return h2h


class RON(nn.Module):
    """
    Batch-first (B, L, I)
    """
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, rho, input_scaling, topology='full',
                 reservoir_scaler=0., sparsity=0.0, device='cpu'):
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.dt = dt
        if isinstance(gamma, tuple):
            gamma_min, gamma_max = gamma
            self.gamma = torch.rand(n_hid, requires_grad=False, device=device) * (gamma_max - gamma_min) + gamma_min
        else:
            self.gamma = gamma
        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            self.epsilon = torch.rand(n_hid, requires_grad=False, device=device) * (eps_max - eps_min) + eps_min
        else:
            self.epsilon = epsilon

        h2h = get_hidden_topology(n_hid, topology, sparsity, reservoir_scaler)
        h2h = spectral_norm_scaling(h2h, rho)
        self.h2h = nn.Parameter(h2h, requires_grad=False)

        x2h = torch.rand(n_inp, n_hid) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)

    def cell(self, x, hy, hz):
        hz = hz + self.dt * (torch.tanh(
            torch.matmul(x, self.x2h) + torch.matmul(hy, self.h2h) + self.bias)
                             - self.gamma * hy - self.epsilon * hz)

        hy = hy + self.dt * hz
        return hy, hz

    def forward(self, x):
        hy = torch.zeros(x.size(0),self.n_hid).to(self.device)
        hz = torch.zeros(x.size(0),self.n_hid).to(self.device)
        all_states = []
        for t in range(x.size(1)):
            hy, hz = self.cell(x[:, t],hy,hz)
            all_states.append(hy)

        return torch.stack(all_states, dim=1), [hy]  # list to be compatible with ESN implementation


def get_mnist_data(bs_train,bs_test):
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [57000,3000])

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs_train,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)

    return train_loader, valid_loader, test_loader


def get_Adiac_data(bs_train,bs_test, whole_train=False, RC=True):

    def fromtxt_to_numpy(filename='Adiac_TRAIN.txt', valid_len=120):
        # read the txt file
        adiacdata = np.genfromtxt(filename, dtype='float64')
        # create a list of lists with each line of the txt file
        l = []
        for i in adiacdata:
            el = list(i)
            while len(el) < 3:
                el.append('a')
            l.append(el)
        # create a numpy array from the list of lists
        arr = np.array(l)
        if valid_len is None:
            test_targets = arr[:,0]-1
            test_series = arr[:,1:]
            return test_series, test_targets
        else:
            if valid_len == 0:
                train_targets = arr[:,0]-1
                train_series = arr[:,1:]
                val_targets = arr[0:0,0] # empty
                val_series = arr[0:0,1:] # empty
            elif valid_len > 0 :
                train_targets = arr[:-valid_len,0]-1
                train_series = arr[:-valid_len,1:]
                val_targets = arr[-valid_len:,0]-1
                val_series = arr[-valid_len:,1:]
            return train_series, train_targets, val_series, val_targets

    # Generate list of input-output pairs
    def inp_out_pairs(data_x, data_y):
        mydata = []
        for i in range(len(data_y)):
            sample = (data_x[i,:], data_y[i])
            mydata.append(sample)
        return mydata

    # generate torch datasets
    if whole_train:
        valid_len = 0
    else:
        valid_len = 120
    train_series, train_targets, val_series, val_targets = fromtxt_to_numpy(filename='Adiac_TRAIN.txt', valid_len=valid_len)
    mytraindata, myvaldata = inp_out_pairs(train_series, train_targets), inp_out_pairs(val_series, val_targets)
    if RC:
        mytraindata, myvaldata = datasetforRC(mytraindata), datasetforRC(myvaldata)
        test_series, test_targets = fromtxt_to_numpy(filename='Adiac_TEST.txt', valid_len=None)
        mytestdata = inp_out_pairs(test_series, test_targets)
        mytestdata = datasetforRC(mytestdata)
    else:
        mytraindata, myvaldata = Adiac_dataset(mytraindata), Adiac_dataset(myvaldata)
        test_series, test_targets = fromtxt_to_numpy(filename='Adiac_TEST.txt', valid_len=None)
        mytestdata = inp_out_pairs(test_series, test_targets)
        mytestdata = Adiac_dataset(mytestdata)


    # generate torch dataloaders
    mytrainloader = data.DataLoader(mytraindata,
                    batch_size=bs_train, shuffle=True, drop_last=False)
    myvaloader = data.DataLoader(myvaldata,
                        batch_size=bs_test, shuffle=False, drop_last=False)
    mytestloader = data.DataLoader(mytestdata,
                batch_size=bs_test, shuffle=False, drop_last=False)
    return mytrainloader, myvaloader, mytestloader

def get_mackey_glass(lag=84, washout=200):
    with open('mackey-glass.csv', 'r') as f:
        dataset = f.readlines()[0]

    # 10k steps
    dataset = torch.tensor([float(el) for el in dataset.split(',')]).float()

    end_train = int(dataset.shape[0] / 2)
    end_val = end_train + int(dataset.shape[0] / 4)
    end_test = dataset.shape[0]

    train_dataset = dataset[:end_train-lag]
    train_target = dataset[washout+lag:end_train]

    val_dataset = dataset[end_train:end_val-lag]
    val_target = dataset[end_train+washout+lag:end_val]

    test_dataset = dataset[end_val:end_test-lag]
    test_target = dataset[end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)
