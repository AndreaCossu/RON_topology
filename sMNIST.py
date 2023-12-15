from torch import nn
import torch.nn.utils
import numpy as np
from utils import get_mnist_data, RON
import argparse
from tqdm import tqdm
from esn import DeepReservoir
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=256,
                    help='hidden size of recurrent net')
parser.add_argument('--batch', type=int, default=1000,
                    help='batch size')
parser.add_argument('--dt', type=float, default=0.042,
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma', type=float, default=2.7,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon', type=float, default=4.7,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--gamma_range', type=float, default=2.7,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon_range', type=float, default=4.7,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--esn', action="store_true")
parser.add_argument('--ron', action="store_true")
parser.add_argument('--inp_scaling', type=float, default=1.,
                    help='ESN input scaling')
parser.add_argument('--rho', type=float, default=0.99,
                    help='ESN spectral radius')
parser.add_argument('--leaky', type=float, default=1.0,
                    help='ESN spectral radius')
parser.add_argument('--use_test', action="store_true")
parser.add_argument('--trials', type=int, default=1,
                    help='How many times to run the experiment')
parser.add_argument('--topology', type=str, default='full',
                    choices=['full', 'ring', 'band', 'lower', 'toeplitz', 'orthogonal'],
                    help='Topology of the reservoir')
parser.add_argument('--sparsity', type=float, default=0.0,
                    help='Sparsity of the reservoir')
parser.add_argument('--reservoir_scaler', type=float, default=1.0,
                    help='Scaler in case of ring/band/toeplitz reservoir')

main_folder = 'result'
args = parser.parse_args()
print(args)

assert 1.0 > args.sparsity >= 0.0, "Sparsity in [0, 1)"

@torch.no_grad()
def test(data_loader, classifier, scaler):
    activations, ys = [], []
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        output = model(images)[-1][0]
        activations.append(output.cpu())
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")

n_inp = 1
n_out = 10

gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)

train_accs, valid_accs, test_accs = [], [], []
for i in range(args.trials):
    if args.esn:
        model = DeepReservoir(n_inp, tot_units=args.n_hid, spectral_radius=args.rho,
                              input_scaling=args.inp_scaling,
                              connectivity_recurrent=int((1 - args.sparsity) * args.n_hid),
                              connectivity_input=args.n_hid, leaky=args.leaky).to(device)
    elif args.ron:
        model = RON(n_inp, args.n_hid, args.dt, gamma, epsilon, args.rho,
                    args.inp_scaling, topology=args.topology, sparsity=args.sparsity,
                    reservoir_scaler=args.reservoir_scaler, device=device).to(device)

    else:
        raise ValueError("Wrong model choice.")

    train_loader, valid_loader, test_loader = get_mnist_data(args.batch,args.batch)

    activations, ys = [], []
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        output = model(images)[-1][0]
        activations.append(output.cpu())
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000).fit(activations, ys)
    train_acc = test(train_loader, classifier, scaler)
    valid_acc = test(valid_loader, classifier, scaler) if not args.use_test else 0.0
    test_acc = test(test_loader, classifier, scaler) if args.use_test else 0.0
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    test_accs.append(test_acc)

if args.ron:
    f = open(f'{main_folder}/sMNIST_log_RON_{args.topology}.txt', 'a')
elif args.esn:
    f = open(f'{main_folder}/sMNIST_log_ESN.txt', 'a')
else:
    raise ValueError("Wrong model choice.")

ar = ''
for k, v in vars(args).items():
    ar += f'{str(k)}: {str(v)}, '
ar += (f'train: {[str(round(train_acc, 2)) for train_acc in train_accs]} '
       f'valid: {[str(round(valid_acc, 2)) for valid_acc in valid_accs]} '
       f'test: {[str(round(test_acc, 2)) for test_acc in test_accs]}' 
       f'mean/std train: {np.mean(train_accs), np.std(train_accs)} '
       f'mean/std valid: {np.mean(valid_accs), np.std(valid_accs)} '
       f'mean/std test: {np.mean(test_accs), np.std(test_accs)}')
f.write(ar + '\n')
f.close()
