import numpy as np
import torch.nn.utils
import argparse
from esn import DeepReservoir
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from utils import get_mackey_glass, RON, PRON


parser = argparse.ArgumentParser(description='training parameters')

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=100,
                    help='hidden size of recurrent net')
parser.add_argument('--batch', type=int, default=30,
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
parser.add_argument('--pron', action="store_true")
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


args = parser.parse_args()
print(args)

assert 1.0 > args.sparsity >= 0.0, "Sparsity in [0, 1)"

main_folder = 'result'

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
print("Using device ", device)
n_inp = 1
n_out = 1
washout = 200

@torch.no_grad()
def test(dataset, target, classifier, scaler):
    dataset = dataset.reshape(1, -1, 1).to(device)
    target = target.reshape(-1, 1).numpy()
    activations = model(dataset)[0].cpu().numpy()
    activations = activations[:, washout:]
    activations = activations.reshape(-1, args.n_hid)
    activations = scaler.transform(activations)
    predictions = classifier.predict(activations)
    mse = np.mean(np.square(predictions - target))
    rmse = np.sqrt(mse)
    norm = np.sqrt(np.square(target).mean())
    nrmse = rmse / (norm + 1e-9)
    return nrmse

gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)


train_mse, valid_mse, test_mse = [], [], []
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
    elif args.pron:
        model = PRON(n_inp, args.n_hid, args.dt, gamma, epsilon,
                     args.inp_scaling, device=device).to(device)
    else:
        raise ValueError('Wrong model name')

    (train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_mackey_glass()

    dataset = train_dataset.reshape(1, -1, 1).to(device)
    target = train_target.reshape(-1, 1).numpy()
    activations = model(dataset)[0].cpu().numpy()
    activations = activations[:, washout:]
    activations = activations.reshape(-1, args.n_hid)
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = Ridge(max_iter=1000).fit(activations, target)
    train_nmse = test(train_dataset, train_target, classifier, scaler)
    valid_nmse = test(valid_dataset, valid_target, classifier, scaler) if not args.use_test else 0.0
    test_nmse = test(test_dataset, test_target, classifier, scaler) if args.use_test else 0.0
    train_mse.append(train_nmse)
    valid_mse.append(valid_nmse)
    test_mse.append(test_nmse)

if args.ron:
    f = open(f'{main_folder}/MG_log_RON_{args.topology}.txt', 'a')
elif args.pron:
    f = open(f'{main_folder}/MG_log_PRON_{args.topology}.txt', 'a')
elif args.esn:
    f = open(f'{main_folder}/MG_log_ESN.txt', 'a')
else:
    raise ValueError("Wrong model choice.")

ar = ''
for k, v in vars(args).items():
    ar += f'{str(k)}: {str(v)}, '
ar += (f'train: {[str(round(train_acc, 2)) for train_acc in train_mse]} '
       f'valid: {[str(round(valid_acc, 2)) for valid_acc in valid_mse]} '
       f'test: {[str(round(test_acc, 2)) for test_acc in test_mse]}'
       f'mean/std train: {np.mean(train_mse), np.std(train_mse)} '
       f'mean/std valid: {np.mean(valid_mse), np.std(valid_mse)} '
       f'mean/std test: {np.mean(test_mse), np.std(test_mse)}')
f.write(ar + '\n')
f.close()
