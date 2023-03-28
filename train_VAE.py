import numpy as np
import torch
import argparse
from utils.model import Model

# Get args
parser = argparse.ArgumentParser(description='Train a model on the blob dataset.')
parser.add_argument('--x', dest='experiment', type=str, help='Which experiment to run.')
parser.add_argument('--m', dest='method', type=str, help='Which training method to use.')
parser.add_argument('--g', dest='gamma', type=float, help='Gamma value (temporal regularization).')
parser.add_argument('--b', dest='beta', type=float, help='Beta value.')
parser.add_argument('--z', dest='latent_dim', type=int, help='Dimensions of latent variable z.')
parser.add_argument('--e', dest='epochs', type=int, help='Epochs for VAE training. In iterations '\
        + 'through the full dataset specified in vaeconfig/dataset_params.')
parser.add_argument('--l', dest='rate_prior', type=float, help='Rate prior used by the Slow-VAE.')
args = parser.parse_args()

# Import experiment specific config functions
if args.experiment == 'ball':
    from blob import vaeconfig
elif args.experiment == 'pong':
    from pong import vaeconfig
elif args.experiment == 'dml':
    from dml import vaeconfig
elif args.experiment == 'kitti':
    from kitti import vaeconfig
else:
    print('Error, unknown experiment: ', args.experiment)
    exit()

vc = vaeconfig.VAEConfig()


torch.manual_seed(vc.training_params['torch_seed'])
np.random.seed(vc.training_params['np_seed'])

vc.model_params['method'] = args.method
vc.model_params['beta'] = args.beta
vc.model_params['gamma'] = args.gamma
vc.model_params['latent_dim'] = args.latent_dim
vc.model_params['train_epochs'] = args.epochs
vc.model_params['rate_prior'] = args.rate_prior
model = Model(vc)

try:
    model.train()
except KeyboardInterrupt:
    print("Keyboard interrupt detected, dataloaders might still be up and running")
    exit()
