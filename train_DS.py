import torch
import argparse
import numpy as np
from utils.model import Model
from network.DS import Downstream

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

# Downstream params
parser.add_argument('--ds_epochs', dest='ds_epochs', type=int, help='Epochs for downstream training.')
parser.add_argument('--subset', dest='subset', type=int, help='Size of the used training data as (1/subset) * train data.')
parser.add_argument('--seed', dest='seed', type=int, help='Seed for downstream task learning.')
parser.add_argument('--ds_task', dest='ds_task', type=str, help='Which dataset label to use for the ds task')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

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

#Load existing model and continue training
model = Model(None)
if args.method == 'bvae':
    if args.gamma != 0.0:
        print("Warning, started bvae training with gamma!=0. Setting it to 0!")
        args.gamma = 0.0
    if args.rate_prior != 0.0:
        print("Warning, started bvae training with rate_prior!=0. Setting it to 0!")
        args.rate_prior = 0.0
if args.method == 'l1' or args.method == 'l2':
    if args.rate_prior != 0.0:
        print("Warning, started l1 or l2 training with rate_prior!=0. Setting it to 0!")
        args.rate_prior = 0.0

restore_path = vc.training_params['output_path'] + '/{}/{}dim_g{}_b{}_l{}/model.pkl'.format(\
        args.method, args.latent_dim, args.gamma, args.beta, args.rate_prior)
model = model.load_model(restore_path)

# Fill params into the vaeconfig in case something changed. The other params should not be changed really
model.vaeconfig.model_params['train_epochs'] = args.epochs
model.training_params['device'] = vc.training_params['device']
model.dataset_params_train['data_path'] = vc.dataset_params_train['data_path']
model.dataset_params_test['data_path'] = vc.dataset_params_test['data_path']

if args.experiment == 'ball':
    layers = [2*model.vaeconfig.model_params['latent_dim'], 50, 50, 2]
elif args.experiment == 'pong':
    layers = [2*model.vaeconfig.model_params['latent_dim'], 50, 50, 3]
elif args.experiment == 'dml':
    layers = [2*model.vaeconfig.model_params['latent_dim'], 50, 50, 3]
elif args.experiment == 'kitti':
    layers = [2*model.vaeconfig.model_params['latent_dim'], 50, 50, 12]
else:
    print('Error, unknown experiment: ', args.experiment)
    exit()

# Setup downstream task model
model.fs_params = {
        'layers' : layers,
        'seed' : args.seed,
        'epochs' : args.ds_epochs,
        'testing' : True,
        'save_tests' : True,
        'save_model' : True,
        'data_label' : args.ds_task,
        'save_loss' : True
        }
model.dataset_params_train['subset'] = args.subset
model.dataset_params_test['subset'] = 0

model.ds = Downstream(model)

try:
    model.ds.train()
except KeyboardInterrupt:
    print("Keyboard interrupt detected, trying to shutdown dataloaders...")
    try:
        train_dataset.stop()
        test_dataset.stop()
        print("Dataloaders stopped")
        exit()
    except Exception as e:
        print("Could not stop dataloaders after being done. Probably they already exited naturally")
        exit()
