import numpy as np
import os
import sys
from blob.network.ds_model import DownstreamFC
from blob.network.network import BetaVAE_H

# Import experiment specific vae/loss/houskeeping functions
from blob.network.train_functions import loss_function, \
        get_data, get_z, setup_housekeeping, \
        housekeeping, save_housekeeping, pixel_diff, plot_batch, get_label


class VAEConfig:
    def __init__(self):
        self.loss_function = loss_function
        self.get_data = get_data
        self.get_z = get_z
        self.setup_housekeeping = setup_housekeeping
        self.housekeeping = housekeeping
        self.save_housekeeping = save_housekeeping
        self.pixel_diff = pixel_diff
        self.plot_batch = plot_batch
        self.get_label = get_label
        self.device = 'cuda'
        self.dataset_path = './'

        self.dataset_params_train = {'memory' : 2000.0,
                'data_path' : self.dataset_path + 'data/blob/blob_20/train/',
                'data_labels' : ['img', 'vel', 'pos'],
                'data_types' : [np.uint8, float, float],
                'data_shapes' : [(20, 100, 100), (20, 2), (20, 2)],
                'batch_size' : 32,
                'subset' : 0,
                'shuffle' : True,
                'device' : self.device,
                'verbose' : False,
                'reshape' : [[100, 100], [2], [2]],
                }

        self.dataset_params_test = {'memory' : 100.0,
                'data_path' : self.dataset_path + 'data/blob/blob_20/test/',
                'data_labels' : ['img', 'vel', 'pos'],
                'data_types' : [np.uint8, float, float],
                'data_shapes' : [(20, 100, 100), (20, 2), (20, 2)],
                'batch_size' : 32,
                'shuffle' : False,
                'subset' : 0,
                'shuffle' : False,
                'device' : self.device,
                'verbose' : False,
                'reshape' : [[100, 100], [2], [2]],
                }

        self.model_params = {
                'dt_max' : 5,
                'learning_rate' : 1e-04,
                'housekeeping_keys': ['batch_train_loss_rec', 'batch_train_loss_kld', \
                        'batch_train_loss_slow', 'epoch_train_loss', 'epoch_test_pixel_diff']
                }

        self.training_params = {
                'output_path' : os.path.abspath(os.path.dirname(__file__)) + '/models/',
                'device' : self.device,
                'torch_seed' : 0,
                'np_seed' : 0,
                'save_loss' : True,
                'show_reconstructions' : False,
                'save_reconstructions' : True,
                'save_model' : True,
                'verbose' : True,
                'testing' : True,
                'overwrite' : True # Overwrite existing model.pkl files (throws warning)
                }

    def setup_network(self):
        net = BetaVAE_H(self.model_params['latent_dim'], nc=1)
        return net

    def setup_downstream(self, fs_params):
        # fs_params is defined in train_DS.py
        return DownstreamFC(fs_params['layers'])

