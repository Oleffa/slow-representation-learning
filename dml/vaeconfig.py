import numpy as np
import os
import sys
from dml.network.ds_model import DownstreamFC
from dml.network.network import BetaVAE_H


# Import experiment specific vae/loss/houskeeping functions
from dml.network.train_functions import loss_function, \
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


        self.dataset_params_train = {'memory' : 4000,
            'data_path' : self.dataset_path + 'data/dml/train/',
            'data_labels' : ['img', 'labels'],
            'data_types' : [np.uint8, float],
            'data_shapes' : [(5, 180, 240), (5, 6)],
            'batch_size' : 16,
            'subset' : 0,
            'shuffle' : True,
            'device' : self.device,
            'verbose' : False,
            'reshape' : [[100, 100], [6]],
            }

        self.dataset_params_test = {'memory' : 500,
            'data_path' : self.dataset_path + 'data/dml/test/',
            'data_labels' : ['img', 'labels'],
            'data_types' : [np.uint8, float],
            'data_shapes' : [(5, 180, 240), (5, 6)],
            'batch_size' : 16,
            'subset' : 500,
            'shuffle' : True,
            'device' : self.device,
            'verbose' : False,
            'reshape' : [[100, 100], [6]],
            }

        self.model_params = {
                'dt_max' : 4,
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

