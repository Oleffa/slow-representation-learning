import os
import shutil 
import pickle
import numpy as np
import sys
import cv2
import torch

sys.path.append('..')
from network.VAE import VAE

class Model:
    def __init__(self, vaeconfig):
        if None is not vaeconfig:
            self.vaeconfig = vaeconfig
            self.training_params = vaeconfig.training_params
            self.model_params = vaeconfig.model_params
            self.save_path = '{}/{}/{}dim_g{}_b{}_l{}/'.format(self.training_params['output_path'], \
                    self.model_params['method'], self.model_params['latent_dim'], \
                    self.model_params['gamma'], self.model_params['beta'], \
                    self.model_params['rate_prior'])
            self.method = self.model_params['method']
            self.dataset_params_train = vaeconfig.dataset_params_train
            self.dataset_params_test = vaeconfig.dataset_params_test

            # Make sure we use a known method from the list defined in the global settings
            # Catch some problems when setting model params
            if self.method == 'bvae':
                if self.model_params['gamma'] != 0.0:
                    print("Warning, started bvae training with gamma!=0. Setting it to 0!")
                    self.model_params['gamma'] = 0.0

            net = self.vaeconfig.setup_network()
            self.vae = VAE(self, self.vaeconfig, net).to(self.training_params['device'])

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)
                print("Created new save path: {}".format(self.save_path))
            else:
                print("Warning the output path {} already exists.".format(self.save_path)) 
                if not self.training_params['overwrite']:
                    print("Exiting")
                    exit()


            # Initially there is no model file yet
            self.model_file = None
            self.epoch = 0
            # Initially there is no housekeeping data
            self.housekeeping = Housekeeping(self.model_params)
            self.test_reconstructions = []
            # This is a list of downstream experiments
            self.experiments = []
            # Overwrite existing model file if desired
            if os.path.isfile(self.save_path + 'model.pkl'):
                if self.training_params['overwrite']:
                    self.save_model()
                else:
                    print("Error, model already exists at {}. Consider setting training_params['overwrite']=True".\
                            format(self.save_path+'model.pkl'))
            else:
                self.save_model()
            print("Created new model at {}!".format(self.save_path))

    def train(self):
        self.vae.train()

    def load_model(self, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        # Update data path based on global config if trained VAE on another device
        split_original = model.vaeconfig.dataset_params_test['data_path'].split('/')
        new_data_path = model.vaeconfig.dataset_path.split('/')[-1]
        cut_idx = split_original.index(new_data_path) + 1
        model.vaeconfig.dataset_params_test['data_path'] = model.vaeconfig.dataset_path
        for i in range(cut_idx, len(split_original)):
            model.vaeconfig.dataset_params_test['data_path'] += '/' + split_original[i]

        split_original = model.vaeconfig.dataset_params_train['data_path'].split('/')
        new_data_path = model.vaeconfig.dataset_path.split('/')[-1]
        cut_idx = split_original.index(new_data_path) + 1
        model.vaeconfig.dataset_params_train['data_path'] = model.vaeconfig.dataset_path
        for i in range(cut_idx, len(split_original)):
            model.vaeconfig.dataset_params_train['data_path'] += '/' + split_original[i]


        model.training_params['output_path'] = os.path.abspath(model.vaeconfig.training_params['output_path'])
        model.save_path = '{}/{}/{}dim_g{}_b{}_l{}/'.format(model.training_params['output_path'], \
                model.model_params['method'], model.model_params['latent_dim'], \
                model.model_params['gamma'], model.model_params['beta'], \
                model.model_params['rate_prior'])
        print("Succesfully loaded model from {}".format(path))
        return model

    def save_model(self):
        # Todo also save the reconstructions here
        if self.training_params['save_model']:
            with open(self.save_path+'model.pkl', 'wb') as f:
                pickle.dump(self, f)
        else:
            print("Warning, training_params['save_model'] = False")
    # Visualisation and evaluation functions
    def verify(self):
        valid = True
        if self.epoch != self.model_params['train_epochs']: return False, "Verification failed: Current model epochs != model_params['train_epochs']"
        return valid, None

    def show_reconstruction(self, comp, epoch, border):
        import matplotlib.pyplot as plt
        plt.imshow(comp)
        plt.show()
        return
        rgb = False
        if comp.shape[-1] == 3 and len(comp.shape) == 5:
            rgb = True
        else:
            # Handle special case for dml dataset
            print("Warning for first blob experiment, plottin has changed and might fail here. Other experiments should work")
            comp = torch.from_numpy(comp).permute(0,3,1,2).numpy()
        if border == 'black':
            if rgb:
                pad = np.zeros((comp.shape[0], comp.shape[1], comp.shape[2]+2, comp.shape[3]+2, 3))
            else:
                pad = np.zeros((comp.shape[0], comp.shape[1], comp.shape[2]+2, comp.shape[3]+2))
        elif border == 'white':
            if comp.max() <= 1:
                scale = 1
            else:
                scale = 255
            if rgb:
                pad = np.ones((comp.shape[0], comp.shape[1], comp.shape[2]+2, comp.shape[3]+2, 3)) * scale
            else:
                pad = np.ones((comp.shape[0], comp.shape[1], comp.shape[2]+2, comp.shape[3]+2)) * scale
        if rgb:
            pad[:, :, 1:-1, 1:-1, :] = comp
            comp = np.concatenate(pad, axis=-3)
            comp = np.concatenate(comp, axis=-2)
        else:
            pad[:, :, 1:-1, 1:-1] = comp
            comp = np.concatenate(pad, axis=-2)
            comp = np.concatenate(comp, axis=-1)

        comp = cv2.cvtColor(comp.astype(np.float32), cv2.COLOR_BGR2RGB)

        vae_name = "{} l: {} b: {}".format(self.model_params['method'], self.model_params['gamma'], \
                self.model_params['beta'])
        comp = cv2.putText(comp, "{}".format(vae_name), (20, 27), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 0, 0), 2)
        comp = cv2.putText(comp, "Epoch: {}/{}".format(epoch+1, len(self.test_reconstructions)), (20, 60), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 0, 0), 2)
        cv2.imshow("Reconstructions", comp)
        cv2.waitKey(10000)

    def show_reconstruction_single(self, epoch, border='white'):
        epoch = range(self.epoch)[epoch]
        comp = self.test_reconstructions[epoch]
        self.show_reconstruction(comp, epoch, border)
        cv2.waitKey(10000)

    def show_reconstruction_history(self, border='white'):
        for epoch, comp in enumerate(self.test_reconstructions):
            self.show_reconstruction(comp, epoch, border)
        cv2.waitKey(10000)

class Housekeeping():
    def __init__(self, model_params):
        self.vae_id = "{}_{}dim_g{}_b{}_l{}".format(model_params['method'], model_params['latent_dim'], model_params['gamma'], \
                model_params['beta'], model_params['rate_prior'])
        self.vae_name = "{} $\gamma$: {} b: {}".format(model_params['method'], model_params['gamma'], model_params['beta'])
        keys = model_params['housekeeping_keys']
        self.data = {}
        for k in keys:
            self.data.update({k : []})

    def add(self, key, value):
        self.data[key].append(value)

    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def plot(self, key, ax, label, moving_average=None):
        if moving_average== None:
            ax.plot(self.data[key], label=self.vae_name)
        else:
            ax.plot(self.running_mean(self.data[key], moving_average), label=self.vae_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
