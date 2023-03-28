import torch
import numpy as np
import os
import sys
# Debug
from parallel_dataloader.dataloader import DataLoader

class Downstream():
    def __init__(self, model):
        self.model = model
        self.fs_params = model.fs_params
        self.device = model.training_params['device']
        self.ds_model = model.vaeconfig.setup_downstream(self.fs_params).to(self.device)

        self.get_data = self.model.vaeconfig.get_data
        self.get_z = self.model.vaeconfig.get_z
        self.get_label = self.model.vaeconfig.get_label
  
    def save_model(self, path):
        print("Saving model to {}".format(path))
        torch.save(self.ds_model.state_dict(), path)

    def load_model(self, path):
        print("Loading model from {}".format(path))
        self.ds_model.load_state_dict(torch.load(path, map_location=self.device))
        print("Succesfully loaded model!")

    def train(self):
        print("Starting training")
        print("Setup dataloaders...")
        train_loader, train_dataset = DataLoader(self.model.dataset_params_train).get_loader()
        test_loader, test_dataset = DataLoader(self.model.dataset_params_test).get_loader()
        print("Dataloaders up and running!")
        
        new_folder = self.model.save_path + \
                '/subset_{}_seed_{}'.format(self.model.dataset_params_train['subset'], self.fs_params['seed'])
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            print("Creating new folder for run {}, seed {}".format(self.model.dataset_params_train['subset'], self.model.fs_params['seed']))
        output_dir = new_folder + '/'
        hk_loss = []
        y_preds = []
        ys = []
        zs = []
        for epoch in range(1, self.fs_params['epochs'] + 1):
            epoch_loss = 0
            for i, data in enumerate(train_loader, 0):
                with torch.no_grad():
                    d, dt = self.get_data(data, self.model.training_params, dt_max=1)
                    _, _, _, _, z = self.get_z(d, self.model.vae)
                    assert z.shape[0] % 2 == 0, "Error something wrong with batch sizes!"
                    split = int(z.shape[0]/2)
                    z = torch.cat((z[:split], z[split:]), axis=1).float()
                y_pred = self.ds_model(z)
                y = self.get_label(data, self.fs_params['data_label'], dt, self.device)
                loss = self.ds_model.loss(y_pred, y)
                y_pred = y_pred.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                self.ds_model.optimizer.zero_grad()
                loss.backward()
                self.ds_model.optimizer.step()
                # Housekeeping
                hk_loss.append(loss.item())
                epoch_loss += loss.item()
            if self.fs_params['save_loss']:
                np.save(output_dir + 'train_loss.npy', hk_loss)
            test_loss = 0.0
            if self.fs_params['testing']:
                # Test ys and y_preds, zs at every epoch or only last?
                y_test, y_pred_test = self.test(test_loader)
                test_loss = np.mean(np.power((y_pred_test-y_test),2))
                y_preds.append(y_pred_test)
                ys.append(y_test)
                zs.append(z.detach().cpu().numpy())
            print('Epoch {}/{} done, epoch loss: {}, test loss: {}'.\
                    format(epoch, self.fs_params['epochs'], epoch_loss/(i+1), test_loss))
        if self.fs_params['save_tests']:
            np.save(output_dir + 'y_preds.npy', np.array(y_preds))
            np.save(output_dir + 'ys.npy', np.array(ys))
            np.save(output_dir + 'zs.npy', np.array(zs))
        if self.fs_params['save_model']:
            self.save_model(output_dir + 'model.mdl')
        train_dataset.memory_loader.stop()
        test_dataset.memory_loader.stop()
        print("Dadaloaders shutdown cleanly")

    def test(self, test_loader):
        for i, data in enumerate(test_loader, 0):
            with torch.no_grad():
                d, dt = self.get_data(data, self.model.training_params, dt_max=1)
                _, _, _, _, z = self.get_z(d, self.model.vae)
                assert z.shape[0] % 2 == 0, "Error something wrong with batch sizes!"
                split = int(z.shape[0]/2)
                z = torch.cat((z[:split], z[split:]), axis=1).float()
                y_pred = self.ds_model(z).cpu().numpy()
                y = self.get_label(data, self.fs_params['data_label'], dt, 'cpu').cpu().numpy()
            if 'y_preds' not in locals():
                y_preds = y_pred
            else:
                y_preds = np.concatenate((y_preds, y_pred), axis=0)
            if 'ys' not in locals():
                ys = y
            else:
                ys = np.concatenate((ys, y), axis=0)
        return ys, y_preds
