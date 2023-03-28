import time
import torch
import torch.optim as optim
# Housekeeping
import numpy as np
import cv2
from parallel_dataloader.dataloader import DataLoader

class VAE(torch.nn.Module):
    def __init__(self, model, vaeconfig, net):
        super(VAE, self).__init__()
        vc = vaeconfig # This is later used to access the experiment specific loss
                # and housekeeping functions specified in the network/train_functions.py
                # of each experiment
        self.model = model
        self.net = net
        self.optimizer = optim.Adam(self.parameters(), \
                lr=self.model.model_params['learning_rate'])

        self.get_data = vc.get_data
        self.get_z = vc.get_z
        self.loss_function = vc.loss_function
        self.pixel_diff = vc.pixel_diff
        self.plot_batch = vc.plot_batch

    def save_model(self, epoch):
        # Set model file
        self.model.model_file = self.state_dict()
        self.model.epoch = epoch
        # Update pkl file
        self.model.save_model()

    def load_model(self):
        self.load_state_dict(self.model.model_file)

    def train(self):
        # Setup dataloaders
        print("Starting training!")
        print("Setup dataloaders...")
        train_loader, train_dataset = DataLoader(self.model.dataset_params_train).get_loader()
        test_loader, test_dataset = DataLoader(self.model.dataset_params_test).get_loader()
        print("Dataloaders up and running!")
        
        if self.model.epoch + 1 >= self.model.model_params['train_epochs'] + 1:
            print("Model cannot continue, set a higher train epochs variable than load epoch")
            print("Training was not resumed, exiting ...")
        else:
            for epoch in range(self.model.epoch + 1, self.model.model_params['train_epochs'] + 1):
                epoch_loss = 0
                for i, data in enumerate(train_loader, 0):
                    data, dt = self.get_data(data, self.model.training_params, \
                            self.model.model_params['dt_max'])
                    z = self.get_z(data, self)
                    loss, hk_data = self.loss_function(self.model, z, dt)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    self.model.housekeeping.add('batch_train_loss_rec', hk_data[0])
                    self.model.housekeeping.add('batch_train_loss_kld', hk_data[1])
                    self.model.housekeeping.add('batch_train_loss_slow', hk_data[2])
                epoch_loss = epoch_loss / len(train_loader) 
                if self.model.training_params['verbose']:
                    print('Epoch {} done, epoch loss: {}'.format(epoch, epoch_loss))
                last = False
                if epoch == self.model.model_params['train_epochs'] - 1:
                    last = True
                if not last:
                    # Save the model at the end of every epoch
                    # Save housekeeping
                    self.model.housekeeping.add('epoch_train_loss', epoch_loss)
                    # Run test on test dataset at the end of each epoch
                    if self.model.training_params['testing']:
                        pixel_diff = self.test(test_loader, epoch)
                        self.model.housekeeping.add('epoch_test_pixel_diff', pixel_diff)

                    # This has to be last because it is also saving all the other things set to pkl
                    self.save_model(epoch)

            # Save housekeeping
            self.model.housekeeping.add('epoch_train_loss', epoch_loss)
            if self.model.training_params['testing']:
                pixel_diff = self.test(test_loader, epoch)
                self.model.housekeeping.add('epoch_test_pixel_diff', pixel_diff)
            # Do the final save of model and testing, also always do that last
            self.save_model(epoch)
        # Shutdown dataloaders
        print("Training done, shutting down dataloaders...")
        train_dataset.memory_loader.stop()
        test_dataset.memory_loader.stop()
        i = 0
        while test_dataset.memory_loader.is_alive() or \
                train_dataset.memory_loader.is_alive():
            print(".")
            time.sleep(0.1)
            i += 1
            if i > 100:
                print("Could not sutdown dataloader!")
                exit()
        print("Dataloaders shutdown cleanly!")

    def test(self, test_loader, epoch):
        test_pixel_diff = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                data, dt = self.get_data(data, self.model.training_params, 1)
                o, o_pred, _, _, _ = self.get_z(data, self)
                o_pred = torch.sigmoid(o_pred)
                split = int(o.shape[0]/2)
                o_prev_true = o[:split]
                o_prev_pred = o_pred[:split]
                o_now_true = o[split:]
                o_now_pred = o_pred[split:]
                # Compute Pixel Difference between pred and the actual next
                test_pixel_diff += 0.5 * (self.pixel_diff(o_prev_pred.cpu().numpy(), \
                        o_prev_true.cpu().numpy()) + \
                        self.pixel_diff(o_now_pred.cpu().numpy(), \
                        o_now_true.cpu().numpy()))
                # Save one batch at the beginning of each test
                if i == 0 and self.model.training_params['save_reconstructions']:
                    comp = self.plot_batch(o_prev_true, o_prev_pred, num=8, \
                            device=self.model.training_params['device'])
                    self.model.test_reconstructions.append(comp)
                    self.model.save_model()
                # Show all batches if the show_test flag is set to True
                if i == 0 and self.model.training_params['show_reconstructions']:
                    comp = self.plot_batch(o_prev_true, o_prev_pred, num=8, \
                            device=self.model.training_params['device'])
                    cv2.imshow("Reconstructions epoch {}".format(epoch), comp)
                    cv2.waitKey(100)
        # Normalize the pixel diff error according to batchsize
        test_pixel_diff /= len(test_loader)
        if self.model.training_params['verbose']:
            print('Test pixel diff: {}'.format(test_pixel_diff))
        return test_pixel_diff
