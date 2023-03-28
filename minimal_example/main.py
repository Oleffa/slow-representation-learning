from network import BetaVAE_H
from torchvision import datasets, transforms
import torch
import torch.optim as optim
import cv2
import numpy as np
import sys
sys.path.append('../network/')
from train_functions_global import rec_loss, slowness_gaussian_difference, compute_cross_ent_combined, l1l2_loss

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = 64
    beta = 1.0
    # SlowVAE rate prior parameter
    rate_prior = 0.0
    gamma = 1.0

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    model = BetaVAE_H(z_dim=4, nc=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    epochs = 10
    for e in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Prep data
            # Concatenate the data along the batch dimension to pass through the network
            # This makes no sense because mnist is not a seqential dataset but it shows
            # that you need to pass the data in the format (batch size * 2, feature dim 1 , feature dim 2 ...)
            data, target = data.to(device), target.to(device)
            data = torch.cat((data, data), dim=0)

            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(data.to(device))
            # Reconstructio loss * 2 because we are doing it for each timestep t and t+1
            l_r = 2 * rec_loss(data, x_recon)
            # KL loss like in beta_vae
            [normal_entropy, cross_ent_normal, cross_ent_laplace] = compute_cross_ent_combined(mu, logvar, rate_prior, z_dim=4, device=device)
            l_kl = beta * (cross_ent_normal - normal_entropy)
            # S-VAE
            l_svae = gamma * slowness_gaussian_difference(mu, logvar, prior_scale=1.0, dt=torch.tensor(np.ones(int(data.shape[0]/2))).long().to(device), z_dim=4, device=device)
            loss = l_r + l_kl + l_svae
            loss.backward()
            optimizer.step()
        print('{}/{}, rec: {}, kld: {}, slowness reg: {}'.format(e+1, epochs, l_r.item(), l_kl.item(), l_svae.item()))
        cv2.imshow('test', torch.cat((torch.sigmoid(x_recon[0, 0]),data[0, 0]), dim=0).detach().cpu().numpy())
        cv2.waitKey(1)
        
if __name__ == '__main__':
    main()
