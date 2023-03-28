import numpy as np
import torch.nn.functional as F
import torch 

def compute_ent_normal(logvar):
    return 0.5 * (logvar + np.log(2 * np.pi * np.e))

def compute_cross_ent_normal(mu, logvar):
    return 0.5 * (mu**2 + torch.exp(logvar)) + np.log(np.sqrt(2 * np.pi))

def rec_loss(o, o_pred):
    bs = o_pred.shape[0]
    # Here we use sum and divide by batch size because averaging is a bit tricky:
    # - When using mean over all pixels, the loss scale would depend on the img size
    # - Average pixel value is not that meaningful, but rather average picture value is
    #print(o.min(), o.max(), o.mean(), o.std())
    #print(o_pred.min(), o_pred.max(), o_pred.mean(), o_pred.std())
    a = F.binary_cross_entropy_with_logits(o_pred.view(o.shape), o, reduction='sum').div(bs)
    return a

# ==================================
# L1 and L2
# ==================================

def l1l2_loss(mu, norm='l1'):
    assert mu.shape[0]%2 == 0, "Warning something went wrong when loading the data"
    split = int(mu.shape[0]/2)

    mu0 = mu[:split]
    mu1 = mu[split:]

    if norm == 'l1':
        loss = torch.nn.L1Loss()
        return loss(mu1, mu0)
    elif norm == 'l2':
        loss = torch.nn.MSELoss()
        return loss(mu1, mu0)
    else:
        print("Error, expected l1 or l2 as norm")
        exit()

# ==================================
# S-VAE
# ==================================

def slowness_gaussian_difference(mu, logvar, prior_scale, dt, z_dim, device):
    assert mu.shape[0]%2 == 0, "Warning something went wrong when loading the data"
    split = int(mu.shape[0]/2)

    mu0 = mu[:split]
    mu1 = mu[split:]
    sigma_0 = torch.sqrt(torch.exp(logvar[:split]))
    sigma_1 = torch.sqrt(torch.exp(logvar[split:]))
    dt = dt.unsqueeze(1).repeat(1, z_dim)

    KLD = - torch.log((sigma_0 + sigma_1)/(torch.sqrt(dt)*prior_scale)) + \
            (torch.pow(sigma_0 + sigma_1, 2) + torch.pow(mu1-mu0, 2))/(2*dt*np.power(prior_scale,2)) + 0.5
    return KLD.sum(1).mean(0)

# ==================================
# SlowVAE
# ==================================

def compute_cross_ent_laplace(mean, logvar, rate_prior, z_dim, device):
    normal_dist = torch.distributions.normal.Normal(
            torch.zeros(z_dim, device=device),
            torch.ones(z_dim, device=device))
    var = torch.exp(logvar)
    sigma = torch.sqrt(var)
    ce = - torch.log(rate_prior / 2) + rate_prior * sigma *\
         np.sqrt(2 / np.pi) * torch.exp(- mean**2 / (2 * var)) -\
         rate_prior * mean * (
                 1 - 2 * normal_dist.cdf(mean / sigma))
    return ce

def compute_cross_ent_combined(mu, logvar, rate_prior, z_dim, device):
    rate_prior = torch.ones(1, requires_grad=False, device=device) * rate_prior
    normal_entropy = compute_ent_normal(logvar)
    cross_ent_normal = compute_cross_ent_normal(mu, logvar)

    assert mu.shape[0]%2 == 0, "Warning something went wrong when loading the data"
    split = int(mu.shape[0]/2)

    mu0 = mu[:split]
    mu1 = mu[split:]
    logvar0 = logvar[:split]
    logvar1 = logvar[split:]
    rate_prior0 = rate_prior
    rate_prior1 = rate_prior

    cross_ent_laplace = (compute_cross_ent_laplace(mu0 - mu1, logvar0, rate_prior0, z_dim, device) \
            + compute_cross_ent_laplace(mu1 - mu0, logvar1, rate_prior1, z_dim, device))
    return [x.sum(1).mean(0) for x in [normal_entropy, cross_ent_normal, cross_ent_laplace]]
