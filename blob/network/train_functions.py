import numpy as np
import torch.nn.functional as F
import torch 
import sys
import cv2
from network.train_functions_global import rec_loss, slowness_gaussian_difference, \
        compute_cross_ent_combined, l1l2_loss

# Preprocess data from the dataset def get_data(data, training_params):
def get_data(data, training_params, dt_max):
    data = data['img']
    dts = []
    for i in range(data.shape[0]):
        dt = np.random.choice(np.arange(1, dt_max+1,1))
        dts.append(dt)
        if 'o_prev' not in locals():
            o_prev = data[i,0,:,:].unsqueeze(0)
            o_now = data[i,dt,:,:].unsqueeze(0)
        else:
            o_prev = torch.cat((o_prev, data[i,0,:,:].unsqueeze(0)), dim=0)
            o_now = torch.cat((o_now, data[i,dt,:,:].unsqueeze(0)), dim=0)

    o = torch.cat((o_prev, o_now), dim=0).float()/255.0
    return o, torch.tensor(np.array(dts).astype(float), device=training_params['device'])

# Return the labele for the downstream task
def get_label(data, label, dt, device):
    dt = dt.long()
    if label == 'pos':
        d = data['pos']/100.0
    if label == 'vel':
        d= (data[label]+5.0)/10.0
    labels = []
    for idx, i in enumerate(dt):
        labels.append((d[idx,i]).unsqueeze(0))
    labels = torch.cat(labels, dim=0).float()

    return labels

# Compute z and other stuff by passing the observation through the encoder/decoder
def get_z(o, vae):
    o_pred, mu, logvar, z = vae.net(o.unsqueeze(1), return_z=True)
    return o, o_pred, mu, logvar, z

def loss_function(model, data, dt):
    o, o_pred, mu, logvar, z = data

    rec = 2 * rec_loss(o, o_pred)
    loss = rec

    [normal_entropy, cross_ent_normal, cross_ent_laplace] = \
            compute_cross_ent_combined(mu, logvar, \
            model.model_params['rate_prior'], model.model_params['latent_dim'], \
            model.training_params['device'])
    kl_normal = model.model_params['beta'] * (cross_ent_normal - normal_entropy)
    loss += kl_normal
    if model.method == 'slowvae':
        [normal_entropy, cross_ent_normal, cross_ent_laplace] = \
                compute_cross_ent_combined(mu, logvar, \
                model.model_params['rate_prior'], model.model_params['latent_dim'], \
                model.training_params['device'])
        kl_laplace = model.model_params['gamma'] * (cross_ent_laplace - normal_entropy)
        loss += kl_laplace
        slowness_loss = kl_laplace.item()
    elif model.method == 'svae':
        loss_svae = model.model_params['gamma'] * slowness_gaussian_difference(mu, logvar, \
                model.model_params['rate_prior'], dt, model.model_params['latent_dim'], \
                model.training_params['device'])
        loss += loss_svae
        slowness_loss = loss_svae.item()
    elif model.method == 'l1' or model.method == 'l2':
        norm = model.method
        lp = model.model_params['gamma'] * l1l2_loss(mu, norm)
        loss += lp
        slowness_loss = lp.item()
    elif model.method == 'bvae':
        tmp = torch.Tensor([0.0])
        slowness_loss = tmp.item()

    return loss, (rec.item(), kl_normal.item(), slowness_loss)

def setup_housekeeping():
    hk_loss_rec = []
    hk_loss_kld = []
    hk_loss_slow = []
    return [hk_loss_rec, hk_loss_kld, hk_loss_slow]

def housekeeping(hk_data, hk_lists):
    rec_loss, kld_loss, slow_loss = hk_data
    hk_lists[0].append(rec_loss)
    hk_lists[1].append(kld_loss)
    hk_lists[2].append(slow_loss)
    return hk_lists

def save_housekeeping(output_dir, hk_lists):
    np.save(output_dir+'loss_rec.npy', hk_lists[0])
    np.save(output_dir+'loss_kld.npy', hk_lists[1])
    np.save(output_dir+'loss_slow.npy', hk_lists[2])

def pixel_diff(o_pred, o):
    diff = np.sum(np.abs(o_pred.reshape((o.shape)) - o))
    return diff/o_pred.shape[0]

def plot_batch(o_in, o_pred, num=16, device='cpu'):
    n = min(o_in.size(0), num)
    o_in = o_in[:n].unsqueeze(1).permute(0,2,3,1)
    o_pred = o_pred[:n].permute(0,2,3,1)
    comp = torch.hstack([o_in.view(o_in.shape[0] * o_in.shape[1], o_in.shape[2], o_in.shape[3]), o_pred.view(o_pred.shape[0] * o_pred.shape[1], o_pred.shape[2], o_pred.shape[3])])
    return comp.cpu().numpy()

######################## Non Training related functions #######################

def vis_dataset(dataset, sequences_to_show=None):
    for i, data in enumerate(dataset, 0):
        seq = data['images']
        pos = data['pos']
        vel = data['vel']
        for s, p, v in zip(seq, pos, vel):
            frame = s.cpu().numpy()
            p_ = tuple(p.cpu().numpy().astype(np.int))
            v_ = tuple(v.cpu().numpy().astype(np.int))
            frame = cv2.circle(frame, p_, 2, (0, 0, 0), -1)
            frame = cv2.circle(frame, tuple([x+y for x,y in zip(p_, v_)]), \
                    2, (128, 128, 128), -1)
            cv2.imshow("Visualizing dataset", frame)
            cv2.waitKey(1)
        if sequences_to_show is not None:
            if i > sequences_to_show:
                break

