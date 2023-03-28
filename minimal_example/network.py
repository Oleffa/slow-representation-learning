import torch
import torch.nn.init as init
import torch.nn as nn
from torch.autograd import Variable

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class PrintLayer(nn.ModuleList):
    def __init__(self):
        super(PrintLayer, self).__init__()
    def forward(self, x):
        #print(x.shape)
        return x

class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""
    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 8, 3, 2, 1),          # B,  32, 32, 32
            PrintLayer(),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, 2, 1),          # B,  32, 16, 16
            PrintLayer(),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, 2, 1),          # B,  64,  8,  8
            PrintLayer(),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 2, 1),          # B,  64,  4,  4
            PrintLayer(),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 2, 1),            # B, 256,  1,  1
            PrintLayer(),
            nn.ReLU(True),
            View((-1, 32*1*1)),                 # B, 256
            nn.Linear(32, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 32),               # B, 256
            PrintLayer(),
            View((-1, 32, 1, 1)),               # B, 256,  1,  1
            PrintLayer(),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 2, 1),      # B,  64,  4,  4
            PrintLayer(),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  64,  8,  8
            PrintLayer(),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, 2, 1), # B,  32, 16, 16
            PrintLayer(),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1), # B,  32, 32, 32
            PrintLayer(),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, nc, 4, 2, 1),  # B, nc, 64, 64
            PrintLayer(),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar, z

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
