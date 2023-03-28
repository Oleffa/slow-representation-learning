import torch
import torch.optim as optim
import torch.nn.functional as F

class DownstreamFC(torch.nn.Module):
    def __init__(self, layers):
        super(DownstreamFC, self).__init__()
        self.linear1 = torch.nn.Linear(layers[0], layers[1])
        self.linear2 = torch.nn.Linear(layers[1], layers[2])
        self.linear3 = torch.nn.Linear(layers[2], layers[3])
        self.sig = torch.nn.Sigmoid()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        # Flatten tensor input
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.sig(self.linear3(x))
