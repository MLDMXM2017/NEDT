
import torch
import torch.nn as nn
import torch.nn.init as init


class Gate(nn.Module):
    def __init__(self, device, data_info):
        super(Gate, self).__init__()
        self.device = device
        self.use_cuda = False
        self.dtype = None

        self.layer1 = nn.Linear(data_info[2], 1, bias=False)
        init.constant_(self.layer1.weight, 1. / self.layer1.weight.shape[1])

        self.layer2 = nn.Linear(data_info[1], 1, bias=False)
        self.layer2.weight.data.normal_(0, 0.25)

        self.sigmoid = nn.Sigmoid()

    def cuda(self):
        self.use_cuda = True
        self.dtype = torch.cuda.FloatTensor
        super(Gate, self).to(self.device)

    def forward(self, x):
        out1 = self.layer1(x).squeeze(dim=2)
        out2 = self.layer2(out1).squeeze(dim=1)
        out3 = self.sigmoid(out2)
        return out3



