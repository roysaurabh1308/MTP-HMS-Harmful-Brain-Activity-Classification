import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res
# detail 
class Wave_Net(nn.Module):
    def __init__(self, inch=1, kernel_size=3):
        super().__init__()
        self.wave_net = nn.Sequential(
            Wave_Block(inch, 8, 12, kernel_size),
            Wave_Block(8, 16, 8, kernel_size),
            Wave_Block(16, 32, 4, kernel_size),
            Wave_Block(32, 64, 1, kernel_size)
        )
        
        self.fc = nn.Linear(64*4, 512)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x1 = self.wave_net(x[:,0:1,:])
        x1 = torch.mean(x1, -1)
        
        x2 = self.wave_net(x[:,1:2,:])
        x2 = torch.mean(x2, -1)
        z1 = torch.mean(torch.stack([x1, x2], dim=0), dim=0)

        x1 = self.wave_net(x[:,2:3,:])
        x1 = torch.mean(x1, -1)
        x2 = self.wave_net(x[:,3:4,:])
        x2 = torch.mean(x2, -1)
        z2 = torch.mean(torch.stack([x1, x2], dim=0), dim=0)

        x1 = self.wave_net(x[:,4:5,:])
        x1 = torch.mean(x1, -1)
        x2 = self.wave_net(x[:,5:6,:])
        x2 = torch.mean(x2, -1)
        z3 = torch.mean(torch.stack([x1, x2], dim=0), dim=0)

        x1 = self.wave_net(x[:,6:7,:])
        x1 = torch.mean(x1, -1)
        x2 = self.wave_net(x[:,7:8,:])
        x2 = torch.mean(x2, -1)
        z4 = torch.mean(torch.stack([x1, x2], dim=0), dim=0)
        y = torch.cat((z1, z2, z3, z4), 1)
        
        y = self.fc(y)
        return y
    
if __name__ == '__main__':
    model = Wave_Net().cuda()
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    x = torch.rand((2, 10000, 8)).cuda()
    x = x[:, ::5,:]
    print(x.shape)
    print(model(x).shape)