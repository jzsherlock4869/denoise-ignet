import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.orthogonal_(m.weight)
        print('init weight for conv layer')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class DnCNNEncoder(nn.Module):
    def __init__(self, depth=10, n_ft=64, in_chs=1, out_ft=32, res=False):
        super(DnCNNEncoder, self).__init__()
        self.use_res = res
        self.pre_conv = nn.Conv2d(in_channels=in_chs, out_channels=n_ft, kernel_size=3, padding=1, bias=True)
        layers = []
        layers.append(nn.Conv2d(in_channels=n_ft, out_channels=n_ft, \
                                kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_ft, out_channels=n_ft, \
                                kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(n_ft, eps=0.0001, momentum = 0.05))
            layers.append(nn.ReLU(inplace=True))
        
        self.last_conv = nn.Conv2d(in_channels=n_ft, out_channels=out_ft, \
                                kernel_size=3, padding=1, bias=False)
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        if not self.use_res:
            x = self.pre_conv(x)
            x = self.encoder(x)
            out = self.last_conv(x)
        else:
            x = self.pre_conv(x)
            x = self.encoder(x) + x
            out = self.last_conv(x)
        return out

if __name__ == "__main__":
    dncnn_encoder = DnCNNEncoder(depth=10, n_ft=64, in_chs=1, out_ft=64)
    x_in = torch.rand(1, 1, 64, 64)
    out = dncnn_encoder(x_in)
    print(dncnn_encoder)
    print(x_in.shape, out.shape)
