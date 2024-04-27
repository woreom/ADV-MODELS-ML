from torch import nn
from itertools import chain

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_sizes, codebook_dim):
        super(Encoder, self).__init__()
        
        layers = []
        layer_sizes = list(chain([input_channels], hidden_sizes, [codebook_dim]))
        
        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Conv2d(input_size, output_size, 3))
            layers.append(nn.BatchNorm2d(output_size))
            layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
        self.codebook_dim = codebook_dim

    def forward(self, x):
        z_e = self.encoder(x)
        return z_e

if __name__ == '__main__':
    import torch
    x = torch.randn(1, 3, 32, 32)
    print(f"x.shape: {x.shape}")
    x_enc = Encoder(3, [512, 64, 32], 64)(x)
    print(f"x_enc.shape: {x_enc.shape}")