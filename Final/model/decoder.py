from torch import nn

from itertools import chain

class Decoder(nn.Module):
    def __init__(self, codebook_dim, hidden_sizes, output_channels):
        super(Decoder, self).__init__()
        
        layers = []
        layer_sizes = list(chain([codebook_dim], hidden_sizes, [output_channels]))
        
        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.ConvTranspose2d(input_size, output_size, 3))
            layers.append(nn.BatchNorm2d(output_size))
            layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        z_e = self.encoder(x)
        return z_e