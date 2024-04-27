import torch

from ..model import Encoder

def test_encoder():
    encoder = Encoder(3, [512, 256, 128], 64)
    x = torch.randn(1, 3, 32, 32)
    y = encoder(x)
    print(y.shape)
    assert(y.shape[1] == 64)

    