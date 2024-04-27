import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from torch import optim

from .encoder import Encoder
from .decoder import Decoder

class AutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate, normalize=True):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(1, [512, 256, 128, 64], 32)
        self.decoder = Decoder(32, [64, 128, 256, 512], 1)
        self.learning_rate = learning_rate
        self.normalize = normalize
    def forward(self, x):
        if self.normalize:
            x, min, max = self._normalize(x)
        z_e = self.encoder(x)
        reconstruction = self.decoder(z_e)
        loss = self._reconstruction_loss(x, reconstruction)
        if self.normalize:
            reconstruction = self._denormalize(reconstruction, min, max)
        return reconstruction, loss
        
    def _reconstruction_loss(self, original, reconstruction):
        return F.mse_loss(reconstruction, original, reduction='mean')
        
    def training_step(self, batch, batch_index):
        x, _ = batch
        _, reconstruction_loss = self.forward(x)
        self.log('recon_loss', reconstruction_loss, prog_bar=True)
        return reconstruction_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
        
    @torch.no_grad()
    def _normalize(self, x):
        #import ipdb; ipdb.set_trace()
        x_min = x.min()
        x_max = x.max()
        return ((x - x_min) / (x_max - x_min)), x_min, x_max
    @torch.no_grad()
    def _denormalize(self, x_norm, min, max):
        return x_norm * (max - min) + min
