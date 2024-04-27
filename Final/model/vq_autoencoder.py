import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from torch import optim

from . import AutoEncoder
from .codebook import VQCodebook

class VQAutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate, normalize=True, codebook_size=512, autoencoder=None):
        super(VQAutoEncoder, self).__init__()
        if autoencoder is None:
            autoencoder = AutoEncoder(learning_rate, normalize)
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.codebook_dim = autoencoder.encoder.codebook_dim
        self.codebook = VQCodebook(self.codebook_dim, codebook_size)
        self.learning_rate = learning_rate
        self.normalize = normalize
    
    def forward(self, x):
        if self.normalize:
            x, min, max = self._normalize(x)
        z_e = self.encoder(x)
        z_q, _, vq_loss = self.codebook(z_e)
        reconstruction = self.decoder(z_q)
        loss = self._reconstruction_loss(x, reconstruction) + vq_loss
        if self.normalize:
            reconstruction = self._denormalize(reconstruction, min, max)
        return reconstruction, loss
        
    def _reconstruction_loss(self, original, reconstruction):
        return F.mse_loss(reconstruction, original, reduction='mean')
        
    def training_step(self, batch, batch_index):
        x, _ = batch
        _, loss = self.forward(x)
        self.log('loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_index):
        x, _ = batch
        _, loss = self.forward(x)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        return optimizer
        
    @torch.no_grad
    def _normalize(self, x):
        x_min = x.min()
        x_max = x.max()
        return ((x - x_min) / (x_max - x_min)), x_min, x_max
        
    @torch.no_grad
    def _denormalize(self, x_norm, min, max):
        return x_norm * (max - min) + min
