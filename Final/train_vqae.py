import torch
import lightning.pytorch as pl
from model.vq_autoencoder import VQAutoEncoder
from utils import *

def train(save_path):
    EPOCHS=50
    LEARNING_RATE = 0.0005
    train_loader, validation_loader = get_dataloaders()
    vqae = VQAutoEncoder(LEARNING_RATE)
    trainer = pl.Trainer(max_epochs=EPOCHS)
    trainer.fit(vqae, train_loader, validation_loader)
    torch.save(vqae, save_path)

if __name__ == '__main__':
    train('vqvae_512ncw_32dim50.pt')
