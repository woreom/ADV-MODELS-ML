import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

import custom_transforms as CT
import torchvision.transforms as T
from model.autoencoder import AutoEncoder
from AudioMNIST import AudioMNIST
from utils import save_model, load_model

if __name__ == '__main__':
    EPOCHS=64
    SAMPLE_RATE = 22050
    N_FFT=512
    HOP_LENGTH=256
    N_MELS = 64
    LEARNING_RATE = 0.0005
    dataset = AudioMNIST(
        root='./AudioMNIST/data', 
        target_sample_rate=SAMPLE_RATE,
        transform=T.Compose([
            CT.TrimSilence(15),
            CT.FixLength(SAMPLE_RATE),
            CT.MelSpectrogram(SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH),
            CT.AmplitudeToDB()
        ]),
        normalize=True
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size = 64,
        shuffle=True,
        num_workers=32
    )

    model = AutoEncoder(learning_rate=0.0005)
    trainer = pl.Trainer(max_epochs=EPOCHS)
    trainer.fit(model, train_dataloader)
    
    torch.save(model, "model.pt")
