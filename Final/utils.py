import torch
import torchvision.transforms as T

import custom_transforms as CT
from AudioMNIST import AudioMNIST

def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model, save_path):
    """
    Save a model in TorchScript format
    """
    model_scripted = model.to_torchscript(method="trace")
    model_scripted.save(save_path)

def load_model(file_path):
    """
    Load a model from TorchScript format
    """
    model = torch.jit.load(file_path)
    return model.eval()

def spectrogram_to_audio(spectrogram):
    spectrogram = librosa.db_to_amplitude(spectrogram)
    return librosa.feature.inverse.mel_to_audio(spectrogram, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)

def get_dataloaders(root='./AudioMNIST/data', normalize=False, sample_rate=22050, n_fft=512, hop_length=256, n_mels = 64):
    dataset = AudioMNIST(
        root,
        target_sample_rate=sample_rate,
        transform=T.Compose([
            CT.FixLength(sample_rate),
            CT.MelSpectrogram(sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length),
            CT.AmplitudeToDB()
        ]),
        normalize=normalize
    )
    total_dataset_length = len(dataset)
    validation_dataset_length = int(total_dataset_length * 0.1)
    train_dataset_length = total_dataset_length - validation_dataset_length
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_dataset_length, validation_dataset_length])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 64,
        shuffle=True,
        num_workers=4
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size = 1,
        shuffle=False,
        num_workers=4
    )
    return train_loader, validation_loader