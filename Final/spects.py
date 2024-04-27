import librosa
import matplotlib.pyplot as plt
import torch
from utils import *

vae_model = torch.load('model.pt')
vq_model = torch.load('vqvae.pt')

spectrograms = []
train, val = get_dataloaders()
for idx, (x, _) in enumerate(val):
    spectrograms.append(x)
    if (idx == 3):
        break
plt.figure(figsize=(14, 10))
for index, spectrogram in enumerate(spectrograms):
    spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)
    reconstruction, _ = vae_model(spectrogram_tensor.to(device()))
    # Convert tensors to numpy arrays for plotting with librosa
    original_spec = spectrogram_tensor.numpy()[0]
    reconstructed_spec = reconstruction.detach().squeeze().cpu().numpy()
    reconstructed_spec_q =  vq_model(spectrogram_tensor.to(device()))[0].detach().squeeze().cpu().numpy()

    print(reconstructed_spec.shape)
    
    # Plot original spectrogram and its reconstruction
    
    plt.subplot(3, 4, index + 1)
    librosa.display.specshow(original_spec[0], sr=22050, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Spectrogram')
    
    plt.subplot(3, 4, 5 + index)
    librosa.display.specshow(reconstructed_spec, sr=22050, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed Spectrogram (AE)')
    plt.subplot(3, 4, 9 + index)
    librosa.display.specshow(reconstructed_spec, sr=22050, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed Spectrogram (VQ-VAE)')
    
plt.tight_layout()
plt.savefig('spects.png')