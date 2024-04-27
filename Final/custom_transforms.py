import librosa
import torch.nn as nn

import numpy as np


class TrimSilence(nn.Module):
    def __init__(self, top_db):
        super(TrimSilence, self).__init__()
        self.top_db = top_db
    def forward(self, waveform):
        waveform, _ = librosa.effects.trim(waveform, top_db = self.top_db)
        return waveform

class FixLength(nn.Module):
    def __init__(self, length):
        super(FixLength, self).__init__()
        self.length = length
    def forward(self, waveform):
        return librosa.util.fix_length(waveform, size=self.length)

class MelSpectrogram(nn.Module):
    def __init__(self, sample_rate, add_channel_dim=True, **melspectrogram_args):
        super(MelSpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.add_channel_dim = add_channel_dim
        self.melspectrogram_args = melspectrogram_args
    def forward(self, waveform):
        spectrogram = librosa.feature.melspectrogram(
                    y=waveform, 
                    sr=self.sample_rate,
                    **self.melspectrogram_args)
        if self.add_channel_dim:
            spectrogram = spectrogram[np.newaxis, ...]
        return spectrogram
        
    
class STFT(nn.Module):
    def __init__(self, **stft_args):
        super(STFT, self).__init__()
        self.stft_args = stft_args
    def forward(self, waveform):
        stft = librosa.stft(
                    y=waveform,
                    **self.stft_args)
        return np.abs(stft)

class FFT(nn.Module):
    def __init__(self, **fft_args):
        super(FFT, self).__init__()
        self.fft_args = fft_args
# FFT -> power spectrum
    def forward(self, waveform):

        # perform Fourier transform

        fft = np.fft.fft(waveform, **self.fft_args)
        fft = np.fft.fftshift(fft)
        spectrum = np.abs(fft)
        
        # take half of the spectrum and frequency

        left_spectrum = spectrum[:int(len(spectrum)/2)]
        return spectrum



class MinMaxNormalization(nn.Module):
    def __init__(self):
        super(MinMaxNormalization, self).__init__()
    def forward(self, x):
        x_min = x.min()
        x_max = x.max()
        return ((x - x_min) / (x_max - x_min)), x_min, x_max

class AmplitudeToDB(nn.Module):
    def __init__(self, ref=1.0):
        super(AmplitudeToDB, self).__init__()
        self.ref = ref
    def forward(self, waveform):
        return librosa.amplitude_to_db(waveform, ref=self.ref)


class PowerToDB(nn.Module):
    def __init__(self, ref=1.0):
        super(PowerToDB, self).__init__()
        self.ref = ref
    def forward(self, waveform):
        return librosa.power_to_db(waveform, ref=self.ref)
