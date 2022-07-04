
import numpy as np
import librosa
import librosa.display
import math
import torch
import matplotlib.pyplot as plt

class Speech_Kit():
    def __init__(self, mode='constant', length=0, feature_dim=-1, pad_value=0):
        self.pad_value = pad_value
        self.t2 = length
        self.f2 = feature_dim
        self.mode = mode

    def pad_input(self, x: np.ndarray):
        '''
        input shape: t, f
        mode: constant or repeat, repeat only performs on time axis
        output shape: t2, f2
        '''
        t, f = x.shape
        if self.f2 > 0:
            x = np.pad(x, ((0, 0), (0, self.f2-f)), 'constant', constant_values=(self.pad_value, self.pad_value)) if self.f2>f else x[:, :self.f2]

        if self.t2>t:
            if self.mode=='constant':
                x = np.pad(x, ((0,self.t2-t), (0,0)), 'constant', constant_values=(self.pad_value, self.pad_value))
            elif self.mode=='repeat':
                time = math.ceil(self.t2/t)
                x = np.tile(x, (time, 1))
                x = x[:self.t2]
            else:
                raise ValueError(f'Unknown pad mode:{self.mode}')
        else:
            x = x[:self.t2]

        return torch.from_numpy(x)

def get_D_P(M):
    '''
    calculate db-based spectrogram and power spectrogram from magnitude spectrogram.
    '''
    D = librosa.amplitude_to_db(M, ref=np.max)
    P = M ** 2
    
    return D, P

def plot_time_spec(wavfile):
    '''
    plot waveform in the time domain and plot db-based spectrogram in linear and logarithmic scale.
    
    retrun: Matplotlib Figure
    '''
    y, sr = librosa.load(wavfile, sr=None)

    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(12,10))
    wav_name = wavfile.split('/')[-1].split('.')[0]
    plt.suptitle(wav_name, fontsize=22)

    img1 = librosa.display.waveshow(y, sr=sr, ax=ax[0], x_axis='time')
    ax[0].set(title='wav show')

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)    # 相对于峰值幅度(np.max)计算dB(log谱)
    img2 = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, ax=ax[1], cmap='gray_r')
    ax[1].set(title='linear scale')

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img3 = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax[2], cmap='gray_r')
    ax[2].set(title='log scale')

    return fig