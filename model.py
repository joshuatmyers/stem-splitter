import torch
import torch.nn as nn
import torchaudio
import sklearn
import pandas as pd
import numpy as np
import math
from typing import List, Tuple

"""

I plan to implement the Band Split Rope Transformer from the following paper https://arxiv.org/pdf/2309.02612

"""

### HYPERPARAMETERS ###
# musdb18hq sample rate
SR = 44_100    
# Hann window length - 46ms
WIN = 2048      
# 10 ms hop -> 441 samples
HOP = int(0.010 * SR)     
# reuse this tensor on the same device   
WINDOW = torch.hann_window(WIN) 

# windows for the 5-resolution spectrogram loss
# defined in secttion 2:
MR_WINLIST = [4096, 2048, 1024, 512, 256]  
# fixed hop size
MR_HOP = 147          

### validation ###
# this class will be used to validate each intermediate step of the pipeline
class Validation():
    def __init__(self):
        pass

### input ###


### stft ###
def stft_forward(wav):
    """
    Input: (B, C, L) - batch, channels, samples
    Output: (B, C, T, F) - batch, channels, time, frequency
    """
    # torch.stft treats the last dimension as time, so we flatten batch+ch first
    B, C, L = wav.shape
    flat = wav.reshape(B * C, L)

    # thankfully pytorch already has a built-in STFT function
    spec = torch.stft(
        flat,
        n_fft = WIN,
        hop_length = HOP,
        window = WINDOW.to(wav.device, wav.dtype),
        center = True,
        pad_mode = "reflect",
        return_complex = True,
        # keep energy linear; let the net learn scaling
        normalized = False,
    )

    # (B*C, F, T) -> (B, C, T, F)
    X = spec.transpose(1, 2).reshape(B, C, spec.shape[-1], spec.shape[-2])
    return X

### inverse stft ###
def istft_inverse(X, length):
    """
    (B,C,T,F) complex -> (B,C,L) real using the same params as stft_forward
    """
    B, C, T, F = X.shape
    X_flat = X.transpose(2, 3).reshape(B * C, F, T)

    wav = torch.istft(
        X_flat,
        n_fft=WIN,
        hop_length=HOP,
        window=WINDOW.to(X.device, X.dtype),
        center=True,
        length=length,
    )
    return wav.reshape(B, C, -1)

### band split module ###

class BandSplit(nn.Module):
    """
    X (B,C,T,F)  ->  H0 (B,T,N,D)
    """
    RANGES = [
        (1_000, 2),
        (2_000, 4),
        (4_000, 12),
        (8_000, 24),
        (16_000, 48),
    ]

    def __init__(self, sr: int = 44_100, n_fft: int = 2_048, d_model: int = 384):
        super().__init__()
        hz_per_bin = sr / n_fft
        max_bin = (n_fft // 2) + 1
        boundaries = [0]
        for hi_hz, bins in self.RANGES:
            hi_bin = math.floor(hi_hz / hz_per_bin)
            while boundaries[-1] + bins <= hi_bin and boundaries[-1] + bins <= max_bin:
                boundaries.append(boundaries[-1] + bins)
        if boundaries[-1] < max_bin:
            # 1024 - 702 = 322
            remaining = max_bin - boundaries[-1]    
            # we need 6 final bands    
            n_extra = 6
            # 322 / 6 → 54
            step = math.ceil(remaining / n_extra)
            b = boundaries[-1]
            while b + step < max_bin:
                b += step
                boundaries.append(b)
            boundaries.append(max_bin)

        self.bands = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        assert len(self.bands) == 62

        # create MLPs for each band
        self.mlps = nn.ModuleList()
        for (l, r) in self.bands:
            fn = r - l
            # (real+imag) * stereo
            in_dim = 4 * fn          
            self.mlps.append(
                nn.Sequential(
                    nn.RMSNorm(in_dim, eps=1e-8),
                    nn.Linear(in_dim, d_model, bias=True),
                )
            )

    def forward(self, X):
        B, C, T, F = X.shape
        # (B,T,C,2,F)
        Xri = torch.view_as_real(X).permute(0, 2, 1, 4, 3) 

        outs = []
        for (l, r), mlp in zip(self.bands, self.mlps):
            # (B,T,4*Fn)
            band = Xri[..., l:r].reshape(B, T, -1) 
            # (B,T,D)         
            outs.append(mlp(band))   
        # (B,T,N,D)                       
        return torch.stack(outs, dim=2)                     


# RoPE transformer blocks

# multi-band mask estimation

# istft