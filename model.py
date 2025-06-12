import torch
import torch.nn as nn
import torchaudio
import sklearn
import pandas as pd
import numpy as np

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
def stft_forward(wav: torch.Tensor) -> torch.Tensor:
    """
    Input: (B, C, L) - batch, channels, samples
    Output: (B, C, T, F) - batch, channels, time, frequency
    """
    # torch.stft treats the last dimension as time, so we flatten batch+ch first
    B, C, L = wav.shape
    flat = wav.reshape(B * C, L)

    # thankfully pytorch aalready has a built-in STFT function
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

### band split module ###
# complex spectrum

# MLP

# RoPE transformer blocks

# multi-band mask estimation

# istft