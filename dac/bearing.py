import numpy as np
from dac import config

def gcc_phat(x_i, x_j):

    dubX = x_i * np.hanning(len(x_i))
    dubY = x_j * np.hanning(len(x_j))

    x_fft = np.fft.rfft(dubX, config.N_FFT)
    y_fft = np.fft.rfft(dubY, config.N_FFT)

    dial_gaps = np.conj(x_fft) * y_fft

    dial_gaps /= np.abs(dial_gaps) + 1e-12

    position_hz = np.fft.rfftfreq(config.N_FFT, 1 / config.FS_HZ)
    mask = (position_hz < config.BAND_HZ[0]) | (position_hz > config.BAND_HZ[1])
    dial_gaps[mask] = 0

    return dial_gaps