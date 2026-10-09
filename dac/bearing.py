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

    cc = np.fft.irfft(dial_gaps, config.N_FFT*config.UPSAMPLE)
    max_shift = int(np.ceil(config.MAX_LAG_SAMPLES*config.UPSAMPLE))

    cc = np.concatenate((cc[-max_shift:], cc[0:max_shift+1]))
    lags = np.arange(len(cc))
    lags = ((lags - max_shift) / config.UPSAMPLE)

    return cc, lags

def pair_delay_s(x_i, x_j):
    cc, lags = gcc_phat(x_i, x_j)
    idxMax = np.argmax(cc)
    return (lags[idxMax] / config.FS_HZ)