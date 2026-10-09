"""Audio sources that replay files or synthetic scenes (Sections 5.1, 6.3). Owner: Srihaas.

Every source is an iterator of (t_s, block): block is float32 (BLOCK_N, 4) in mic order,
blocks start HOP_N samples apart, and t_s is the block's first sample / FS_HZ.
"""
import time

import numpy as np
import soundfile as sf

from dac import config


def _blocks(data, realtime=False):
    """Yield (t_s, block) from (n, 4) mic-order data. A partial final block is dropped."""
    data = np.asarray(data, dtype=np.float32)
    for start in range(0, len(data) - config.BLOCK_N + 1, config.HOP_N):
        if realtime:
            time.sleep(config.HOP_N / config.FS_HZ)
        yield start / config.FS_HZ, data[start:start + config.BLOCK_N]


class WavSource:
    """Replay a 16 kHz WAV: 6-channel raw (reordered by CHANNEL_ORDER) or 4-channel mic order."""

    def __init__(self, path, realtime=False):
        data, fs = sf.read(path, dtype='float32', always_2d=True)
        if fs != config.FS_HZ:
            raise ValueError(f'{path}: sample rate {fs} Hz, expected {config.FS_HZ} Hz')
        if data.shape[1] == config.N_RAW_CHANNELS:
            data = data[:, config.CHANNEL_ORDER]
        elif data.shape[1] != config.N_MICS:
            raise ValueError(f'{path}: {data.shape[1]} channels, expected '
                             f'{config.N_RAW_CHANNELS} (raw) or {config.N_MICS} (mic order)')
        self._data = data
        self._realtime = realtime

    def __iter__(self):
        return _blocks(self._data, self._realtime)


class SyntheticSource:
    """Render a synth.Scenario once and replay it like a WAV file."""

    def __init__(self, scenario):
        self._scenario = scenario
        self._data = np.asarray(scenario.render(), dtype=np.float32)

    def __iter__(self):
        return _blocks(self._data)

    def truth(self, t_s):
        """dict with bearing_deg, dist_m, approaching at time t_s."""
        return self._scenario.truth(t_s)
