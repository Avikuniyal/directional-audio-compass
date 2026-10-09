# tests/test_sources.py  (owner: Srihaas)
import numpy as np
import pytest
import soundfile as sf

from dac import config, sources
from dac.sources import SyntheticSource, WavSource


def _write(path, data, fs=config.FS_HZ):
    sf.write(str(path), data.astype(np.float32), fs, subtype='FLOAT')
    return path


def test_six_channel_reorder(tmp_path):
    raw = np.tile(np.arange(config.N_RAW_CHANNELS, dtype=np.float32) / 10, (config.BLOCK_N, 1))
    blocks = list(WavSource(_write(tmp_path / 'a.wav', raw)))
    assert len(blocks) == 1
    for col, ch in enumerate(config.CHANNEL_ORDER):
        assert np.allclose(blocks[0][1][:, col], ch / 10)


def test_four_channel_passthrough(tmp_path):
    data = np.random.default_rng(0).uniform(-0.5, 0.5, (config.BLOCK_N, 4)).astype(np.float32)
    (_, block), = list(WavSource(_write(tmp_path / 'a.wav', data)))
    assert np.array_equal(block, data)


def test_wrong_sample_rate(tmp_path):
    p = _write(tmp_path / 'a.wav', np.zeros((config.BLOCK_N, 4)), fs=44100)
    with pytest.raises(ValueError, match='44100'):
        WavSource(p)


@pytest.mark.parametrize('n_ch', [1, 2, 5])
def test_wrong_channel_count(tmp_path, n_ch):
    with pytest.raises(ValueError, match='channels'):
        WavSource(_write(tmp_path / 'a.wav', np.zeros((config.BLOCK_N, n_ch))))


def test_partial_final_block_dropped_and_times(tmp_path):
    extra = config.HOP_N - 1                      # leftover that cannot fill a block
    n = config.BLOCK_N + 3 * config.HOP_N + extra
    data = np.zeros((n, 4), dtype=np.float32)
    data[:, 0] = np.arange(n) / n
    out = list(WavSource(_write(tmp_path / 'a.wav', data)))
    assert len(out) == 4
    for i, (t_s, block) in enumerate(out):
        assert t_s == pytest.approx(i * config.HOP_N / config.FS_HZ)
        assert block.dtype == np.float32 and block.shape == (config.BLOCK_N, 4)
        assert np.allclose(block[:, 0], data[i * config.HOP_N:i * config.HOP_N + config.BLOCK_N, 0])


def test_short_file_yields_nothing(tmp_path):
    p = _write(tmp_path / 'a.wav', np.zeros((config.BLOCK_N - 1, 4)))
    assert list(WavSource(p)) == []


def test_int_wav_gives_float32(tmp_path):
    p = tmp_path / 'a.wav'
    sf.write(str(p), np.full((config.BLOCK_N, 4), 1000, dtype=np.int16), config.FS_HZ)
    (_, block), = list(WavSource(p))
    assert block.dtype == np.float32


def test_realtime_sleeps(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sources.time, 'sleep', calls.append)
    p = _write(tmp_path / 'a.wav', np.zeros((config.BLOCK_N + config.HOP_N, 4)))
    assert len(list(WavSource(p, realtime=True))) == 2
    assert calls == [config.HOP_N / config.FS_HZ] * 2
    calls.clear()
    list(WavSource(p))
    assert calls == []


class _FakeScenario:
    def __init__(self, n):
        self.n = n

    def render(self):
        return np.ones((self.n, 4))               # float64 on purpose

    def truth(self, t_s):
        return {'bearing_deg': 90.0, 'dist_m': 2.0, 'approaching': False}


def test_synthetic_source():
    src = SyntheticSource(_FakeScenario(config.BLOCK_N + 2 * config.HOP_N + 5))
    out = list(src)
    assert len(out) == 3
    assert all(b.shape == (config.BLOCK_N, 4) and b.dtype == np.float32 for _, b in out)
    assert [t for t, _ in out] == pytest.approx([i * config.HOP_N / config.FS_HZ for i in range(3)])
    assert src.truth(0.5) == {'bearing_deg': 90.0, 'dist_m': 2.0, 'approaching': False}
    assert len(list(src)) == 3                    # re-iterable
