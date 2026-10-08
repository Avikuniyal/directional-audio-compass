# tests/test_synth.py  (owner: Srihaas)
import numpy as np

from dac import config, synth


def test_integer_delay_is_exact():
    x = synth.white_noise(1024, seed=1)
    y = synth.fractional_delay(x, 3)
    assert np.allclose(y[3:], x[:-3], atol=1e-9)
    assert np.allclose(y[:3], 0, atol=1e-9)


def test_plane_wave_45deg_delay_between_columns_0_and_2():
    n = 4096
    t = np.arange(n) / config.FS_HZ
    sig = np.sin(2 * np.pi * 500 * t)
    out = synth.render_plane_wave(sig, 45.0, snr_db=None)
    f_bin = 500 * n / config.FS_HZ    # exactly on a bin (500 Hz * 4096 / 16000 = 128)
    ph = [np.angle(np.fft.rfft(out[:, c].astype(np.float64))[int(f_bin)]) for c in (0, 2)]
    dphi = (ph[0] - ph[1] + np.pi) % (2 * np.pi) - np.pi     # column 0 leads: negative delay
    delay_s = -dphi / (2 * np.pi * 500)                      # t_2 - t_0, positive
    assert abs(delay_s - 274.1e-6) < 0.5e-6                  # 4.39 samples


def test_distance_doubling_lowers_level_6db():
    sig = synth.white_noise(8000, seed=2)
    levels = []
    for d in (1.0, 2.0):
        out = synth.render_point_source(sig, synth.static(0.0, d))
        levels.append(20 * np.log10(np.sqrt(np.mean(out[2000:6000] ** 2))))
    # exact per-mic distances differ slightly from d, so compare with a tolerance
    assert abs((levels[0] - levels[1]) - 6.02) < 0.1


def test_truth_static_90_is_90_not_270():
    sc = synth.Scenario('s', synth.white_noise(1600), synth.static(90.0, 2.0), 0.1)
    t = sc.truth(0.05)
    assert abs(t['bearing_deg'] - 90.0) < 1e-6
    assert abs(t['dist_m'] - 2.0) < 1e-6
    assert not t['approaching']


def test_truth_approach_and_crossing_flags():
    a = synth.make_scenario('approach_0deg')
    assert a.truth(1.0)['approaching']
    c = synth.make_scenario('crossing_2m')
    assert not c.truth(c.duration_s / 2)['approaching']   # closest point: bearing rate is 40 deg/s


def test_render_shape_dtype_and_snr():
    out = synth.render_plane_wave(synth.white_noise(2048), 120.0, snr_db=20.0, seed=3)
    assert out.shape == (2048, 4) and out.dtype == np.float32
