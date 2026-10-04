"""Tests for dac/srp.py (step 4, SRP-PHAT). Run with: pytest tests/test_srp.py

The fake-curve tests need no bearing.py: they build the six GCC curves the Section 5.2
contract describes (lags in samples, step 1/16, about 141 points) with narrow bumps at the
delays a source at a given bearing would produce.
"""
import time

import numpy as np
import pytest

from dac import config
from dac.srp import build_steering_table, srp_phat

LAGS = np.arange(-70, 71) / config.UPSAMPLE      # contract grid: +/-4.375 samples, 141 points
PAIR_INDEX = {p: k for k, p in enumerate(config.PAIRS)}


def true_tau(theta_deg, pair, mic_xy=None):
    """Independent re-derivation of tau_ij in samples (not using srp.py)."""
    pos = np.asarray(config.MIC_XY_M if mic_xy is None else mic_xy)
    i, j = pair
    th = np.deg2rad(theta_deg)
    u = np.array([np.sin(th), np.cos(th)])
    return float((pos[i] - pos[j]) @ u / config.C_MPS * config.FS_HZ)


def bump_curves(theta_deg, sigma=0.3, lags=LAGS):
    return [(np.exp(-0.5 * ((lags - true_tau(theta_deg, p)) / sigma) ** 2), lags.copy())
            for p in config.PAIRS]


def circ_err(est, truth):
    return ((est - truth + 180.0) % 360.0) - 180.0


# ---------- steering table ----------

def test_table_shape_and_cached():
    t = build_steering_table()
    assert t.shape == (len(config.ANGLE_GRID_DEG), len(config.PAIRS)) == (180, 6)
    assert build_steering_table() is t


def test_table_sign_worked_example_1():
    # Section 5.4: source at 45 deg, pair (0,2) -> +4.39 samples (mic 0 hears first).
    # 44 deg is on the grid: expect 4.385 * cos(1 deg), and positive.
    t = build_steering_table()
    v = t[config.ANGLE_GRID_DEG.index(44), PAIR_INDEX[(0, 2)]]
    assert v > 0
    assert v == pytest.approx(4.385 * np.cos(np.deg2rad(1.0)), abs=0.01)
    assert true_tau(45, (0, 2)) == pytest.approx(4.39, abs=0.01)


def test_table_sign_worked_example_2():
    # Section 5.4: source at 90 deg (right), pair (0,3) -> +193.8 us = +3.10 samples
    t = build_steering_table()
    v = t[config.ANGLE_GRID_DEG.index(90), PAIR_INDEX[(0, 3)]]
    assert v == pytest.approx(3.10, abs=0.01), "negative here means sin/cos swapped"


def test_table_matches_independent_formula_everywhere():
    t = build_steering_table()
    for a, theta in enumerate(config.ANGLE_GRID_DEG):
        for p, pair in enumerate(config.PAIRS):
            assert t[a, p] == pytest.approx(true_tau(theta, pair), abs=1e-9)


def test_table_within_physical_limits():
    t = build_steering_table()
    assert np.abs(t[:, [PAIR_INDEX[(0, 2)], PAIR_INDEX[(1, 3)]]]).max() <= 4.40
    for pair in [(0, 1), (0, 3), (1, 2), (2, 3)]:
        assert np.abs(t[:, PAIR_INDEX[pair]]).max() <= 3.11


def test_table_antisymmetric_under_180():
    t = build_steering_table()
    idx = {th: a for a, th in enumerate(config.ANGLE_GRID_DEG)}
    for th in config.ANGLE_GRID_DEG:
        np.testing.assert_allclose(t[idx[th]], -t[idx[(th + 180) % 360]], atol=1e-9)


# ---------- required: fake curves ----------

def test_60_degrees():
    b, c, _ = srp_phat(bump_curves(60))
    assert abs(circ_err(b, 60)) <= 1.0, f"expected 60, got {b}"
    assert abs(circ_err(b, 240)) > 90, "60 deg curves must be nowhere near 240"


def test_240_degrees_front_back():
    b, c, _ = srp_phat(bump_curves(240))
    assert abs(circ_err(b, 240)) <= 1.0, f"expected 240, got {b}"
    assert abs(circ_err(b, 60)) > 90


@pytest.mark.parametrize("theta,name", [(0, "front"), (90, "right"), (180, "back"), (270, "left")])
def test_cardinal_directions(theta, name):
    b, c, _ = srp_phat(bump_curves(theta))
    assert abs(circ_err(b, theta)) <= 1.0, f"expected {theta} ({name}), got {b}"


@pytest.mark.parametrize("theta", config.ANGLE_GRID_DEG)
def test_every_grid_angle_round_trips(theta):
    b, _, _ = srp_phat(bump_curves(theta))
    assert abs(circ_err(b, theta)) <= 0.5, f"grid angle {theta} came back as {b}"


def test_off_grid_angles_and_output_range():
    errs = []
    for theta in np.arange(0.5, 360, 1.0):
        b, _, _ = srp_phat(bump_curves(theta))
        assert 0.0 <= b < 360.0, f"{theta} -> {b} out of range"
        errs.append(abs(circ_err(b, theta)))
    assert max(errs) <= 1.5, f"worst off-grid error {max(errs):.2f} deg"
    assert np.mean(errs) <= 0.5


@pytest.mark.parametrize("theta", [359.5, 0.5, 359.9, 0.1, 0.0])
def test_wraparound_near_north(theta):
    b, _, _ = srp_phat(bump_curves(theta))
    assert 0.0 <= b < 360.0
    assert abs(circ_err(b, theta)) <= 1.5


def test_refinement_beats_grid_quantization():
    b, _, _ = srp_phat(bump_curves(61.0))
    assert abs(b - 61.0) < 0.7, "parabola refinement should land between grid points"


def test_output_types_and_srp_shape():
    b, c, srp = srp_phat(bump_curves(123))
    assert isinstance(b, float) and isinstance(c, float)
    assert srp.shape == (180,) and np.all(np.isfinite(srp))
    assert config.ANGLE_GRID_DEG[int(np.argmax(srp))] == 122 or config.ANGLE_GRID_DEG[int(np.argmax(srp))] == 124


# ---------- robustness ----------

def test_dead_pair_still_works():
    for dead in range(6):
        curves = bump_curves(135)
        curves[dead] = (np.zeros_like(LAGS, dtype=float), LAGS.copy())
        b, _, _ = srp_phat(curves)
        assert abs(circ_err(b, 135)) <= 3.0, f"dead pair {dead}: got {b}"


def test_echo_bump_on_one_pair():
    curves = bump_curves(30)
    cc, lags = curves[PAIR_INDEX[(1, 2)]]
    echo = 0.8 * np.exp(-0.5 * ((lags - (-2.0)) / 0.3) ** 2)
    curves[PAIR_INDEX[(1, 2)]] = (cc + echo, lags)
    b, _, _ = srp_phat(curves)
    assert abs(circ_err(b, 30)) <= 3.0


def test_negative_correlation_values_ok():
    curves = [(cc - 0.5, lags) for cc, lags in bump_curves(200)]
    b, c, _ = srp_phat(curves)
    assert abs(circ_err(b, 200)) <= 1.5 and 0 <= c <= 1


def test_additive_noise_on_curves():
    rng = np.random.default_rng(1)
    bad = 0
    for theta in np.arange(5, 360, 10):
        curves = [(cc + 0.15 * rng.standard_normal(cc.size), lags) for cc, lags in bump_curves(theta)]
        b, _, _ = srp_phat(curves)
        bad += abs(circ_err(b, theta)) > 6
    assert bad <= 1


def test_amplitude_scale_invariance():
    b1, c1, _ = srp_phat(bump_curves(77))
    b2, c2, _ = srp_phat([(5.0 * cc, lags) for cc, lags in bump_curves(77)])
    assert b1 == pytest.approx(b2, abs=1e-6) and c1 == pytest.approx(c2, abs=1e-6)


def test_other_lag_window_sizes_work():
    # srp.py must follow the lags it is given, not a hard-coded +/-4.4 window
    for half in (60, 80, 100):
        lags = np.arange(-half, half + 1) / config.UPSAMPLE
        b, _, _ = srp_phat(bump_curves(300, lags=lags))
        assert abs(circ_err(b, 300)) <= 1.0, f"window +/-{half}/16"


def test_lag_grid_offset_not_assumed_symmetric():
    lags = np.arange(-72, 69) / config.UPSAMPLE
    b, _, _ = srp_phat(bump_curves(15, lags=lags))
    assert abs(circ_err(b, 15)) <= 1.0


def test_input_as_lists_and_float32():
    curves = [(cc.astype(np.float32), lags.astype(np.float32)) for cc, lags in bump_curves(88)]
    b, _, _ = srp_phat(curves)
    assert abs(circ_err(b, 88)) <= 1.5
    b, _, _ = srp_phat([(list(cc), list(lags)) for cc, lags in bump_curves(88)])
    assert abs(circ_err(b, 88)) <= 1.5


def test_input_not_mutated():
    curves = bump_curves(10)
    before = [(cc.copy(), lags.copy()) for cc, lags in curves]
    srp_phat(curves)
    for (a, l), (b, m) in zip(curves, before):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(l, m)


# ---------- convention symptoms ----------

def test_negated_lags_gives_opposite_bearing():
    # Documents the Section 7.3 failure "180 deg errors": every pair's sign flipped
    curves = [(cc, -lags[::-1]) for cc, lags in bump_curves(50)]
    curves = [(cc[::-1], lags) for cc, lags in curves]   # keep lags increasing
    b, _, _ = srp_phat(curves)
    assert abs(circ_err(b, 230)) <= 1.5


def test_wrong_pair_order_breaks_result():
    curves = bump_curves(50)
    b, _, _ = srp_phat(curves[::-1])      # wrong order must not silently still work
    assert abs(circ_err(b, 50)) > 3.0


def test_mirror_image_detected():
    # a right-side source must not read as left-side (x mirrored)
    b, _, _ = srp_phat(bump_curves(90))
    assert abs(circ_err(b, 270)) > 90


# ---------- confidence and degenerate input ----------

def test_clean_source_high_confidence():
    for theta in (0, 60, 137, 240, 333):
        _, c, _ = srp_phat(bump_curves(theta))
        assert c > 0.7, f"conf {c} at {theta}"


def test_confidence_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(100):
        curves = [(rng.standard_normal(141) * 10, LAGS.copy()) for _ in range(6)]
        _, c, _ = srp_phat(curves)
        assert 0.0 <= c <= 1.0


def test_noise_confidence_well_below_clean_source():
    # NOTE: with the Section 5.4 formula, pure noise scores about 0.5 (by symmetry the mean
    # sits midway between max and min), not near 0 as the spec text says.
    rng = np.random.default_rng(42)
    noise = [srp_phat([(rng.standard_normal(141), LAGS.copy()) for _ in range(6)])[1]
             for _ in range(200)]
    clean = min(srp_phat(bump_curves(t))[1] for t in range(0, 360, 15))
    assert np.mean(noise) < 0.6
    assert clean > np.mean(noise) + 0.2


@pytest.mark.xfail(reason="Section 7.2 B4 wants confidence < 0.3 on noise in >= 95% of trials; "
                          "the Section 5.4 formula gives ~0.5 (5th-95th pct 0.39-0.62). "
                          "Formula or CONF_MIN needs revisiting by the team.")
def test_b4_noise_confidence_below_0_3():
    rng = np.random.default_rng(0)
    low = 0
    for _ in range(200):
        X = rng.standard_normal((config.BLOCK_N, 4)).astype(np.float32)   # independent per channel
        _, c, _ = srp_phat([_ref_gcc_phat(X[:, i], X[:, j]) for i, j in config.PAIRS])
        low += c < 0.3
    assert low >= 190


def test_confidence_drops_as_peaks_flatten():
    _, c_sharp, _ = srp_phat(bump_curves(60, sigma=0.2))
    rng = np.random.default_rng(3)
    noisy = [(cc * 0.2 + rng.standard_normal(cc.size), lags) for cc, lags in bump_curves(60)]
    _, c_noisy, _ = srp_phat(noisy)
    assert c_sharp > c_noisy


def test_all_zero_curves_return_none():
    b, c, srp = srp_phat([(np.zeros(141), LAGS.copy()) for _ in range(6)])
    assert b is None and c == 0.0 and srp.shape == (180,)


def test_constant_curves_return_none():
    b, c, _ = srp_phat([(np.full(141, 0.3), LAGS.copy()) for _ in range(6)])
    assert b is None and c == 0.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_returns_none(bad):
    curves = bump_curves(60)
    curves[3][0][10] = bad
    b, c, srp = srp_phat(curves)
    assert b is None and c == 0.0 and srp.shape == (180,)


def test_low_confidence_still_returns_bearing():
    rng = np.random.default_rng(5)
    b, c, _ = srp_phat([(rng.standard_normal(141), LAGS.copy()) for _ in range(6)])
    assert b is not None and 0 <= b < 360


# ---------- bad input ----------

def test_wrong_number_of_curves_raises():
    with pytest.raises(ValueError):
        srp_phat(bump_curves(60)[:5])
    with pytest.raises(ValueError):
        srp_phat(bump_curves(60) + bump_curves(60)[:1])
    with pytest.raises(ValueError):
        srp_phat([])


def test_mismatched_lengths_raise():
    curves = bump_curves(60)
    curves[2] = (curves[2][0][:-1], curves[2][1])
    with pytest.raises(ValueError):
        srp_phat(curves)


def test_non_1d_or_unsorted_lags_raise():
    curves = bump_curves(60)
    curves[0] = (curves[0][0].reshape(1, -1), curves[0][1].reshape(1, -1))
    with pytest.raises(ValueError):
        srp_phat(curves)
    curves = bump_curves(60)
    curves[1] = (curves[1][0], curves[1][1][::-1])
    with pytest.raises(ValueError):
        srp_phat(curves)


# ---------- window edge: 141 points cover only +/-4.375 samples, diagonal max is 4.385 ----------

@pytest.mark.parametrize("theta", [45, 135, 225, 315])
def test_diagonal_pair_at_window_edge(theta):
    # steered lag of the diagonal pair is just outside the 141-point window; np.interp clamps
    # to the edge value, so the answer must still be right
    tau = abs(true_tau(theta, (0, 2)))
    assert tau > LAGS.max() or abs(true_tau(theta, (1, 3))) > LAGS.max()
    b, _, _ = srp_phat(bump_curves(theta))
    assert abs(circ_err(b, theta)) <= 1.5, f"{theta} -> {b}"


# ---------- timing ----------

def test_timing_budget():
    curves = bump_curves(60)
    srp_phat(curves)
    t0 = time.perf_counter()
    for _ in range(500):
        srp_phat(curves)
    per_call_ms = (time.perf_counter() - t0) / 500 * 1e3
    assert per_call_ms < 2.0, f"{per_call_ms:.3f} ms per call"


# ---------- end-to-end against a reference implementation of the 5.2 contract ----------
# NOT a replacement for bearing.py: this is a minimal GCC-PHAT written only to check that the
# steering table's sign matches the contract (peak at +D when mic j hears AFTER mic i).

def _ref_gcc_phat(xi, xj, band=config.BAND_HZ):
    n = config.N_FFT
    w = np.hanning(len(xi))
    Xi, Xj = np.fft.rfft(xi * w, n), np.fft.rfft(xj * w, n)
    G = np.conj(Xi) * Xj
    G = G / (np.abs(G) + 1e-12)
    f = np.fft.rfftfreq(n, 1 / config.FS_HZ)
    G[(f < band[0]) | (f > band[1])] = 0
    r = np.fft.irfft(G, n * config.UPSAMPLE)
    idx = np.arange(-70, 71)
    return r[idx % r.size], idx / config.UPSAMPLE


def _plane_wave_block(theta_deg, rng, snr_db=20.0, n=config.BLOCK_N):
    th = np.deg2rad(theta_deg)
    u = np.array([np.sin(th), np.cos(th)])
    pos = np.asarray(config.MIC_XY_M)
    t_arrive = -(pos @ u) / config.C_MPS                 # t_i = t0 - p_i.u / c
    pad = 64
    s = rng.standard_normal(n + 2 * pad)
    S = np.fft.rfft(s)
    f = np.fft.rfftfreq(s.size, 1 / config.FS_HZ)
    cols = []
    for ti in t_arrive:
        x = np.fft.irfft(S * np.exp(-2j * np.pi * f * ti), s.size)[pad:pad + n]
        cols.append(x)
    X = np.stack(cols, axis=1)
    X += rng.standard_normal(X.shape) * np.sqrt(np.mean(X ** 2) / 10 ** (snr_db / 10))
    return X.astype(np.float32)


@pytest.mark.parametrize("theta", [0, 60, 90, 180, 240, 270, 333])
def test_end_to_end_reference_gcc(theta):
    rng = np.random.default_rng(100 + theta)
    X = _plane_wave_block(theta, rng)
    curves = [_ref_gcc_phat(X[:, i], X[:, j]) for i, j in config.PAIRS]
    b, c, _ = srp_phat(curves)
    assert abs(circ_err(b, theta)) <= 6.0, f"expected {theta}, got {b}"
    assert c > 0.3


def test_end_to_end_sweep_accuracy():
    rng = np.random.default_rng(7)
    errs = []
    for theta in np.arange(0.5, 360, 7.0):
        X = _plane_wave_block(theta, rng)
        curves = [_ref_gcc_phat(X[:, i], X[:, j]) for i, j in config.PAIRS]
        b, _, _ = srp_phat(curves)
        errs.append(abs(circ_err(b, theta)))
    assert np.mean(errs) <= 3.0 and max(errs) <= 6.0, f"max {max(errs):.1f}, mean {np.mean(errs):.1f}"


def test_end_to_end_conj_on_wrong_channel_flips_bearing():
    # If GCC conjugated X_j instead of X_i the peak would sit at -D and the bearing would flip
    rng = np.random.default_rng(11)
    X = _plane_wave_block(60, rng)
    curves = []
    for i, j in config.PAIRS:
        cc, lags = _ref_gcc_phat(X[:, j], X[:, i])      # arguments swapped = conj on the other side
        curves.append((cc, lags))
    b, _, _ = srp_phat(curves)
    assert abs(circ_err(b, 240)) <= 6.0
