# tests/test_harness.py  (owner: Srihaas). Metrics on hand-computed cases and an evaluate() smoke test.
import numpy as np
import pytest

from dac import config
from dac.stub_bearing import StubBearing
from harness import metrics, run_eval

HOP_S = config.HOP_N / config.FS_HZ


def test_circular_error_wrap():
    assert metrics.circular_error_deg(359, 1) == pytest.approx(-2)
    assert metrics.circular_error_deg(1, 359) == pytest.approx(2)
    assert metrics.circular_error_deg(10, 350) == pytest.approx(20)
    assert metrics.circular_error_deg(0, 180) == pytest.approx(-180)
    assert np.allclose(metrics.circular_error_deg([359, 90], [1, 80]), [-2, 10])


def test_rms_error_wrap_and_nan():
    assert metrics.rms_error_deg([359, 1], [1, 359]) == pytest.approx(2)
    assert metrics.rms_error_deg([3, np.nan, 4], [0, 0, 0]) == pytest.approx(np.sqrt(12.5))
    assert np.isnan(metrics.rms_error_deg([np.nan], [0]))


def test_rate_error():
    t = np.arange(0, 2, 0.5)
    truth = [358, 0, 2, 4]                       # +2 deg per 0.5 s across the wrap = +4 deg/s
    assert np.allclose(metrics.true_rate_dps(t, truth), 4)
    err = metrics.rate_error([5, np.nan, 3, 4], [4, 4, 4, 4])
    assert np.allclose(err, [1, np.nan, -1, 0], equal_nan=True)
    assert metrics.rms_rate_error([5, np.nan, 3], [4, 4, 4]) == pytest.approx(1)


def _approach(flag_from):
    t = np.arange(0, 5.0, 0.5)                   # 10 blocks
    dist = 6.0 - t                               # 6 m .. 1.5 m, reaches 3 m at t = 3.0
    return t, t >= flag_from, dist, np.ones_like(t, dtype=bool)


def test_detection_before_3m():
    r = metrics.approach_result(*_approach(2.0))
    assert r['detected'] and r['time_to_flag_s'] == pytest.approx(2.0)


def test_detection_late_or_never():
    r = metrics.approach_result(*_approach(3.5))     # flagged at 2.5 m
    assert not r['detected'] and r['time_to_flag_s'] == pytest.approx(3.5)
    r = metrics.approach_result(*_approach(99.0))
    assert not r['detected'] and np.isnan(r['time_to_flag_s'])


def test_detection_rate():
    rs = [{'detected': True}, {'detected': False}, {'detected': True}, {'detected': True}]
    assert metrics.detection_rate(rs) == pytest.approx(0.75)


def test_false_alarm_fraction():
    assert metrics.false_alarm_fraction([0, 0, 1, 1, 0, 0, 0, 0, 0, 0]) == pytest.approx(0.2)
    assert metrics.false_alarm_fraction([]) == 0.0


# ---- evaluate() smoke test with a fake source and fake truth ----

class FakeSource:
    def __init__(self, n_blocks=60, level_db=-30.0):
        self.n, self.amp = n_blocks, 10 ** (level_db / 20)

    def __iter__(self):
        for k in range(self.n):
            yield k * HOP_S, np.full((config.BLOCK_N, 4), self.amp, dtype=np.float32)


def fake_truth(t_s):
    return {'bearing_deg': 90.0, 'dist_m': 5.0, 'approaching': False}


def test_evaluate_smoke():
    stub = StubBearing(lambda t: fake_truth(t)['bearing_deg'], noise_deg=2.0, dropout=0.0, seed=0)
    recs = run_eval.evaluate(FakeSource(), stub.estimate, fake_truth)
    assert len(recs) == 60
    assert recs[0]['t_s'] == 0.0 and recs[-1]['t_s'] == pytest.approx(59 * HOP_S)
    assert all(abs(metrics.circular_error_deg(r['est_bearing'], 90.0)) < 15 for r in recs)
    assert recs[-1]['true_bearing'] == 90.0
    s = run_eval.summarize(recs, 'crossing')
    assert s['rms_bearing_err_deg'] < 5 and s['false_alarm_frac'] == 0.0


def test_evaluate_no_truth_and_overrides():
    stub = StubBearing(lambda t: 0.0, dropout=0.0)
    old = config.RATE_MAX_DPS
    recs = run_eval.evaluate(FakeSource(10), stub.estimate, None, overrides={'RATE_MAX_DPS': 99.0})
    assert config.RATE_MAX_DPS == old                # override restored
    assert len(recs) == 10 and np.isnan(recs[0]['true_bearing'])
    assert not run_eval.summarize(recs)['has_truth']


def test_manifest_truth_interpolation():
    row = {'kind': 'crossing', 'start_bearing_deg': '303.7', 'end_bearing_deg': '56.3',
           'start_dist_m': '3.6', 'end_dist_m': '3.6', 'move_start_s': '2', 'move_end_s': '6'}
    tr = run_eval.manifest_truth_fn(row)
    assert tr(0)['bearing_deg'] == pytest.approx(303.7)
    assert tr(4)['bearing_deg'] == pytest.approx(0.0, abs=1e-6)      # midpoint via 0, not 180
    assert tr(9)['bearing_deg'] == pytest.approx(56.3)
    static = {'kind': 'static', 'start_bearing_deg': '45', 'end_bearing_deg': '45',
              'start_dist_m': '1.5', 'end_dist_m': '1.5', 'move_start_s': '', 'move_end_s': ''}
    assert run_eval.manifest_truth_fn(static)(7)['bearing_deg'] == pytest.approx(45)
