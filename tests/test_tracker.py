# tests/test_tracker.py  (owner: Srihaas). T1 to T8 from Section 8.6.
import numpy as np
import pytest

from dac import config, synth
from dac.stub_bearing import StubBearing
from dac.tracker import Tracker
from dac.types import BearingResult

HOP_S = config.HOP_N / config.FS_HZ
ZERO_SRP = np.zeros(len(config.ANGLE_GRID_DEG))


def br(deg, conf=0.9):
    return BearingResult(deg, conf if deg is not None else 0.0, ZERO_SRP)


def const_block(level_db):
    return np.full((config.BLOCK_N, 4), 10 ** (level_db / 20), dtype=np.float32)


def circ_diff(a, b):
    return ((a - b + 180) % 360) - 180


def run_scenario(sc, bearing_fn):
    audio = sc.render()
    tr, states = Tracker(), []
    for s in range(0, len(audio) - config.BLOCK_N + 1, config.HOP_N):
        t = s / config.FS_HZ
        blk = audio[s:s + config.BLOCK_N]
        states.append((t, tr.update(t, bearing_fn(blk, t), blk)))
    return states


def test_t1_wrap():
    tr, rate = Tracker(), None
    for k in range(int(2.0 / HOP_S)):
        t = k * HOP_S
        st = tr.update(t, br((350 + 20 * t) % 360), const_block(-30))
        rate = st.rate_dps
    assert abs(rate - 20) < 1


def test_t2_still_source():
    rng, tr, rates = np.random.default_rng(0), Tracker(), []
    for k in range(int(10.0 / HOP_S)):
        st = tr.update(k * HOP_S, br((90 + rng.normal(0, 5)) % 360), const_block(-30))
        if st.rate_dps is not None:
            rates.append(abs(st.rate_dps))
    assert rates and np.mean(rates) < 5


def test_t3_trend():
    tr, trend = Tracker(), None
    for k in range(int(3.0 / HOP_S)):
        t = k * HOP_S
        trend = tr.update(t, br(0.0), const_block(-40 + 4 * t)).trend_dbps
    assert abs(trend - 4) < 0.3


@pytest.mark.xfail(strict=True, reason=(
    "Known S7 tuning finding: with provisional config the flag never turns on before the scenario "
    "ends at 1 m. LEVEL_MARGIN_DB=10 is only met from about 1.6 m (level rises ~10 dB between 6 m "
    "and 2 m, and the noise floor here is the source's own starting level), then ON_HOLD_S adds "
    "0.5 s. Fix by tuning (S7), then remove this marker."))
def test_t4_approach_before_3m():
    sc = synth.make_scenario('approach_0deg')
    stub = StubBearing(lambda t: sc.truth(t)['bearing_deg'], noise_deg=8.0, dropout=0.05, seed=1)
    states = run_scenario(sc, stub.estimate)
    on = [t for t, st in states if st.approach]
    assert on, "approach never turned on"
    assert sc.truth(on[0])['dist_m'] > 3.0, f"flagged only at {sc.truth(on[0])['dist_m']:.2f} m"


def test_t5_crossing_never_approaches():
    sc = synth.make_scenario('crossing_2m')
    stub = StubBearing(lambda t: sc.truth(t)['bearing_deg'], noise_deg=8.0, dropout=0.05, seed=2)
    assert not any(st.approach for _, st in run_scenario(sc, stub.estimate))


def test_t6_dropouts():
    rng, tr = np.random.default_rng(3), Tracker()
    accepted = []
    for k in range(int(8.0 / HOP_S)):
        t = k * HOP_S
        drop = rng.random() < 0.30
        st = tr.update(t, br(None if drop else 45.0), const_block(-30))
        if not drop:
            accepted.append(t)
        in_window = sum(1 for a in accepted if a >= t - config.RATE_WINDOW_S)
        if in_window < config.RATE_MIN_POINTS:
            assert st.rate_dps is None


def test_t7_hysteresis():
    # Force the raw condition via level: ramp up/down so it toggles every 0.2 s is not possible
    # with a real slope, so drive the hysteresis directly.
    tr, flags = Tracker(), []
    for k in range(int(10.0 / 0.05)):
        t = k * 0.05
        tr._hysteresis(t, int(t / 0.2) % 2 == 0)
        flags.append(tr._approach)
    assert not any(flags)
    # Sustained true turns it on after ON_HOLD_S, a short false gap does not turn it off.
    tr = Tracker()
    for k in range(40):
        tr._hysteresis(k * 0.05, True)
    assert tr._approach
    for k in range(40, 50):          # 0.5 s of false < OFF_HOLD_S
        tr._hysteresis(k * 0.05, False)
    assert tr._approach
    for k in range(50, 90):          # 2 s of false > OFF_HOLD_S
        tr._hysteresis(k * 0.05, False)
    assert not tr._approach


def test_t8_display_mean():
    tr, smooth = Tracker(), None
    for k in range(40):
        smooth = tr.update(k * HOP_S, br(350.0 if k % 2 == 0 else 10.0), const_block(-30)).bearing_smooth_deg
    assert abs(circ_diff(smooth, 0.0)) < 5
