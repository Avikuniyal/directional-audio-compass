"""Steps 5 and 6a: loudness, bearing rate, smoothed bearing, approach flag (Sections 5.5, 5.6).

Owner: Srihaas. The only stateful signal-processing module. Histories are deques of (t_s, value)
trimmed by time, not count, so a dropped block does not distort a window.
"""
from collections import deque

import numpy as np

from dac import config
from dac.types import BearingResult, TrackState

_HOP_S = config.HOP_N / config.FS_HZ


def _slope(ts, vs):
    """Least-squares slope of vs against ts."""
    ts, vs = np.asarray(ts, float), np.asarray(vs, float)
    dt = ts - ts.mean()
    denom = float(np.sum(dt * dt))
    return None if denom == 0 else float(np.sum(dt * (vs - vs.mean())) / denom)


def loudness_dbfs(block):
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(block, dtype=np.float64))) + 1e-12))


class Tracker:
    def __init__(self):
        self._levels = deque()       # (t, L) over FLOOR_WINDOW_S (also feeds the trend)
        self._accepted = deque()     # (t, bearing_deg) over max(RATE, DISPLAY) window
        self._last_bearing = None
        self._approach = False
        self._true_since = None
        self._false_since = None

    @staticmethod
    def _trim(dq, t_s, window):
        while dq and dq[0][0] < t_s - window:
            dq.popleft()

    def update(self, t_s, bearing: BearingResult, block):
        level = loudness_dbfs(block)
        self._levels.append((t_s, level))
        self._trim(self._levels, t_s, config.FLOOR_WINDOW_S)

        if bearing.bearing_deg is not None and bearing.confidence >= config.CONF_MIN:
            self._accepted.append((t_s, float(bearing.bearing_deg)))
            self._last_bearing = float(bearing.bearing_deg)
        self._trim(self._accepted, t_s, max(config.RATE_WINDOW_S, config.DISPLAY_WINDOW_S))

        trend = self._trend(t_s)
        rate = self._rate(t_s)
        smooth = self._smooth(t_s)
        floor = float(np.percentile([v for _, v in self._levels], 10))
        raw = (rate is not None and abs(rate) <= config.RATE_MAX_DPS
               and trend is not None and trend >= config.TREND_MIN_DBPS
               and level >= floor + config.LEVEL_MARGIN_DB)
        self._hysteresis(t_s, raw)

        return TrackState(t_s=t_s, bearing_deg=self._last_bearing, bearing_smooth_deg=smooth,
                          confidence=float(bearing.confidence), level_dbfs=level,
                          trend_dbps=trend, rate_dps=rate, noise_floor_dbfs=floor,
                          approach=self._approach)

    def _trend(self, t_s):
        pts = [(t, v) for t, v in self._levels if t >= t_s - config.TREND_WINDOW_S]
        # Window counts as filled once it spans TREND_WINDOW_S to within one block.
        if len(pts) < 2 or t_s - pts[0][0] < config.TREND_WINDOW_S - 1.5 * _HOP_S:
            return None
        return _slope(*zip(*pts))

    def _rate(self, t_s):
        pts = [(t, b) for t, b in self._accepted if t >= t_s - config.RATE_WINDOW_S]
        if len(pts) < config.RATE_MIN_POINTS:
            return None
        ts, bs = zip(*pts)
        ref = bs[-1]
        delta = [((b - ref + 180.0) % 360.0) - 180.0 for b in bs]
        return _slope(ts, delta)

    def _smooth(self, t_s):
        bs = [b for t, b in self._accepted if t >= t_s - config.DISPLAY_WINDOW_S]
        if not bs:
            return None
        r = np.deg2rad(bs)
        return float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 360.0)

    def _hysteresis(self, t_s, raw):
        if raw:
            self._false_since = None
            if self._true_since is None:
                self._true_since = t_s
            if not self._approach and t_s - self._true_since >= config.ON_HOLD_S:
                self._approach = True
        else:
            self._true_since = None
            if self._false_since is None:
                self._false_since = t_s
            if self._approach and t_s - self._false_since >= config.OFF_HOLD_S:
                self._approach = False
