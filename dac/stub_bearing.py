"""Stand-in for dac/bearing.py that fakes bearings from ground truth (Section 6.3). Owner: Srihaas."""
import numpy as np

from dac import config
from dac.types import BearingResult


class StubBearing:
    def __init__(self, truth_fn, noise_deg=5.0, conf=0.8, dropout=0.05, seed=0):
        self.truth_fn = truth_fn
        self.noise_deg = noise_deg
        self.conf = conf
        self.dropout = dropout
        self._rng = np.random.default_rng(seed)
        self._grid = np.asarray(config.ANGLE_GRID_DEG, dtype=np.float64)

    def estimate(self, block, t_s):
        # Draw both values every call so the sequence does not depend on which calls drop out.
        noise = self._rng.normal(0.0, self.noise_deg)
        dropped = self._rng.random() < self.dropout
        if dropped:
            return BearingResult(None, 0.0, np.zeros(len(self._grid)))
        bearing = (self.truth_fn(t_s) + noise) % 360.0
        diff = ((self._grid - bearing + 180.0) % 360.0) - 180.0
        srp = np.exp(-0.5 * (diff / 10.0) ** 2)
        return BearingResult(float(bearing), float(self.conf), srp)

    __call__ = estimate
