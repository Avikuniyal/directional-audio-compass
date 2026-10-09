"""Synthetic microphone-array signals with known ground truth (Section 8.4). Owner: Srihaas.

Conventions (Section 4.2): bearing is clockwise from front, u(theta) = (sin, cos), positions are
(x right, y front) in meters. A mic at p hears a plane wave from theta at t = -(p . u) / c
relative to the array center, so the mic nearest the source hears it first.

Level convention: every signal generator returns RMS SIGNAL_RMS (-20 dBFS) as heard at 1 m.
`snr_db` is measured against that 1 m reference level, so the noise floor is constant over a
trajectory and a source that moves away gets quieter relative to it. snr_db=None adds no noise.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from dac import config

SIGNAL_RMS = 0.1
_BASE_DELAY = 16.0   # common delay (samples) added to every channel so no delay is negative


def fractional_delay(x, delay_samples):
    """Delay 1-D x by a possibly fractional number of samples (>= 0), same length out.

    Zero-pad to >= 2x length, FFT, multiply bin k by exp(-2 pi i k d / N_pad), inverse, trim.
    """
    x = np.asarray(x, dtype=np.float64)
    if delay_samples < 0:
        raise ValueError("delay_samples must be >= 0")
    n = len(x)
    n_pad = 1 << int(np.ceil(np.log2(2 * n + 2 * int(np.ceil(delay_samples)) + 2)))
    spec = np.fft.rfft(x, n_pad)
    k = np.arange(spec.size)
    spec *= np.exp(-2j * np.pi * k * delay_samples / n_pad)
    return np.fft.irfft(spec, n_pad)[:n]


# ---- signals ----

def _scale(x, rms):
    return (x * (rms / (np.sqrt(np.mean(x ** 2)) + 1e-30))).astype(np.float64)


def white_noise(n, seed=0, rms=SIGNAL_RMS):
    return _scale(np.random.default_rng(seed).standard_normal(n), rms)


def lowpass_noise(n, cutoff_hz, seed=0, rms=SIGNAL_RMS):
    x = np.random.default_rng(seed).standard_normal(n)
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(n, 1.0 / config.FS_HZ) > cutoff_hz] = 0.0
    return _scale(np.fft.irfft(spec, n), rms)


def engine_like(n, f0_hz=40.0, seed=0, rms=SIGNAL_RMS):
    """Harmonics of a 30 to 60 Hz firing frequency plus low-passed noise."""
    t = np.arange(n) / config.FS_HZ
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for h in range(1, 41):
        if h * f0_hz >= 2000:
            break
        x += (1.0 / h) * np.sin(2 * np.pi * h * f0_hz * t + rng.uniform(0, 2 * np.pi))
    x = _scale(x, 1.0) + 0.5 * _scale(lowpass_noise(n, 800.0, seed + 1), 1.0)
    return _scale(x, rms)


# ---- trajectories: t_s -> (x_m, y_m) relative to the array center ----

def _xy(bearing_deg, dist_m):
    b = np.deg2rad(bearing_deg)
    return dist_m * np.sin(b), dist_m * np.cos(b)


def static(bearing_deg, dist_m):
    xy = _xy(bearing_deg, dist_m)
    return lambda t_s: xy


def straight_line(start_xy, end_xy, speed_mps):
    a, b = np.asarray(start_xy, float), np.asarray(end_xy, float)
    length = float(np.linalg.norm(b - a))

    def traj(t_s):
        f = 1.0 if length == 0 else min(max(speed_mps * t_s / length, 0.0), 1.0)
        p = a + f * (b - a)
        return float(p[0]), float(p[1])
    return traj


def crossing(offset_m, speed_mps, left_to_right=True, half_span_m=6.0):
    """Walk parallel to the array's x axis, offset_m in front, closest at t = half_span / speed."""
    sign = 1.0 if left_to_right else -1.0
    return straight_line((-sign * half_span_m, offset_m), (sign * half_span_m, offset_m), speed_mps)


def circle(radius_m, period_s):
    """Clockwise circle (bearing increasing) starting at front."""
    return lambda t_s: _xy(360.0 * t_s / period_s, radius_m)


def _bearing_of(x, y):
    return float(np.degrees(np.arctan2(x, y)) % 360.0)   # atan2(x, y): compass order, not (y, x)


# ---- rendering ----

def _noise(shape, signal_rms, snr_db, rng):
    if snr_db is None:
        return 0.0
    return rng.standard_normal(shape) * signal_rms / 10 ** (snr_db / 20.0)


def render_plane_wave(signal, bearing_deg, snr_db=None, seed=0):
    """(n, 4) float32 in mic order for a plane wave from bearing_deg."""
    signal = np.asarray(signal, dtype=np.float64)
    pos = np.asarray(config.MIC_XY_M, dtype=np.float64)
    b = np.deg2rad(bearing_deg)
    t_i = -(pos @ np.array([np.sin(b), np.cos(b)])) / config.C_MPS
    out = np.stack([fractional_delay(signal, _BASE_DELAY + t * config.FS_HZ) for t in t_i], axis=1)
    rng = np.random.default_rng(seed)
    out += _noise(out.shape, np.sqrt(np.mean(signal ** 2)), snr_db, rng)
    return out.astype(np.float32)


def render_point_source(signal, trajectory, snr_db=None, seed=0, chunk_n=256):
    """(n, 4) float32 for a possibly moving point source.

    Per chunk: exact distance |s - p_i| to each mic, delay dist / c, amplitude 1 / dist. Chunks
    overlap by 50% with a periodic Hann crossfade (sums to 1), so there are no clicks.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    pos = np.asarray(config.MIC_XY_M, dtype=np.float64)
    hop = chunk_n // 2
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(chunk_n) / chunk_n)
    guard = 4
    out = np.zeros((n + 2 * chunk_n, 4))      # output sample m lives at index m + chunk_n
    for s in range(-hop, n, hop):
        t_mid = (s + chunk_n / 2) / config.FS_HZ
        src = np.asarray(trajectory(max(t_mid, 0.0)), dtype=np.float64)
        dists = np.linalg.norm(pos - src, axis=1)
        for m in range(4):
            d = dists[m] / config.C_MPS * config.FS_HZ
            d_int = int(np.floor(d))
            lo = s - d_int - guard
            idx = np.arange(lo, lo + chunk_n + 2 * guard)
            seg = np.where((idx >= 0) & (idx < n), signal[np.clip(idx, 0, n - 1)], 0.0)
            y = fractional_delay(seg, d - d_int)[guard:guard + chunk_n]
            out[s + chunk_n:s + 2 * chunk_n, m] += win * y / dists[m]
    out = out[chunk_n:chunk_n + n]
    out += _noise(out.shape, np.sqrt(np.mean(signal ** 2)), snr_db, np.random.default_rng(seed))
    return out.astype(np.float32)


@dataclass
class Scenario:
    name: str
    signal: np.ndarray
    trajectory: Callable
    duration_s: float
    snr_db: float = 20.0
    seed: int = 0

    def render(self):
        n = int(round(self.duration_s * config.FS_HZ))
        return render_point_source(self.signal[:n], self.trajectory, self.snr_db, self.seed)

    def truth(self, t_s):
        """dict: bearing_deg, dist_m, approaching (distance falling and |bearing rate| < RATE_MAX_DPS)."""
        x, y = self.trajectory(t_s)
        eps = 0.05
        x0, y0 = self.trajectory(max(t_s - eps, 0.0))
        x1, y1 = self.trajectory(t_s + eps)
        dt = t_s + eps - max(t_s - eps, 0.0)
        dist_rate = (np.hypot(x1, y1) - np.hypot(x0, y0)) / dt
        dbear = ((_bearing_of(x1, y1) - _bearing_of(x0, y0) + 180.0) % 360.0) - 180.0
        return {
            'bearing_deg': _bearing_of(x, y),
            'dist_m': float(np.hypot(x, y)),
            'approaching': bool(dist_rate < 0 and abs(dbear / dt) < config.RATE_MAX_DPS),
        }


def make_scenario(kind, seed=0, snr_db=20.0):
    """Named scenarios used by the harness and tests: approach_0deg, crossing_2m."""
    if kind == 'approach_0deg':       # 6 m to 1 m straight at the array, 1.4 m/s
        traj, dur = straight_line((0.0, 6.0), (0.0, 1.0), 1.4), 5.0 / 1.4
    elif kind == 'crossing_2m':       # passes 2 m in front, 1.4 m/s
        traj, dur = crossing(2.0, 1.4, True, 4.0), 8.0 / 1.4
    else:
        raise ValueError(f"unknown scenario {kind!r}")
    n = int(round(dur * config.FS_HZ))
    return Scenario(kind, engine_like(n, seed=seed), traj, dur, snr_db, seed)
