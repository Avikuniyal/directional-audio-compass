"""Evaluation metrics (Sections 7.2, 8.7, 8.8). Owner: Srihaas.

Pure functions on arrays and lists. Time is in seconds, angles in degrees, and a missing
value (no bearing, no rate, no truth) is NaN.
"""
import numpy as np

NEAR_M = 3.0   # an approach must be flagged before the source is this close (Section 8.7)


def circular_error_deg(est, truth):
    """Signed error in [-180, 180): ((est - truth + 180) mod 360) - 180. 359 vs 1 -> -2."""
    est = np.asarray(est, dtype=float)
    truth = np.asarray(truth, dtype=float)
    out = np.mod(est - truth + 180.0, 360.0) - 180.0
    return float(out) if out.ndim == 0 else out


def rms_error_deg(est, truth):
    """RMS of the circular error over entries where both est and truth are finite. NaN if none."""
    err = np.atleast_1d(circular_error_deg(est, truth))
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err ** 2))) if err.size else float('nan')


def true_rate_dps(t, truth_deg):
    """True bearing rate in deg/s: gradient of the unwrapped true bearing, so 358 -> 2 is +4."""
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return np.full(t.shape, np.nan)
    return np.gradient(np.rad2deg(np.unwrap(np.deg2rad(np.asarray(truth_deg, dtype=float)))), t)


def rate_error(est_rate, true_rate):
    """Per-block rate error = estimated minus true rate (deg/s). NaN where either is missing."""
    return np.asarray(est_rate, dtype=float) - np.asarray(true_rate, dtype=float)


def rms_rate_error(est_rate, true_rate):
    """RMS of rate_error over blocks where both rates exist. NaN if none."""
    err = rate_error(est_rate, true_rate)
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err ** 2))) if err.size else float('nan')


def approach_result(t, flag, dist, approaching, near_m=NEAR_M):
    """Score one approach scenario. All inputs are equal-length per-block arrays.

    t: block times. flag: tracker approach flag (bool). dist: true distance (m).
    approaching: true approach state (bool). near_m: the deadline distance.
    Approach start = first block with approaching True. Deadline = first block with
    dist <= near_m (end of the recording if the source never gets that close).
    detected = the flag was on at some block in [start, deadline). time_to_flag_s =
    first flagged time minus start, over the whole recording (NaN if never flagged).
    """
    t = np.asarray(t, dtype=float)
    flag = np.asarray(flag, dtype=bool)
    dist = np.asarray(dist, dtype=float)
    appr = np.asarray(approaching, dtype=bool)
    if not appr.any():
        return {'detected': False, 'time_to_flag_s': float('nan'), 'start_s': float('nan')}
    i0 = int(np.argmax(appr))
    near = np.flatnonzero(dist <= near_m)
    deadline = t[near[0]] if near.size else np.inf
    on = np.flatnonzero(flag & (np.arange(t.size) >= i0))
    t_flag = t[on[0]] if on.size else float('nan')
    return {'detected': bool(on.size and t_flag < deadline),
            'time_to_flag_s': float(t_flag - t[i0]) if on.size else float('nan'),
            'start_s': float(t[i0])}


def detection_rate(results):
    """Fraction of approach_result dicts with detected True. NaN for an empty list."""
    return float(np.mean([r['detected'] for r in results])) if len(results) else float('nan')


def false_alarm_fraction(flag):
    """Fraction of blocks (blocks are evenly spaced, so also of time) with the flag on.

    Meant for crossing scenarios, where the flag should stay off. 0.0 for an empty input.
    """
    flag = np.asarray(flag, dtype=bool)
    return float(flag.mean()) if flag.size else 0.0
