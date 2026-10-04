"""Step 4: SRP-PHAT direction search (Section 5.4). Owner: Srihaas.

Input is the six per-pair GCC-PHAT curves from the Section 5.2 contract: (cc, lags) with
lags in samples (fractional), in PAIRS order. For every candidate bearing we predict each
pair's delay, read the correlation at that delay, and sum over pairs.
"""
import numpy as np

from dac import config

_TAU_SAMPLES = None   # (n_angles, n_pairs), predicted tau_ij in samples, built once


def build_steering_table():
    """Predicted delay tau_ij(theta) = t_j - t_i, in samples, shape (n_angles, n_pairs).

    tau_ij = (p_i - p_j) . (sin theta, cos theta) / c, positive when mic i hears first.
    Depends only on geometry, so it is computed once and cached.
    """
    global _TAU_SAMPLES
    if _TAU_SAMPLES is None:
        pos = np.asarray(config.MIC_XY_M, dtype=np.float64)
        theta = np.deg2rad(np.asarray(config.ANGLE_GRID_DEG, dtype=np.float64))
        u = np.stack([np.sin(theta), np.cos(theta)], axis=1)              # (A, 2)
        d = np.array([pos[i] - pos[j] for i, j in config.PAIRS])          # (P, 2)
        _TAU_SAMPLES = (u @ d.T) / config.C_MPS * config.FS_HZ
    return _TAU_SAMPLES


def _validate(curves):
    if len(curves) != len(config.PAIRS):
        raise ValueError(f"expected {len(config.PAIRS)} curves, got {len(curves)}")
    out = []
    for k, (cc, lags) in enumerate(curves):
        cc = np.asarray(cc, dtype=np.float64)
        lags = np.asarray(lags, dtype=np.float64)
        if cc.ndim != 1 or cc.shape != lags.shape or cc.size < 2:
            raise ValueError(f"curve {k}: cc and lags must be 1-D, same length, >= 2 points")
        if not np.all(np.diff(lags) > 0):
            raise ValueError(f"curve {k}: lags must be strictly increasing")
        out.append((cc, lags))
    return out


def srp_phat(curves):
    """Return (bearing_deg, confidence, srp).

    srp is P(theta_k) on ANGLE_GRID_DEG. bearing is None (confidence 0) only for degenerate
    input: non-finite values or a flat P map. A low confidence still returns a bearing;
    gating on CONF_MIN is the tracker's job. Silence is checked in estimate_bearing.
    """
    curves = _validate(curves)
    tau = build_steering_table()
    n_angles = tau.shape[0]
    srp = np.zeros(n_angles)
    finite = True
    for p, (cc, lags) in enumerate(curves):
        if not np.all(np.isfinite(cc)):
            finite = False
            break
        srp += np.interp(tau[:, p], lags, cc)
    if not finite:
        return None, 0.0, np.zeros(n_angles)

    p_max, p_min = srp.max(), srp.min()
    if p_max - p_min < 1e-12:
        return None, 0.0, srp

    k = int(np.argmax(srp))
    # Parabola through the peak and its circular neighbours, shift clamped to one grid step.
    left, right = srp[(k - 1) % n_angles], srp[(k + 1) % n_angles]
    denom = left - 2.0 * srp[k] + right
    shift = 0.0 if abs(denom) < 1e-12 else 0.5 * (left - right) / denom
    shift = float(np.clip(shift, -1.0, 1.0))
    step = 360.0 / n_angles
    bearing = (config.ANGLE_GRID_DEG[k] + shift * step) % 360.0
    if bearing >= 360.0:    # float rounding of a tiny negative value
        bearing = 0.0

    conf = (p_max - srp.mean()) / (p_max - p_min + 1e-12)
    return float(bearing), float(np.clip(conf, 0.0, 1.0)), srp
