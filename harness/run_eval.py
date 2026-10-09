"""Evaluation harness (Section 8.8). Owner: Srihaas.

    python -m harness.run_eval --source synth:approach_0deg --bearing stub
    python -m harness.run_eval --source wav:data/recordings/R14a_approach_000.wav --bearing real

Runs source -> bearing function -> Tracker, prints a results table, saves a plot to harness/out/.
`evaluate()` is the reusable core (no CLI, no plotting) for the tuning grid.
"""
import argparse
import contextlib
import csv
import pathlib
import sys

import numpy as np

from dac import config
from harness import metrics

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'data' / 'manifest.csv'
OUT_DIR = ROOT / 'harness' / 'out'
NAN = float('nan')


@contextlib.contextmanager
def config_overrides(overrides):
    """Temporarily set dac.config attributes, e.g. {'RATE_MAX_DPS': 20.0}."""
    old = {k: getattr(config, k) for k in (overrides or {})}
    for k, v in (overrides or {}).items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def evaluate(source, bearing_fn, truth_fn=None, tracker=None, overrides=None, max_blocks=None):
    """Run every block of `source` through bearing_fn and the tracker; return per-block records.

    bearing_fn(block, t_s) -> BearingResult. truth_fn(t_s) -> dict with bearing_deg, dist_m,
    approaching (as SyntheticSource.truth), or None for no truth. tracker: a Tracker-like
    object (default: a new dac.tracker.Tracker built after `overrides` are applied).
    overrides: dict of dac.config names to temporarily set (the tracker is built inside).
    Returns a list of dicts: t_s, est_bearing, conf, smooth, rate, trend, level, floor,
    approach, true_bearing, true_dist, true_approaching (NaN / False when there is no truth).
    """
    with config_overrides(overrides):
        if tracker is None:
            from dac.tracker import Tracker
            tracker = Tracker()
        records = []
        for i, (t_s, block) in enumerate(source):
            if max_blocks is not None and i >= max_blocks:
                break
            res = bearing_fn(block, t_s)
            st = tracker.update(t_s, res, block)
            tr = truth_fn(t_s) if truth_fn is not None else None
            records.append({
                't_s': float(t_s),
                'est_bearing': NAN if res.bearing_deg is None else float(res.bearing_deg),
                'conf': float(res.confidence),
                'smooth': _f(st.bearing_smooth_deg),
                'rate': _f(st.rate_dps),
                'trend': _f(st.trend_dbps),
                'level': _f(st.level_dbfs),
                'floor': _f(st.noise_floor_dbfs),
                'approach': bool(st.approach),
                'true_bearing': NAN if tr is None else float(tr['bearing_deg']),
                'true_dist': NAN if tr is None else float(tr['dist_m']),
                'true_approaching': False if tr is None else bool(tr['approaching']),
            })
    return records


def _f(x):
    return NAN if x is None else float(x)


def summarize(records, kind='other'):
    """Metrics dict for one run. kind: 'approach', 'crossing' or anything else."""
    col = {k: np.array([r[k] for r in records]) for k in records[0]} if records else {}
    if not records:
        return {'blocks': 0}
    has_truth = bool(np.isfinite(col['true_bearing']).any())
    accepted = np.isfinite(col['est_bearing']) & (col['conf'] >= config.CONF_MIN)
    out = {'blocks': len(records), 'accepted_frac': float(accepted.mean()), 'has_truth': has_truth}
    if has_truth:
        est = np.where(accepted, col['est_bearing'], NAN)
        out['rms_bearing_err_deg'] = metrics.rms_error_deg(est, col['true_bearing'])
        out['rms_rate_err_dps'] = metrics.rms_rate_error(
            col['rate'], metrics.true_rate_dps(col['t_s'], col['true_bearing']))
        if kind == 'approach':
            out.update(metrics.approach_result(col['t_s'], col['approach'], col['true_dist'],
                                               col['true_approaching']))
    if kind != 'approach':
        out['false_alarm_frac'] = metrics.false_alarm_fraction(col['approach'])
    out['kind'] = kind
    return out


def print_table(name, s):
    """Print the results of summarize() as a two-column table."""
    rows = [('source', name), ('kind', s.get('kind')), ('blocks', s['blocks'])]
    if s['blocks']:
        rows.append(('accepted bearings', f"{100 * s['accepted_frac']:.0f}%"))
        if s['has_truth']:
            rows.append(('RMS bearing error', f"{s['rms_bearing_err_deg']:.2f} deg"))
            rows.append(('RMS rate error', f"{s['rms_rate_err_dps']:.2f} deg/s"))
        else:
            rows.append(('truth', 'none (error metrics skipped)'))
        if 'detected' in s:
            rows.append((f'detected before {metrics.NEAR_M:g} m', 'yes' if s['detected'] else 'NO'))
            rows.append(('time from start to flag', f"{s['time_to_flag_s']:.2f} s"))
        if 'false_alarm_frac' in s:
            rows.append(('false-alarm time fraction', f"{100 * s['false_alarm_frac']:.2f}%"))
    w = max(len(k) for k, _ in rows)
    print('\n'.join(f'{k:<{w}}  {v}' for k, v in rows))


def save_plot(records, title, path):
    """Plot true vs estimated bearing, rate, level and the flag over time to a PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {k: np.array([r[k] for r in records]) for k in records[0]}
    t = col['t_s']
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ok = np.isfinite(col['est_bearing']) & (col['conf'] >= config.CONF_MIN)
    ax[0].plot(t[ok], col['est_bearing'][ok], '.', ms=3, label='estimated')
    ax[0].plot(t, col['smooth'], '-', lw=1, label='smoothed')
    if np.isfinite(col['true_bearing']).any():
        ax[0].plot(t, col['true_bearing'], 'k--', lw=1, label='true')
    ax[0].set_ylabel('bearing (deg)')
    ax[0].set_ylim(0, 360)
    ax[0].legend(loc='upper right', fontsize=8)
    ax[1].plot(t, col['rate'], label='estimated')
    if np.isfinite(col['true_bearing']).any():
        ax[1].plot(t, metrics.true_rate_dps(t, col['true_bearing']), 'k--', lw=1, label='true')
    for v in (config.RATE_MAX_DPS, -config.RATE_MAX_DPS):
        ax[1].axhline(v, color='gray', lw=0.5)
    ax[1].set_ylabel('rate (deg/s)')
    ax[1].legend(loc='upper right', fontsize=8)
    ax[2].plot(t, col['level'], label='level')
    ax[2].plot(t, col['floor'], lw=1, label='noise floor')
    ax[2].set_ylabel('level (dBFS)')
    ax[2].legend(loc='upper right', fontsize=8)
    ax[3].step(t, col['approach'].astype(int), where='post', label='approach flag')
    if np.isfinite(col['true_bearing']).any():
        ax[3].step(t, col['true_approaching'].astype(int), where='post', color='k', ls='--',
                   lw=1, label='truly approaching')
    ax[3].set_ylabel('flag')
    ax[3].set_xlabel('time (s)')
    ax[3].legend(loc='upper right', fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---- manifest truth for recordings (Section 10.5) ----

def _num(s):
    s = (s or '').strip()
    return float(s) if s else None


def manifest_row(filename, manifest=MANIFEST):
    """Manifest row (dict) for a file's basename, or None if absent or no manifest."""
    try:
        with open(manifest, newline='') as f:
            for row in csv.DictReader(f):
                if row['file'] == pathlib.Path(filename).name:
                    return row
    except FileNotFoundError:
        pass
    return None


def manifest_truth_fn(row):
    """truth_fn(t_s) -> dict from a manifest row.

    Static rows (no move times) are constant. Walks interpolate linearly between
    move_start_s and move_end_s (bearing along the shorter arc, so 303.7 -> 56.3 passes
    through 0), holding the start before and the end after. approaching is True during
    the move of an 'approach' row.
    """
    b0, b1 = _num(row['start_bearing_deg']), _num(row['end_bearing_deg'])
    d0, d1 = _num(row['start_dist_m']), _num(row['end_dist_m'])
    m0, m1 = _num(row['move_start_s']), _num(row['move_end_s'])
    moving = m0 is not None and m1 is not None and m1 > m0
    is_approach = row['kind'] == 'approach'

    def truth(t_s):
        f = float(np.clip((t_s - m0) / (m1 - m0), 0.0, 1.0)) if moving else 0.0
        db = metrics.circular_error_deg(b1, b0)
        return {'bearing_deg': (b0 + f * db) % 360.0,
                'dist_m': d0 + f * (d1 - d0),
                'approaching': bool(is_approach and moving and 0.0 < f < 1.0)}
    return truth


# ---- command line ----

def _kind_from_name(name):
    for k, v in (('approach', 'approach'), ('cross', 'crossing'), ('recede', 'recede')):
        if name.startswith(k):
            return v
    return 'other'


def _load_source(spec):
    """Returns (source, truth_fn or None, kind, label)."""
    mode, _, arg = spec.partition(':')
    if mode == 'synth':
        from dac import synth
        from dac.sources import SyntheticSource
        src = SyntheticSource(synth.make_scenario(arg))
        return src, src.truth, _kind_from_name(arg), arg
    if mode == 'wav':
        from dac.sources import WavSource
        row = manifest_row(arg)
        if row is None:
            print(f'note: no manifest row for {pathlib.Path(arg).name}; running without truth')
        truth = manifest_truth_fn(row) if row else None
        return WavSource(arg), truth, (row['kind'] if row else 'other'), pathlib.Path(arg).stem
    sys.exit(f"bad --source {spec!r}: use synth:<scenario> or wav:<path>")


def _load_bearing(args, truth_fn):
    if args.bearing == 'stub':
        if truth_fn is None:
            sys.exit('--bearing stub needs ground truth (a synth scenario or a wav in the manifest)')
        from dac.stub_bearing import StubBearing
        return StubBearing(lambda t: truth_fn(t)['bearing_deg'], noise_deg=args.noise_deg,
                           dropout=args.dropout, seed=args.seed).estimate
    from dac import bearing
    fn = getattr(bearing, 'estimate_bearing', None)
    if fn is None:
        sys.exit('--bearing real: dac.bearing.estimate_bearing does not exist yet (module is empty)')
    return fn


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--source', required=True, help='synth:<scenario> or wav:<path>')
    p.add_argument('--bearing', choices=['stub', 'real'], default='stub')
    p.add_argument('--noise-deg', type=float, default=5.0, help='stub bearing noise (deg)')
    p.add_argument('--dropout', type=float, default=0.05, help='stub dropout probability')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default=str(OUT_DIR))
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args(argv)

    source, truth_fn, kind, label = _load_source(args.source)
    records = evaluate(source, _load_bearing(args, truth_fn), truth_fn)
    if not records:
        sys.exit('source produced no blocks')
    print_table(args.source, summarize(records, kind))
    if not args.no_plot:
        path = pathlib.Path(args.out_dir) / f'{label}_{args.bearing}.png'
        save_plot(records, f'{label} ({args.bearing})', path)
        print(f'plot: {path}')


if __name__ == '__main__':
    main()
