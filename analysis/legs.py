"""Load controller logs and compute the per-leg and per-run metrics used in the paper.

A *leg* is the stretch between two consecutive waypoint switches. Runs of the same
mission fly the same waypoints, so legs are matched across runs by their target
waypoint (rounded to 0.1 m). Definitions follow Section V of the paper:

  e_settled    median |heading error| over the second half of the leg (deg)
  cte          median |cross-track error| over the leg (m)
  t_settle2    time after the switch until |e| stays below 2 deg (s)
  overshoot    max |e| after the first zero crossing of e (deg)
  effort       integral of (|F1|+|F2|) dt per metre sailed (N s/m)
  sat          share of steps with a thruster at its limit (%)
  tau_settled  std of the yaw torque over the second half of the leg (N)
  lyap         share of steps violating the Lyapunov decrease (%)
  phi          share of steps with safety index phi > 0 (%)
"""
import glob
import os

import numpy as np
import pandas as pd

DT = 0.05            # controller period (20 Hz)
F_MAX = 800.0        # thruster limit (N)


def find_csv(path):
    """Accept a CSV file or a folder that contains one controller_*.csv log."""
    if os.path.isfile(path):
        return path
    files = sorted(glob.glob(os.path.join(path, '*controller*.csv')))
    if not files:
        raise FileNotFoundError(f'no controller_*.csv in {path}')
    if len(files) > 1:
        files.sort(key=os.path.getmtime)
        print(f'[legs] {path}: {len(files)} logs found, using the newest: {os.path.basename(files[-1])}')
    return files[-1]


def load_run(path, skip_s=60.0):
    """Read a controller log and drop the first `skip_s` seconds (start-up turn)."""
    d = pd.read_csv(find_csv(path))
    d = d[d.time_elapsed >= d.time_elapsed.iloc[0] + skip_s].reset_index(drop=True)
    return d


def distance(d):
    return np.r_[0.0, np.cumsum(np.hypot(np.diff(d.x.values), np.diff(d.y.values)))]


def cut_length(d, length_m):
    """Keep the first `length_m` metres sailed (None = whole run)."""
    dist = distance(d)
    if length_m is None:
        return d, dist
    keep = dist <= length_m
    return d[keep].reset_index(drop=True), dist[keep]


def leg_key(d):
    return (d.target_x.round(1).astype(str) + ',' + d.target_y.round(1).astype(str)).values


def leg_metrics(d, dist, min_rows=60, min_dist=2.0):
    """Metrics for every leg of one run. Returns {leg_key: dict}; a waypoint flown twice keeps its first leg."""
    K = leg_key(d)
    seg = np.cumsum(np.r_[True, K[1:] != K[:-1]]) - 1
    out = {}
    for s in range(1, seg.max() + 1):           # the first segment starts mid-leg
        m = np.where(seg == s)[0]
        if len(m) < min_rows or K[m[0]] in out:
            continue
        g = d.iloc[m]
        t = g.time_elapsed.values - g.time_elapsed.values[0]
        dd = dist[m[-1]] - dist[m[0]]
        if dd < min_dist:
            continue
        e = g.heading_error_deg.values
        e0 = e[:10][np.argmax(np.abs(e[:10]))]
        zc = np.where(np.sign(e) != np.sign(e0))[0]
        big = np.where(np.abs(e) > 2)[0]
        thr = (g.left_thrust.abs() >= F_MAX - 1) | (g.right_thrust.abs() >= F_MAX - 1)
        st = g[t >= 0.5 * t[-1]]
        out[K[m[0]]] = dict(
            e_settled=float(st.heading_error_deg.abs().median()),
            cte=float(g.cross_track_error.abs().median()),
            t_settle2=float(t[big[-1]]) if len(big) else 0.0,
            overshoot=float(np.abs(e[zc[0]:]).max()) if len(zc) else 0.0,
            effort=float(((g.left_thrust.abs() + g.right_thrust.abs()) * DT).sum() / dd),
            sat=float(thr.mean() * 100),
            tau_settled=float(st.tau_r.std()),
            lyap=float(g.lyap_violation.mean() * 100),
            phi=float((g.phi > 0).mean() * 100),
            duration=float(t[-1]),
            obstacle=float(g.obstacle_detected.mean()),
        )
    return out


def run_metrics(d):
    """Whole-run shares (%) used in Table IV of the paper."""
    nt = d.ref_transient == 0
    thr = (d.left_thrust.abs() >= F_MAX - 1) | (d.right_thrust.abs() >= F_MAX - 1)
    return dict(
        sat_pct=float(thr.mean() * 100),
        lyap_pct=float(d.lyap_violation.mean() * 100),
        phi_pos_pct=float((d.phi > 0).mean() * 100),
        phi_outside_transients_pct=float(((d.phi > 0) & nt).mean() * 100),
        shield_interventions_pct=float(d.intervention.mean() * 100),
        gate_frozen_pct=float(d.disturbance_gate.mean() * 100) if 'disturbance_gate' in d else float('nan'),
        cte_median=float(d.cross_track_error.abs().median()),
        effort_kNs=float(((d.left_thrust.abs() + d.right_thrust.abs()) * DT).sum() / 1e3),
    )
