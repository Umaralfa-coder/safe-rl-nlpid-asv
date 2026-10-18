"""Replay the water-quality probe and GP map on logged trajectories (Section VI-C, E3).

All runs in one call sample the same hidden concentration fields with the same probe
model and the same GP; only the vehicle motion differs. Map RMSE is computed on the grid
cells (2 m) that lie within 2 m of the path in every run.

Probe variants:
  raw      5-s first-order lag + motion-dependent noise (as in wq_sim_node)
  nolag    the same noise without the lag (idealised fast probe)
  lagcorr  raw readings, smoothed by a first-order filter (Tf) and with the known lag
           inverted: c_hat = y_s + tau * dy_s/dt

Example (E1, one seed):
  python analysis/wq_replay.py --run A=logs/A_s7 --run BU=logs/BU_s7 --run CU=logs/CU_s7 \
      --fields 10 --out results/wq_s7.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ros2_ws', 'src', 'safe_rl_nlpid', 'safe_rl_nlpid'))
from wq_sim_node import OnlineGP, PlumeField, ProbeSensor, mission_bounds  # noqa: E402

from legs import cut_length, load_run  # noqa: E402

STEP = 10          # use every 10th log row: 2 Hz sampling at 20 Hz logging
TAU = 5.0          # probe lag (s)


def sample(d, field, seed, tau=TAU, k_r=1.0, k_dF=0.05):
    """Simulate the probe along one trajectory. Returns times, positions, readings and noise std."""
    sensor = ProbeSensor(tau_s=tau, sigma0=0.1, k_r=k_r, k_dF=k_dF, seed=seed)
    t = d.time_elapsed.values; x = d.x.values; y = d.y.values; r = d.yaw_rate.values
    F = d.left_thrust.abs().values + d.right_thrust.abs().values
    dF = 0.0; Fp = None; tp = None; rows = []
    for i in range(0, len(d), STEP):
        dt = 0.5 if tp is None else t[i] - tp
        if Fp is not None and dt > 0:
            dF = 0.5 * dF + 0.5 * (F[i] - Fp) / dt
        Fp, tp = F[i], t[i]
        c = float(field(x[i], y[i], 0.0))
        sensor.step(c, dt)
        z, sig = sensor.read(r[i], dF)
        rows.append((t[i], x[i], y[i], z, sig))
    return np.array(rows)


def lag_correct(t, z, tf, tau=TAU):
    ys = np.empty_like(z); ch = np.empty_like(z); ys[0] = ch[0] = z[0]
    for i in range(1, len(z)):
        dt = max(t[i] - t[i - 1], 1e-3)
        ys[i] = ys[i - 1] + (1 - np.exp(-dt / tf)) * (z[i] - ys[i - 1])
        ch[i] = ys[i] + tau * (ys[i] - ys[i - 1]) / dt
    return ch


def map_rmse(rows, z, P, mask, truth):
    gp = OnlineGP(length=10.0, noise_std=0.3, cell=1.5)
    for (_, x, y, _, _), zi in zip(rows, z):
        gp.add(x, y, zi)
    m, _ = gp.predict(P)
    return float(np.sqrt(np.mean((m[mask] - truth[mask]) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True, help='NAME=PATH (log or folder)')
    ap.add_argument('--fields', type=int, default=10, help='number of random fields (seeds 1..N)')
    ap.add_argument('--length', type=float, default=560.0, help='metres from the start (default 560)')
    ap.add_argument('--tf', type=float, default=1.0, help='smoothing time constant of the lag correction (s)')
    ap.add_argument('--out', help='CSV with one row per run, field and probe variant')
    ap.add_argument('--skip', type=float, default=60.0, help='seconds dropped at the start of every log (default 60)')
    a = ap.parse_args()

    runs = dict(it.split('=', 1) for it in a.run)
    D = {n: cut_length(load_run(p, a.skip), a.length)[0] for n, p in runs.items()}
    allxy = np.vstack([np.column_stack([d.x, d.y]) for d in D.values()])
    b = mission_bounds(allxy, 10.0)
    gx = np.arange(b[0][0], b[0][1] + 1e-6, 2.0); gy = np.arange(b[1][0], b[1][1] + 1e-6, 2.0)
    GX, GY = np.meshgrid(gx, gy); P = np.column_stack([GX.ravel(), GY.ravel()])
    mask = np.ones(len(P), bool)
    for d in D.values():
        xy = np.column_stack([d.x, d.y])[::10]; near = np.zeros(len(P), bool)
        for i in range(0, len(xy), 256):
            near |= (((P[:, None, :] - xy[None, i:i + 256, :]) ** 2).sum(-1) <= 4.0).any(1)
        mask &= near
    print(f'{mask.sum()} common cells')

    out = []
    for fs in range(1, a.fields + 1):
        field = PlumeField(b, seed=fs); truth = field(P[:, 0], P[:, 1], 0.0)
        for n, d in D.items():
            raw = sample(d, field, fs)
            fast = sample(d, field, fs, tau=1e-3)
            corr = lag_correct(raw[:, 0], raw[:, 3], a.tf)
            for var, rows, z in (('raw', raw, raw[:, 3]), ('nolag', fast, fast[:, 3]), ('lagcorr', raw, corr)):
                out.append(dict(run=n, field=fs, probe=var, rmse=map_rmse(rows, z, P, mask, truth),
                                noisy_share_pct=float((raw[:, 4] > 0.3).mean() * 100), field_std=float(truth[mask].std())))
        print(f'field {fs} done', flush=True)
    df = pd.DataFrame(out)
    if a.out:
        df.to_csv(a.out, index=False)
    names = list(runs); ref = names[0]
    print('\nmean map RMSE (field std %.2f)' % df.field_std.mean())
    print(df.pivot_table(index='probe', columns='run', values='rmse')[names].round(3).to_string())
    print('\nshare of noisy samples (sigma > 0.3), %')
    print(df[df.probe == 'raw'].groupby('run').noisy_share_pct.mean()[names].round(1).to_string())
    for var in ('raw', 'nolag', 'lagcorr'):
        w = df[df.probe == var].pivot(index='field', columns='run', values='rmse')
        for n in names[1:]:
            p = wilcoxon(w[n], w[ref]).pvalue if (w[n] != w[ref]).any() else 1.0
            print(f'{var:8s} {n} < {ref} on {(w[n] < w[ref]).sum()}/{len(w)} fields, p = {p:.3g}')
    print('\nNote: the fields share each trajectory, so these p-values are optimistic.')


if __name__ == '__main__':
    main()
