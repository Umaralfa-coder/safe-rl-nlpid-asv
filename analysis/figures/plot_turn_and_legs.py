"""Fig. 4-style figure: (a) heading error after one waypoint switch for several runs,
(b, c) one per-leg metric per run as dots with the median.

  python analysis/figures/plot_turn_and_legs.py --leg "16.2,84.4" \
      --run A=logs/A_s7 --run La=logs/EST_eval_s7 --run Lt=logs/TURN_eval \
      --legs results/calm_legs.csv --metrics cte,overshoot --out results/fig_e2.pdf

--legs is the per-leg CSV written by compare_runs.py --out (same run names).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from legs import leg_key, load_run  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ['#eb6834', '#2a78d6', '#1baf7a', '#4a3aa7', '#c2410c', '#0e7490']
LABELS = {'cte': 'cross-track (m)', 'overshoot': 'overshoot ($^\\circ$)', 'e_settled': 'settled |e| ($^\\circ$)',
          'effort': 'effort (N s/m)', 't_settle2': 'time within 2$^\\circ$ (s)'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, help='NAME=PATH, in plotting order')
    ap.add_argument('--leg', required=True, help='target waypoint "x,y" of the switch to plot in (a)')
    ap.add_argument('--legs', required=True, help='per-leg CSV from compare_runs.py --out')
    ap.add_argument('--metrics', default='cte,overshoot')
    ap.add_argument('--out', default='fig_turn_legs.pdf')
    a = ap.parse_args()
    runs = dict(it.split('=', 1) for it in a.run)
    col = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(runs)}
    mets = a.metrics.split(',')
    plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#d9d8d4', 'grid.linewidth': 0.5, 'font.family': 'serif'})
    fig = plt.figure(figsize=(3.45, 2.95))
    gs = fig.add_gridspec(2, len(mets), height_ratios=[1.05, 1], hspace=0.85, wspace=0.42)
    ax = fig.add_subplot(gs[0, :])
    for n, p in runs.items():
        d = load_run(p); K = leg_key(d)
        idx = np.where((K == a.leg) & np.r_[True, K[1:] != K[:-1]])[0]
        if len(idx) == 0:
            print(f'{n}: leg {a.leg} not found'); continue
        i = idx[0]; s = d.iloc[max(i - 10, 0):i + 170]
        t = s.time_elapsed.values - d.time_elapsed.values[i]
        e = s.heading_error_deg.values
        ax.plot(t, e * np.sign(d.heading_error_deg.values[min(i + 2, len(d) - 1)]), color=col[n], lw=1.1, label=n)
    ax.axhline(0, color='#52514e', lw=0.6); ax.set_ylim(-3, 12); ax.set_xlim(-0.5, 7.2)
    ax.set_xlabel('time after the waypoint switch (s)'); ax.set_ylabel('heading error ($^\\circ$)')
    ax.set_title('(a) Turn-in after a switch (off scale at the start)', loc='left')
    ax.legend(frameon=False, loc='upper right')
    legs = pd.read_csv(a.legs)
    for j, m in enumerate(mets):
        ax = fig.add_subplot(gs[1, j]); rng = np.random.default_rng(1)
        W = legs.pivot_table(index='leg', columns='cond', values=m)
        for x, n in enumerate(runs):
            if n not in W:
                continue
            v = W[n].dropna().values
            ax.scatter(x + rng.uniform(-0.17, 0.17, len(v)), v, s=5, color=col[n], lw=0, alpha=0.85, zorder=2)
            ax.plot([x - 0.28, x + 0.28], [np.median(v)] * 2, color='#0b0b0b', lw=1.5, zorder=3)
        ax.set_xticks(range(len(runs))); ax.set_xticklabels(list(runs)); ax.grid(axis='x', visible=False)
        ax.set_ylabel(LABELS.get(m, m)); ax.set_title(f'({chr(98 + j)}) {m}', loc='left')
    fig.savefig(a.out, bbox_inches='tight')
    print('saved', a.out)


if __name__ == '__main__':
    main()
