"""Leg-paired comparison of controller runs (the analysis behind Tables III and IV).

Each --run is NAME=PATH, where PATH is a controller log (CSV) or a folder containing one.
Several paths separated by commas (NAME=p1,p2,p3) are averaged per leg, e.g. three seeds.

Example (Table IV, doubled wind):
  python analysis/compare_runs.py --length 540 --ref A \
      --run A=logs/S1_strong_A --run Lt=logs/S2b_strong_LT --run Lt+=logs/R3_strong_eval \
      --out results/strong_legs.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from legs import cut_length, leg_metrics, load_run, run_metrics

METRICS = ['e_settled', 'cte', 't_settle2', 'overshoot', 'effort', 'sat', 'tau_settled', 'lyap', 'phi']


def parse_runs(items):
    runs = {}
    for it in items:
        name, paths = it.split('=', 1)
        runs[name] = paths.split(',')
    return runs


def wilcoxon_p(a, b):
    d = a - b
    return stats.wilcoxon(a, b).pvalue if (d != 0).any() else 1.0


def rank_biserial(a, b):
    d = a - b
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True, help='NAME=PATH[,PATH...]')
    ap.add_argument('--ref', help='reference run name for the paired tests (default: first run)')
    ap.add_argument('--length', type=float, default=540.0, help='metres from the start to analyse (default 540; 0 = whole run)')
    ap.add_argument('--max-obstacle', type=float, default=0.2, help='drop legs with more obstacle-avoidance time than this share')
    ap.add_argument('--out', help='CSV file for the per-leg table')
    ap.add_argument('--skip', type=float, default=60.0, help='seconds dropped at the start of every log (default 60)')
    a = ap.parse_args()

    runs = parse_runs(a.run)
    ref = a.ref or next(iter(runs))
    L = None if a.length <= 0 else a.length

    per_run, whole = {}, {}
    for name, paths in runs.items():
        tables = []
        for p in paths:
            d, dist = cut_length(load_run(p, a.skip), L)
            tables.append(pd.DataFrame(leg_metrics(d, dist)).T)
            whole.setdefault(name, []).append(run_metrics(d))
        common_legs = set.intersection(*[set(t.index) for t in tables])
        per_run[name] = sum(t.loc[sorted(common_legs)] for t in tables) / len(tables)

    legs = set.intersection(*[set(t.index) for t in per_run.values()])
    legs = [g for g in sorted(legs) if max(per_run[n].loc[g, 'obstacle'] for n in per_run) < a.max_obstacle]
    print(f'legs per run: { {n: len(t) for n, t in per_run.items()} }  common: {len(legs)}')

    long = pd.concat([per_run[n].loc[legs].assign(cond=n).rename_axis('leg').reset_index() for n in per_run])
    if a.out:
        long.to_csv(a.out, index=False)
        print('per-leg table written to', a.out)

    rows = []
    for m in METRICS:
        W = long.pivot_table(index='leg', columns='cond', values=m)[list(runs)]
        row = {'metric': m}
        for n in runs:
            row[n] = W[n].median()
        for n in runs:
            if n == ref:
                continue
            row[f'{n}<{ref} legs'] = int((W[n] < W[ref]).sum())
            row[f'p({n} vs {ref})'] = wilcoxon_p(W[n], W[ref])
            row[f'r_rb({n} vs {ref})'] = rank_biserial(W[n], W[ref])
        rows.append(row)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print('\nmedians over the common legs, and paired tests against', ref)
    print(pd.DataFrame(rows).set_index('metric').round(4).to_string())

    print('\nwhole-run shares (mean over the paths of each run)')
    print(pd.DataFrame({n: pd.DataFrame(v).mean() for n, v in whole.items()}).round(3).to_string())
    print('\nNote: legs of one run are not independent, so leg-level p-values are optimistic.')


if __name__ == '__main__':
    main()
