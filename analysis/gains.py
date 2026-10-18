"""Distribution of the applied NLPID parameters and gains in one or more runs.

Shows whether a learned policy applies constant gains or changes them with the state
(5th percentile, median and 95th percentile over the settled steps).

  python analysis/gains.py --run Lt=logs/S2b_strong_LT --run Lt+=logs/R3_strong_eval
"""
import argparse

import pandas as pd

from legs import load_run

COLS = ['mu_p', 'mu_d', 'mu_i', 'eps_p', 'eps_d', 'eps_i', 'lambda', 'boundary_layer', 'robust_gain',
        'heading_K_p', 'heading_K_d', 'heading_K_i']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, help='NAME=PATH')
    ap.add_argument('--skip-rows', type=int, default=400, help='rows dropped after the start (rate limiter settling)')
    a = ap.parse_args()
    pd.set_option('display.width', 220)
    for it in a.run:
        name, path = it.split('=', 1)
        d = load_run(path).iloc[a.skip_rows:]
        s = d[d.ref_transient == 0]
        t = pd.DataFrame({'p5': s[COLS].quantile(0.05), 'median': s[COLS].median(), 'p95': s[COLS].quantile(0.95)})
        print(f'\n{name} ({path}), mode {d.rl_mode.iloc[0]}')
        print(t.round(3).T.to_string())


if __name__ == '__main__':
    main()
