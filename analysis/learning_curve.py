"""Learning curve from a saved policy file (the controller stores every episode return).

  python analysis/learning_curve.py logs/policy_turnL_strong.json --split 53 --png results/curve.png

--split N marks where a new phase starts (e.g. the 53 calm-water episodes of L_t before
learning resumed in doubled wind) and prints quarter means of the phase after it.
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('policy')
    ap.add_argument('--split', type=int, default=0, help='episode index where a new phase starts')
    ap.add_argument('--png', help='save a plot of the returns')
    a = ap.parse_args()
    p = json.load(open(a.policy))
    r = np.array(p['return_history'])
    print(f"{a.policy}: {len(r)} episodes, {p['iteration']} ES updates, sigma {p['sigma']}, lr {p['lr']}")
    phases = [r[:a.split], r[a.split:]] if a.split else [r]
    for k, ph in enumerate(phases):
        if len(ph) == 0:
            continue
        q = np.array_split(ph, 4)
        print(f'phase {k}: {len(ph)} episodes; quarter means ' + ', '.join(f'{x.mean():.3f}' for x in q))
    if a.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 2.4))
        ax.plot(r, '.', ms=3, color='#2a78d6')
        k = 8
        if len(r) >= k:
            ax.plot(np.arange(k - 1, len(r)), np.convolve(r, np.ones(k) / k, 'valid'), color='#0b0b0b', lw=1.2, label=f'{k}-episode mean')
        if a.split:
            ax.axvline(a.split - 0.5, color='#eb6834', lw=1, ls='--', label='new phase')
        ax.set_xlabel('episode'); ax.set_ylabel('return (= -cost)'); ax.legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(a.png, dpi=200)
        print('saved', a.png)


if __name__ == '__main__':
    main()
