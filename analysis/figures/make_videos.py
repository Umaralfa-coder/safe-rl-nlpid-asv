"""Mission replay videos from controller logs (no simulator needed).

Runs are synchronised by distance sailed, so the boats stay side by side even when one is
slower. Left: the whole mission with the trails; right: a close-up that follows the first run;
bottom: cross-track error against distance.

  python analysis/figures/make_videos.py replay --title "Calm water" --out video/calm.mp4 \
      --run "A (hand-tuned)=logs/A_s8" --run "L_t (learned)=logs/TURN_eval"
  python analysis/figures/make_videos.py wqmap --run logs/A_s8 --out video/wq_map.mp4

Needs ffmpeg on the PATH.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from legs import cut_length, distance, load_run  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

COLS = ['#eb6834', '#4a3aa7', '#1baf7a', '#2a78d6']
INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#d9d8d4', '#fbfbfa'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': BG, 'axes.facecolor': BG, 'font.family': 'DejaVu Sans'})


def waypoints(d):
    k = d.target_x.round(1).astype(str) + ',' + d.target_y.round(1).astype(str)
    first = ~k.duplicated()
    return d.loc[first, ['target_x', 'target_y']].values


def align(D):
    """Start every run at the first waypoint switch that all runs share."""
    keys = [(d.target_x.round(1).astype(str) + ',' + d.target_y.round(1).astype(str)).values for _, d in D]
    common = set(keys[0]).intersection(*map(set, keys[1:]))
    k0 = next(k for i, k in enumerate(keys[0]) if k in common and (i == 0 or keys[0][i - 1] != k) and i > 0)
    out = []
    for (n, d), K in zip(D, keys):
        i = int(np.argmax(K == k0))
        out.append((n, d.iloc[i:].reset_index(drop=True)))
    return out


def interp_run(d, s_grid):
    s = distance(d)
    keep = np.r_[True, np.diff(s) > 1e-6]
    s, dd = s[keep], d[keep]
    yaw = np.unwrap(np.radians(dd.yaw_deg.values))
    f = lambda v: np.interp(s_grid, s, v)
    # On long transit legs the guidance switches to pursuit and logs no cross-track error (0):
    # show those stretches (exact zeros lasting more than 2 s) as gaps, not as perfect tracking.
    c = dd.cross_track_error.abs().values.astype(float)
    z = c < 1e-6; blank = np.zeros(len(c), bool); i = 0
    while i < len(c):
        if z[i]:
            j = i
            while j < len(c) and z[j]:
                j += 1
            if j - i > 40:
                blank[i:j] = True
            i = j
        else:
            i += 1
    cte = f(np.where(blank, np.nan, c))
    cte[f(blank.astype(float)) > 0.5] = np.nan
    return dict(x=f(dd.x.values), y=f(dd.y.values), yaw=np.degrees(f(yaw)),
                cte=cte, e=f(dd.heading_error_deg.values),
                t=f(dd.time_elapsed.values - dd.time_elapsed.values[0]))


def boat(ax, color):
    (m,) = ax.plot([], [], marker=(3, 0, 0), ms=11, color=color, mec=INK, mew=0.6, ls='none', zorder=5)
    return m


def replay(a):
    runs = [it.split('=', 1) for it in a.run]
    D = align([(n, load_run(p, a.skip)) for n, p in runs])
    D = [(n, cut_length(d, a.length)[0]) for n, d in D]
    smax = min(distance(d)[-1] for _, d in D)
    s_grid = np.arange(0, smax, a.step)
    R = [(n, interp_run(d, s_grid)) for n, d in D]
    wp = waypoints(D[0][1])
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], width_ratios=[1.35, 1], hspace=0.38, wspace=0.08,
                          left=0.06, right=0.98, top=0.84, bottom=0.12)
    axm, axz, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    fig.text(0.06, 0.955, a.title, fontsize=12, color=INK, fontweight='bold', va='top')
    if a.subtitle:
        fig.text(0.06, 0.905, a.subtitle, fontsize=8.5, color=INK2, va='top')
    for ax in (axm, axz):
        ax.plot(wp[:, 0], wp[:, 1], color='#bdbcb8', lw=1.0, zorder=1)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(GRID)
    xs = np.r_[np.concatenate([r['x'] for _, r in R]), wp[:, 0]]; ys = np.r_[np.concatenate([r['y'] for _, r in R]), wp[:, 1]]
    axm.set_xlim(xs.min() - 5, xs.max() + 5); axm.set_ylim(ys.min() - 5, ys.max() + 5)
    axm.set_title('mission (planned path in grey)', fontsize=8.5, color=INK2, loc='left')
    axz.set_title('close-up (30 m)', fontsize=8.5, color=INK2, loc='left')
    trails, trailz, boats, boatz, lines, labels = [], [], [], [], [], []
    for i, (n, r) in enumerate(R):
        c = COLS[i % len(COLS)]
        trails.append(axm.plot([], [], color=c, lw=1.2, zorder=3)[0])
        trailz.append(axz.plot([], [], color=c, lw=1.6, zorder=3)[0])
        boats.append(boat(axm, c)); boatz.append(boat(axz, c))
        lines.append(axc.plot([], [], color=c, lw=1.1, label=n)[0])

    axc.set_xlim(0, s_grid[-1]); axc.set_ylim(0, a.cte_max)
    axc.set_xlabel('distance sailed (m)'); axc.set_ylabel('cross-track (m)')
    axc.grid(True, color=GRID, lw=0.5)
    leg = axc.legend(loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, ncol=len(R), fontsize=8.5,
                     borderaxespad=0.1, handlelength=1.5)
    labels = leg.get_texts()
    axc.text(0.0, -0.55, 'gaps: long transit legs, where the guidance flies straight to the next waypoint and logs no cross-track error',
             transform=axc.transAxes, fontsize=7, color=INK2)
    clock = axm.text(0.01, 0.01, '', transform=axm.transAxes, fontsize=8, color=INK2)
    frames = range(0, len(s_grid), a.every)
    writer = FFMpegWriter(fps=a.fps, bitrate=1600, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with writer.saving(fig, a.out, dpi=100):
        for k in frames:
            j0 = max(0, k - int(a.trail / a.step))
            for i, (n, r) in enumerate(R):
                trails[i].set_data(r['x'][:k + 1], r['y'][:k + 1])
                trailz[i].set_data(r['x'][j0:k + 1], r['y'][j0:k + 1])
                for b in (boats[i], boatz[i]):
                    b.set_data([r['x'][k]], [r['y'][k]]); b.set_marker((3, 0, r['yaw'][k] - 90))
                lines[i].set_data(s_grid[:k + 1], r['cte'][:k + 1])
                med = np.nanmedian(r['cte'][:k + 1]) if np.isfinite(r['cte'][:k + 1]).any() else np.nan
                labels[i].set_text(f"{n}   median so far {med:.2f} m")
            cx, cy = R[0][1]['x'][k], R[0][1]['y'][k]
            axz.set_xlim(cx - 15, cx + 15); axz.set_ylim(cy - 15, cy + 15)
            clock.set_text(f"{s_grid[k]:.0f} m   ~{R[0][1]['t'][k]:.0f} s simulated")
            writer.grab_frame()
    print('saved', a.out, f'({len(frames)} frames)')


def wqmap(a):
    sys.path.insert(0, os.path.join(HERE, '..'))
    from wq_replay import lag_correct, sample  # noqa: E402
    from wq_sim_node import OnlineGP, PlumeField, mission_bounds  # noqa: E402
    d = cut_length(load_run(a.run, a.skip), a.length)[0]
    b = mission_bounds(np.column_stack([d.x, d.y]), 10.0)
    gx = np.arange(b[0][0], b[0][1] + 1e-6, 2.0); gy = np.arange(b[1][0], b[1][1] + 1e-6, 2.0)
    GX, GY = np.meshgrid(gx, gy); P = np.column_stack([GX.ravel(), GY.ravel()])
    field = PlumeField(b, seed=a.field); truth = field(P[:, 0], P[:, 1], 0.0).reshape(GX.shape)
    rows = sample(d, field, a.field); corr = lag_correct(rows[:, 0], rows[:, 3], 1.0)
    gps = [OnlineGP(length=10.0, noise_std=0.3, cell=1.5), OnlineGP(length=10.0, noise_std=0.3, cell=1.5)]
    vmin, vmax = np.percentile(truth, 2), np.percentile(truth, 98)
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 4.2), dpi=100)
    fig.subplots_adjust(left=0.02, right=0.93, top=0.8, bottom=0.04, wspace=0.05)
    fig.suptitle('Water-quality map built along the path', x=0.02, ha='left', fontsize=12, fontweight='bold', color=INK)
    fig.text(0.02, 0.86, 'same trajectory and field; only the probe processing differs (5-s probe lag, GP map)',
             fontsize=8.5, color=INK2)
    ext = [gx[0], gx[-1], gy[0], gy[-1]]
    titles = ['true field', 'raw 5-s probe', 'lag-corrected probe']
    ims = []
    for ax, t in zip(axs, titles):
        ims.append(ax.imshow(np.full(GX.shape, np.nan), origin='lower', extent=ext, vmin=vmin, vmax=vmax, cmap='viridis'))
        ax.set_title(t, fontsize=9, color=INK, loc='left'); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    ims[0].set_data(truth)
    cb = fig.colorbar(ims[0], ax=axs, fraction=0.015, pad=0.01)
    cb.set_label('concentration (a.u.)', fontsize=8); cb.outline.set_visible(False)
    paths = [ax.plot([], [], color='white', lw=0.8)[0] for ax in axs]
    errs = [axs[i].text(0.02, 0.02, '', transform=axs[i].transAxes, color='white', fontsize=8.5,
                        bbox=dict(facecolor='#00000088', lw=0, pad=2)) for i in (1, 2)]
    near = np.zeros(len(P), bool)
    writer = FFMpegWriter(fps=a.fps, bitrate=1600, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n = len(rows); chunk = max(1, n // a.frames)
    with writer.saving(fig, a.out, dpi=100):
        for k in range(chunk, n + chunk, chunk):
            k = min(k, n)
            for i in range(k - chunk, k):
                gps[0].add(rows[i, 1], rows[i, 2], rows[i, 3]); gps[1].add(rows[i, 1], rows[i, 2], corr[i])
                near |= (P[:, 0] - rows[i, 1]) ** 2 + (P[:, 1] - rows[i, 2]) ** 2 <= 36.0
            for j, gp in enumerate(gps):
                m, _ = gp.predict(P); M = np.where(near, m, np.nan).reshape(GX.shape)
                ims[j + 1].set_data(M)
                rm = np.sqrt(np.nanmean((m[near] - truth.ravel()[near]) ** 2))
                errs[j].set_text(f'RMSE {rm:.2f}')
            for p in paths:
                p.set_data(rows[:k, 1], rows[:k, 2])
            writer.grab_frame()
        for _ in range(a.fps * 2):
            writer.grab_frame()
    print('saved', a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('replay')
    r.add_argument('--run', action='append', required=True, help='"LABEL=PATH", first run is followed by the close-up')
    r.add_argument('--title', default='Mission replay'); r.add_argument('--subtitle', default='')
    r.add_argument('--out', default='video/replay.mp4')
    r.add_argument('--length', type=float, default=540.0); r.add_argument('--skip', type=float, default=60.0)
    r.add_argument('--step', type=float, default=0.25, help='metres between interpolated samples')
    r.add_argument('--every', type=int, default=3, help='samples per video frame')
    r.add_argument('--trail', type=float, default=25.0, help='trail length in the close-up (m)')
    r.add_argument('--cte-max', type=float, default=4.0); r.add_argument('--fps', type=int, default=25)
    w = sub.add_parser('wqmap')
    w.add_argument('--run', required=True); w.add_argument('--out', default='video/wq_map.mp4')
    w.add_argument('--field', type=int, default=3); w.add_argument('--frames', type=int, default=180)
    w.add_argument('--length', type=float, default=560.0); w.add_argument('--skip', type=float, default=60.0)
    w.add_argument('--fps', type=int, default=20)
    a = ap.parse_args()
    replay(a) if a.cmd == 'replay' else wqmap(a)


if __name__ == '__main__':
    main()
