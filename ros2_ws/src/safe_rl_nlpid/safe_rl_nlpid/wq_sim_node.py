#!/usr/bin/env python3
"""
wq_sim_node.py - simulated water-quality sensing and online GP mapping for the WAM-V in VRX.

Runs next to buran_controller.py without changing it. It
  1. defines a hidden "true" concentration field c(x, y) over the mission area
     (a few Gaussian plumes on a smooth background gradient, reproducible by seed),
  2. simulates a real in-situ probe along the boat track:
       first-order response lag   dc_m/dt = (c(x,y) - c_m) / tau_s
       noise that grows with motion  sigma = sigma0 + k_r*|r| + k_dF*|dF/dt|/100
     and samples it at sample_hz,
  3. builds a Gaussian-process map online from the samples (pure numpy, no sklearn),
  4. scores the map against the true field (RMSE over the surveyed swath and over the
     whole area, mean posterior std) together with energy and distance travelled,
  5. publishes /wq/measurement and /wq/metrics (JSON strings) and writes CSV logs plus a
     final map figure to $SAFE_RL_LOG_DIR/wq/ (default ~/seal_ws/logs/wq/).

Frames: positions come from /planning/current_target ('current_position'), the same local
frame the controller and planner use. Waypoints (/planning/waypoints) define the area.

Everything the paper needs to vary is a ROS parameter (see declare_parameter calls).
The core classes (PlumeField, ProbeSensor, OnlineGP, MapScorer) do not import ROS and are
tested offline by test_wq_offline.py.
"""
import csv
import json
import math
import os
import time

import numpy as np


# =============================================================================
# CORE (ROS-free)
# =============================================================================
class PlumeField:
    """Hidden true concentration field, in arbitrary units (e.g. NTU for turbidity).

    c(x, y, t) = background + gradient . p + sum_k A_k exp(-|p - (m_k + v t)|^2 / (2 s_k^2))
    """

    def __init__(self, bounds, seed=1, n_plumes=3, amp=(4.0, 10.0), sigma=(8.0, 20.0),
                 background=5.0, grad_mag=0.03, drift=(0.0, 0.0)):
        (x0, x1), (y0, y1) = bounds
        rng = np.random.default_rng(seed)
        self.bounds = bounds
        self.background = background
        ang = rng.uniform(0, 2 * np.pi)
        self.grad = grad_mag * np.array([math.cos(ang), math.sin(ang)])
        self.centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        self.means = np.column_stack([rng.uniform(x0, x1, n_plumes), rng.uniform(y0, y1, n_plumes)])
        self.amps = rng.uniform(*amp, n_plumes)
        self.sigmas = rng.uniform(*sigma, n_plumes)
        self.drift = np.asarray(drift, float)

    def __call__(self, x, y, t=0.0):
        p = np.stack([np.asarray(x, float), np.asarray(y, float)], axis=-1)
        c = self.background + (p - self.centre) @ self.grad
        for m, a, s in zip(self.means, self.amps, self.sigmas):
            d2 = np.sum((p - (m + self.drift * t)) ** 2, axis=-1)
            c = c + a * np.exp(-d2 / (2 * s * s))
        return c


class ProbeSensor:
    """First-order lag + motion-dependent Gaussian noise (a generic in-situ probe)."""

    def __init__(self, tau_s=5.0, sigma0=0.1, k_r=1.0, k_dF=0.05, seed=0):
        self.tau = max(1e-3, tau_s)
        self.sigma0, self.k_r, self.k_dF = sigma0, k_r, k_dF
        self.c_m = None
        self.rng = np.random.default_rng(seed + 1000)

    def step(self, c_true, dt):
        if self.c_m is None:
            self.c_m = c_true
        else:
            a = 1.0 - math.exp(-dt / self.tau)          # exact discretisation of the lag
            self.c_m += a * (c_true - self.c_m)
        return self.c_m

    def read(self, yaw_rate, dF_dt, r_cap=1.0, dF_cap=2000.0):
        # caps guard against heading-wrap spikes and thrust steps being read as physical motion
        sigma = (self.sigma0 + self.k_r * min(abs(yaw_rate), r_cap)
                 + self.k_dF * min(abs(dF_dt), dF_cap) / 100.0)
        return self.c_m + self.rng.normal(0.0, sigma), sigma


class OnlineGP:
    """Exact GP regression with a squared-exponential kernel and fixed hyperparameters.

    Samples are binned into cells of `cell` metres (mean per cell) so the matrix stays small
    over long missions; the noise variance of a cell shrinks with its sample count.
    """

    def __init__(self, length=10.0, signal_std=None, noise_std=0.3, cell=1.5, max_cells=1500):
        self.length, self.signal_std, self.noise_std = length, signal_std, noise_std
        self.cell, self.max_cells = cell, max_cells
        self.bins = {}                                   # (i, j) -> [sum_x, sum_y, sum_z, n]

    def add(self, x, y, z):
        key = (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))
        b = self.bins.setdefault(key, [0.0, 0.0, 0.0, 0])
        b[0] += x; b[1] += y; b[2] += z; b[3] += 1

    def _data(self):
        B = np.array(list(self.bins.values()), float)
        if len(B) > self.max_cells:                      # keep the most-sampled cells
            B = B[np.argsort(-B[:, 3])[:self.max_cells]]
        n = B[:, 3]
        return B[:, :2] / n[:, None], B[:, 2] / n, n

    def _k(self, A, B):
        d2 = np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2 * A @ B.T
        return np.exp(-np.maximum(d2, 0) / (2 * self.length ** 2))

    def predict(self, P):
        if not self.bins:
            return np.full(len(P), np.nan), np.full(len(P), np.nan)
        X, z, n = self._data()
        mu0 = z.mean()
        s2 = (self.signal_std if self.signal_std else max(z.std(), 1e-3)) ** 2
        K = s2 * self._k(X, X) + np.diag(self.noise_std ** 2 / n) + 1e-8 * np.eye(len(X))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, z - mu0))
        Ks = s2 * self._k(P, X)
        mean = mu0 + Ks @ alpha
        v = np.linalg.solve(L, Ks.T)
        var = np.maximum(s2 - np.sum(v * v, 0), 0.0)
        return mean, np.sqrt(var)


def dist_to_polyline(P, wps):
    """Distance of each point in P (N,2) to the polyline through wps (M,2)."""
    d = np.full(len(P), np.inf)
    for a, b in zip(wps[:-1], wps[1:]):
        ab = b - a; l2 = ab @ ab
        t = np.clip(((P - a) @ ab) / l2, 0, 1) if l2 > 1e-9 else np.zeros(len(P))
        d = np.minimum(d, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
    return d


class MapScorer:
    """Scores the GP map on a regular grid against the true field."""

    def __init__(self, field, bounds, res=2.0, swath_wps=None, swath_width=8.0):
        (x0, x1), (y0, y1) = bounds
        gx, gy = np.arange(x0, x1 + 1e-9, res), np.arange(y0, y1 + 1e-9, res)
        self.GX, self.GY = np.meshgrid(gx, gy)
        self.P = np.column_stack([self.GX.ravel(), self.GY.ravel()])
        self.field = field
        self.swath = (dist_to_polyline(self.P, np.asarray(swath_wps, float)) <= swath_width
                      if swath_wps is not None and len(swath_wps) >= 2 else np.ones(len(self.P), bool))

    def score(self, gp, t=0.0, visit_radius=5.0):
        truth = self.field(self.P[:, 0], self.P[:, 1], t)
        mean, std = gp.predict(self.P)
        err = mean - truth
        s = self.swath
        # grid points within visit_radius of where the boat actually sampled (progress-independent score)
        visited = np.zeros(len(self.P), bool)
        if gp.bins:
            C, _, _ = gp._data()
            for i in range(0, len(C), 256):
                d2 = ((self.P[:, None, :] - C[None, i:i + 256, :]) ** 2).sum(-1)
                visited |= (d2 <= visit_radius ** 2).any(1)
        v = s & visited
        return dict(rmse_visited=float(np.sqrt(np.nanmean(err[v] ** 2))) if v.any() else float('nan'),
                    rmse_swath=float(np.sqrt(np.nanmean(err[s] ** 2))),
                    mae_swath=float(np.nanmean(np.abs(err[s]))),
                    rmse_area=float(np.sqrt(np.nanmean(err ** 2))),
                    mean_std_swath=float(np.nanmean(std[s])),
                    field_std_swath=float(np.std(truth[s]))), (mean, std, truth)


def mission_bounds(wps, margin=10.0):
    w = np.asarray(wps, float)
    return ((w[:, 0].min() - margin, w[:, 0].max() + margin),
            (w[:, 1].min() - margin, w[:, 1].max() + margin))


# =============================================================================
# ROS 2 NODE
# =============================================================================
def make_node_class():
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float64, String

    class WQSimNode(Node):
        def __init__(self):
            super().__init__('wq_sim_node')
            P = self.declare_parameter
            P('seed', 1)                       # plume layout; change per mission for statistics
            P('n_plumes', 3)
            P('plume_amp', [4.0, 10.0])        # units above background
            P('plume_sigma', [8.0, 20.0])      # m
            P('background', 5.0)
            P('gradient', 0.03)                # units per m
            P('drift', [0.0, 0.0])             # m/s, plume advection (0 = static field)
            P('tau_s', 5.0)                    # s, probe response time (check your datasheet)
            P('sigma0', 0.1)                   # units, noise at rest
            P('k_yaw', 1.0)                    # units per rad/s of yaw rate
            P('k_dthrust', 0.05)               # units per 100 N/s of thrust change
            P('sample_hz', 2.0)
            P('gp_length', 10.0)               # m
            P('gp_noise', 0.3)
            P('gp_cell', 1.5)                  # m, binning cell
            P('grid_res', 2.0)                 # m, scoring grid
            P('swath_width', 8.0)              # m each side of the planned path
            P('area_margin', 10.0)             # m around the waypoints
            P('eval_period', 30.0)             # s between map scores
            P('start_after_waypoint', 1)       # ignore the transit before this waypoint index
            g = lambda n: self.get_parameter(n).value
            self.g = g

            self.field = self.sensor = self.gp = self.scorer = None
            self.wps = None
            self.pos = None
            self.yaw = None; self.yaw_prev = None; self.t_yaw_prev = None; self.yaw_rate = 0.0
            self.thr = [0.0, 0.0]; self.F_prev = None; self.dF = 0.0
            self.energy = 0.0; self.dist = 0.0; self.pos_prev = None; self.anchor = None; self.speed = 0.0
            self.n_samples = 0; self.target_idx = 0; self.active = False
            self.t0 = None; self.t_last = None; self.t_last_eval = 0.0

            self.create_subscription(String, '/planning/waypoints', self.on_wps, 10)
            self.create_subscription(String, '/planning/current_target', self.on_target, 10)
            self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.on_imu, 10)
            self.create_subscription(Float64, '/wamv/thrusters/left/thrust', lambda m: self.on_thr(0, m), 10)
            self.create_subscription(Float64, '/wamv/thrusters/right/thrust', lambda m: self.on_thr(1, m), 10)
            self.pub_meas = self.create_publisher(String, '/wq/measurement', 10)
            self.pub_metrics = self.create_publisher(String, '/wq/metrics', 10)

            stamp = time.strftime('%Y%m%d_%H%M%S')
            self.dir = os.path.join(os.path.expanduser(os.environ.get('SAFE_RL_LOG_DIR', '~/seal_ws/logs')), 'wq'); os.makedirs(self.dir, exist_ok=True)
            self.tag = f"wq_seed{g('seed')}_tau{g('tau_s'):g}_{stamp}"
            self.f_s = open(os.path.join(self.dir, self.tag + '_samples.csv'), 'w', newline='')
            self.w_s = csv.writer(self.f_s)
            self.w_s.writerow(['t', 'x', 'y', 'c_true', 'c_lagged', 'y_meas', 'sigma', 'speed', 'yaw_rate', 'dF_dt', 'energy_J_proxy'])
            self.f_m = open(os.path.join(self.dir, self.tag + '_metrics.csv'), 'w', newline='')
            self.w_m = csv.writer(self.f_m)
            self.w_m.writerow(['t', 'n_samples', 'n_cells', 'rmse_visited', 'rmse_swath', 'mae_swath', 'rmse_area', 'mean_std_swath',
                               'field_std_swath', 'distance_m', 'energy_proxy', 'mean_speed'])
            self.create_timer(1.0 / float(g('sample_hz')), self.on_sample)
            self.get_logger().info(f"wq_sim_node: logging to {self.dir}/{self.tag}_*.csv (waiting for waypoints)")

        def now(self):
            # node clock: wall time by default, Gazebo time with --ros-args -p use_sim_time:=true
            return self.get_clock().now().nanoseconds * 1e-9

        # ---------- inputs ----------
        def on_wps(self, msg):
            try:
                wps = [(float(w[0]), float(w[1])) for w in json.loads(msg.data).get('waypoints', []) if len(w) >= 2]
            except (ValueError, TypeError, json.JSONDecodeError):
                return
            if len(wps) < 2 or self.field is not None:
                return
            g = self.g
            self.wps = np.asarray(wps)
            k = int(g('start_after_waypoint'))
            bounds = mission_bounds(self.wps[k:] if len(self.wps) - k >= 2 else self.wps, g('area_margin'))
            self.field = PlumeField(bounds, seed=int(g('seed')), n_plumes=int(g('n_plumes')), amp=tuple(g('plume_amp')),
                                    sigma=tuple(g('plume_sigma')), background=g('background'),
                                    grad_mag=g('gradient'), drift=tuple(g('drift')))
            self.sensor = ProbeSensor(g('tau_s'), g('sigma0'), g('k_yaw'), g('k_dthrust'), seed=int(g('seed')))
            self.gp = OnlineGP(length=g('gp_length'), noise_std=g('gp_noise'), cell=g('gp_cell'))
            self.scorer = MapScorer(self.field, bounds, g('grid_res'), self.wps[k:], g('swath_width'))
            np.savez(os.path.join(self.dir, self.tag + '_setup.npz'), wps=self.wps, bounds=np.array(bounds),
                     means=self.field.means, amps=self.field.amps, sigmas=self.field.sigmas,
                     grad=self.field.grad, centre=self.field.centre, background=self.field.background)
            self.get_logger().info("waypoints (index: x, y): " + ", ".join(
                f"{i}: ({w[0]:.0f}, {w[1]:.0f})" for i, w in enumerate(wps[:12])) + (" ..." if len(wps) > 12 else ""))
            self.get_logger().info(f"wq field ready: {len(wps)} waypoints, area x({bounds[0][0]:.0f}, {bounds[0][1]:.0f}) y({bounds[1][0]:.0f}, {bounds[1][1]:.0f}), "
                                   f"{int(g('n_plumes'))} plumes, seed {int(g('seed'))}, tau_s {g('tau_s')} s")

        def on_target(self, msg):
            try:
                d = json.loads(msg.data)
                self.pos = (float(d['current_position'][0]), float(d['current_position'][1]))
                tgt = d.get('target_waypoint')
                if self.wps is not None and tgt is not None:
                    self.target_idx = int(np.argmin(np.linalg.norm(self.wps - np.asarray(tgt, float), axis=1)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass

        def on_imu(self, msg):
            q = msg.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            now = self.now()
            if self.yaw_prev is not None and now - self.t_yaw_prev > 1e-3:   # heading-derived rate (gyro is unreliable in VRX)
                r = math.atan2(math.sin(yaw - self.yaw_prev), math.cos(yaw - self.yaw_prev)) / (now - self.t_yaw_prev)
                self.yaw_rate = 0.8 * self.yaw_rate + 0.2 * r
            self.yaw_prev, self.t_yaw_prev = yaw, now

        def on_thr(self, i, msg):
            self.thr[i] = float(msg.data)

        # ---------- sampling loop ----------
        def on_sample(self):
            if self.field is None or self.pos is None:
                return
            now = self.now()
            if self.t0 is None:
                self.t0 = now
            t = now - self.t0
            dt = 1.0 / float(self.g('sample_hz')) if self.t_last is None else now - self.t_last
            self.t_last = now
            if not self.active:
                self.active = self.target_idx >= int(self.g('start_after_waypoint'))
                if not self.active:
                    return
                self.get_logger().info("wq sampling started (survey reached)")
            x, y = self.pos
            F = abs(self.thr[0]) + abs(self.thr[1])
            if self.F_prev is not None and dt > 0:
                self.dF = 0.5 * self.dF + 0.5 * (F - self.F_prev) / dt
            self.F_prev = F
            self.energy += (self.thr[0] ** 2 + self.thr[1] ** 2) * dt / 1e3     # thrust^2 proxy (kN^2 s)
            # distance by chords of >= 1 m, so position jitter is not summed up as travel
            if self.anchor is None:
                self.anchor = (x, y, t)
            else:
                chord = math.hypot(x - self.anchor[0], y - self.anchor[1])
                if chord >= 1.0:
                    if chord < 10.0:                       # ignore position jumps
                        self.dist += chord
                        self.speed = chord / max(t - self.anchor[2], 1e-3)
                    self.anchor = (x, y, t)
            speed = self.speed
            self.pos_prev = (x, y)
            c = float(self.field(x, y, t))
            c_lag = self.sensor.step(c, dt)
            z, sigma = self.sensor.read(self.yaw_rate, self.dF)
            self.gp.add(x, y, z); self.n_samples += 1
            self.w_s.writerow([f'{t:.2f}', f'{x:.2f}', f'{y:.2f}', f'{c:.4f}', f'{c_lag:.4f}', f'{z:.4f}',
                               f'{sigma:.4f}', f'{speed:.3f}', f'{self.yaw_rate:.4f}', f'{self.dF:.2f}', f'{self.energy:.2f}'])
            # err = measurement - true value at the boat: the per-step training signal for task-aware RL
            self.pub_meas.publish(String(data=json.dumps(dict(t=round(t, 2), x=round(x, 2), y=round(y, 2),
                                                               value=round(z, 4), sigma=round(sigma, 4),
                                                               err=round(z - c, 4)))))
            if t - self.t_last_eval >= float(self.g('eval_period')):
                self.t_last_eval = t
                self.evaluate(t)

        def evaluate(self, t, final=False):
            m, maps = self.scorer.score(self.gp, t)
            row = dict(t=round(t, 1), n_samples=self.n_samples, n_cells=len(self.gp.bins), **{k: round(v, 4) for k, v in m.items()},
                       distance_m=round(self.dist, 1), energy_proxy=round(self.energy, 1),
                       mean_speed=round(self.dist / max(t, 1e-3), 3))
            self.w_m.writerow([row[k] for k in ['t', 'n_samples', 'n_cells', 'rmse_visited', 'rmse_swath', 'mae_swath', 'rmse_area',
                                                'mean_std_swath', 'field_std_swath', 'distance_m', 'energy_proxy', 'mean_speed']])
            self.f_m.flush(); self.f_s.flush()
            self.pub_metrics.publish(String(data=json.dumps(row)))
            self.get_logger().info(f"wq map: t={t:.0f}s n={self.n_samples} RMSE(visited)={m['rmse_visited']:.3f} "
                                   f"RMSE(whole path)={m['rmse_swath']:.3f} dist={self.dist:.0f} m "
                                   f"speed={self.dist / max(t, 1e-3):.2f} m/s energy={self.energy:.0f}")
            if final:
                self.save_figure(maps)

        def save_figure(self, maps):
            try:
                import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            except ImportError:
                return
            mean, std, truth = maps; S = self.scorer.GX.shape
            fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
            vmin, vmax = np.nanmin(truth), np.nanmax(truth)
            for a, data, title in zip(ax, [truth, mean, std], ['true field', 'GP map', 'GP std']):
                im = a.pcolormesh(self.scorer.GX, self.scorer.GY, data.reshape(S), shading='auto',
                                  vmin=None if title == 'GP std' else vmin, vmax=None if title == 'GP std' else vmax)
                a.plot(self.wps[:, 0], self.wps[:, 1], 'w--', lw=.6); a.set_title(title); a.set_aspect('equal')
                a.set_xlim(self.scorer.GX.min(), self.scorer.GX.max()); a.set_ylim(self.scorer.GY.min(), self.scorer.GY.max())
                fig.colorbar(im, ax=a, shrink=.8)
            fig.tight_layout(); fig.savefig(os.path.join(self.dir, self.tag + '_map.png'), dpi=150); plt.close(fig)

        def shutdown(self):
            if self.gp is not None and self.gp.bins:
                self.evaluate(self.now() - (self.t0 or self.now()), final=True)
            self.f_s.close(); self.f_m.close()

    return WQSimNode


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = make_node_class()()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
