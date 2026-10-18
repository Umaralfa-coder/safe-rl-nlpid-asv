"""1) vertex certificate for the rounded P printed in the paper; 2) sanity check on the real nonlinear law."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import sys, math, numpy as np
sys.path.insert(0, _os.path.join(_HERE, '..', '..', 'tests', 'stubs'))
sys.path.insert(0, _os.path.join(_HERE, '..', '..', 'ros2_ws', 'src', 'safe_rl_nlpid', 'safe_rl_nlpid'))
from lmi_common_P import *
import buran_controller as bc
K, T, e_max, ed_max = 1.5e-3, 0.68, math.radians(30), 1.0
V, kp, kd, ph = vertices(e_max, ed_max, (K, K), (T, T), 1.0); V = list({A.tobytes(): A for A in V}.values())
for P in (np.array([[1, 0.20], [0.20, 0.22]]), np.array([[1, 0.2], [0.2, 0.2]])):
    print('P', P.tolist(), 'alpha at the', len(V), 'vertices:', round(alpha_of(P, V), 4), '| eig(P)', np.linalg.eigvalsh(P).round(3))
P = np.array([[1, 0.20], [0.20, 0.22]])
ctrl = bc.HeadingNLPID(4.0, 2.5, 0.08, {k: (lo + hi) / 2 for k, (lo, hi) in bc.PARAM_BOUNDS.items()},
                       gain_limits={'K_p': (0.05, 20.0), 'K_d': (0.1, 10.0), 'K_i': (0.001, 0.5)})
rng = np.random.default_rng(3); worst = np.inf; n = 0; kmin = [np.inf, np.inf]; kmax = [0, 0]
for _ in range(200000):
    v = bc.PARAM_LO + (bc.PARAM_HI - bc.PARAM_LO) * (rng.random(9) if rng.random() < 0.7 else rng.integers(0, 2, 9))
    th = bc.vec_to_theta(v); e = rng.uniform(-e_max, e_max); ed = rng.uniform(-ed_max, ed_max)
    if rng.random() < 0.3: e *= 10 ** rng.uniform(-3, 0); ed *= 10 ** rng.uniform(-3, 0)
    u, Kp, Kd, Ki, sig = ctrl.law(th, e, ed, 0.0)
    x = np.array([e, ed]); dx = np.array([ed, -(K * 200.0 * u + ed) / T])       # unsaturated
    Vv = 0.5 * x @ P @ x
    if Vv < 1e-14: continue
    worst = min(worst, -(x @ P @ dx) / Vv); n += 1
    kmin = [min(kmin[0], Kp), min(kmin[1], Kd)]; kmax = [max(kmax[0], Kp), max(kmax[1], Kd)]
print('nonlinear law, %d random (theta, state) samples: min of -Vdot/V = %.4f' % (n, worst), '| k_p seen', np.round([kmin[0], kmax[0]], 2), 'k_d seen', np.round([kmin[1], kmax[1]], 2))
# integral term bound
m = 0
for mu in np.linspace(.1, .9, 9):
    for eps in np.linspace(.01, .5, 8):
        for xi in np.linspace(0, 1, 2001):
            m = max(m, bc.nl_gain(0.08, xi, mu, eps, 0.001, 0.5) * xi)
print('max |K_i(xi) xi| for |xi| <= 1:', round(m, 4))
# largest |e_dot|, |u| where the loop is unsaturated inside the box
print('torque limit 610 N -> |u| <= %.2f ; at theta_nom small-signal k_p=13.7: e up to %.2f deg unsaturated' % (610 / 200, math.degrees(610 / 200 / 13.73)))
