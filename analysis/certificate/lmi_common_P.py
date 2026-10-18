"""Common quadratic Lyapunov function for the heading loop over the whole certified set C_theta.

Model (Nomoto, constant reference):  e_ddot = -w*e_dot - K*w*TS*s*u,   w = 1/T,
u = k_p(e) e + k_d(e_dot) e_dot + phi(sigma) (e_dot + lam e) + k_i(xi) xi,   s in [s_min, 1] the saturation factor.
For theta in C_theta and (e, e_dot) in the box |e|<=e_max, |e_dot|<=ed_max every coefficient lies in an interval:
  k_p in [kp_lo, kp_hi],  k_d in [kd_lo, kd_hi],  phi in [0, 0.7 rho_max/Phi_min],  lam in [lam_lo, lam_hi].
A is multi-affine in (K, w, s, k_p, k_d, phi, lam), so A stays in the convex hull of the vertex matrices; a P with
A_v' P + P A_v + alpha P <= 0 at all vertices gives  V_dot <= -alpha V  for every theta(t) in C_theta (any switching).
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import sys, json, itertools, math, numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
TS = 200.0
B = dict(mu=(0.1, 0.9), eps=(0.01, 0.5), lam=(0.01, 0.5), Phi=(0.05, 2.0), rho=(0.0, 0.2))
KP0, KD0, KP_CAP, KD_CAP = 4.0, 2.5, (0.05, 20.0), (0.1, 10.0)
def gain_range(base, xmax, cap):
    """range of clamp(base*max(|x|,eps)^(mu-1)) for |x|<=xmax, mu, eps in their boxes"""
    ms = [B['eps'][0], max(xmax, B['eps'][1])]                 # m = max(|x|, eps) ranges over [eps_lo, max(xmax, eps_hi)]
    vals = [base * m ** (mu - 1.0) for m in ms for mu in B['mu']]   # monotone in m and in mu -> extremes at corners
    return max(cap[0], min(vals)), min(cap[1], max(vals))
def vertices(e_max, ed_max, Kr, Tr, s_min):
    kp = gain_range(KP0, e_max, KP_CAP); kd = gain_range(KD0, ed_max, KD_CAP); ph = (0.0, 0.7 * B['rho'][1] / B['Phi'][0])
    V = []
    for K, w, s, a, b, f, l in itertools.product(Kr, (1 / Tr[1], 1 / Tr[0]), (s_min, 1.0), kp, kd, ph, B['lam']):
        V.append(np.array([[0.0, 1.0], [-K * w * TS * s * (a + f * l), -w - K * w * TS * s * (b + f)]]))
    return V, kp, kd, ph
def alpha_of(P, V):
    A = np.asarray(V); w, U = np.linalg.eigh(P)
    if w[0] <= 1e-9: return -1e9
    S = U @ np.diag(w ** -0.5) @ U.T                      # P^(-1/2)
    Q = -(np.swapaxes(A, 1, 2) @ P + P @ A)
    return float(np.linalg.eigvalsh(S @ Q @ S)[:, 0].min())
def best_P(V):
    def neg(z):
        p = math.exp(z[0]); q = math.tanh(z[1]) * math.sqrt(p) * 0.999
        return -alpha_of(np.array([[1.0, q], [q, p]]), V)
    best = None
    for z0 in itertools.product(np.linspace(-4, 1, 6), np.linspace(-0.5, 1.5, 5)):
        r = minimize(neg, z0, method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000))
        if best is None or r.fun < best.fun: best = r
    p = math.exp(best.x[0]); q = math.tanh(best.x[1]) * math.sqrt(p) * 0.999
    return -best.fun, np.array([[1.0, q], [q, p]])
