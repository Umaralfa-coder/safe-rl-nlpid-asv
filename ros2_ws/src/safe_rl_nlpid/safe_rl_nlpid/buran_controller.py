#!/usr/bin/env python3
"""
Buran Controller - Safe-RL-NLPID (paper-aligned implementation)
================================================================
Implements "Safety-Shielded Reinforcement Learning-Based Robust Control for
Autonomous Surface Vehicles":

  * NLPID heading law with saturation-type nonlinear gains (Guerrero et al.),
    parameterised by
        theta = [mu_p, mu_d, mu_i, eps_p, eps_d, eps_i, lambda, Phi, rho].
  * Outer RL policy pi_psi(a | s) mapping the state
        s = [e, e_dot, sigma, y_e, d_hat_lat, u - u_d]
    to gain adjustments (linear-tanh policy trained with antithetic evolution
    strategies). Changes of theta are rate-limited and held for `rl_hold_steps`
    control steps (bounded switching, needed for the certificate to be
    meaningful under time-varying gains).
  * Parameter-space, minimal-intervention safety shield (paper Sec. III):
    the candidate theta is projected onto C_theta and then pulled back toward
    the certified nominal theta (line search) until the discrete barrier
    condition
        phi_k <= (1 - eta*dt)^k * phi_0 + eps*k*dt,   k = 1..H
    holds on a short-horizon prediction (online IV-identified Nomoto yaw model
    + actuator model with thrust clipping and slew). If even the nominal
    controller cannot meet it (recovery mode, e.g. after a reference jump) the
    candidate must do no worse than nominal. Worst case = certified nominal
    NLPID (Theorem 1 fallback).
    The shield never attenuates the control output itself.
  * Disturbance observer (Kalman filter on lateral drift) with a d_tilde proxy
    gate: learning is frozen and theta := theta_nom while the innovation
    proxy exceeds d_safe (Theorem 1 precondition).
  * Runtime Lyapunov monitor: V = 0.5*sigma^2 + 0.5*k_e*e^2, checks
    V_dot <= -alpha0*V + beta0*|d_hat| and logs violations.
  * ILOS path-following guidance (signed cross-track error) replacing the
    unsigned, saturated additive CTE correction.
  * Heading derivative on measurement (filtered derivative of the measured
    heading, or IMU yaw rate if clean): no derivative kick at waypoint switches.
  * Surge-preserving thrust allocation below `surge_priority_deg` of error.
  * Velocity loop evaluated exactly once per step. Its 9 NLPID parameters
    theta_v are adapted by the same policy (action = [theta_h, theta_v]) and
    filtered by a velocity shield (surge-model prediction, barrier on phi_v).
  * Speed-scheduled surge feed-forward base_speed * (u_d_eff / u_d)^2
    (`ff_speed_scaling`), so the boat can slow down when the planner lowers
    the desired speed; velocity K_p cap raised to 400.
  * Ablation switches: rl_mode in {off, random, es, eval}, shield_enabled.

Sign conventions (ENU): yaw positive counter-clockwise; heading error
e = psi_d - psi; e_dot = -r for a (locally) constant reference;
tau_r > 0 -> right thruster > left thruster -> counter-clockwise (left) turn.
"""

import rclpy
from rclpy.node import Node
import math
import json
import numpy as np
import csv
import os
from datetime import datetime

# Log and policy directory (override with the SAFE_RL_LOG_DIR environment variable).
LOG_DIR = os.path.expanduser(os.environ.get('SAFE_RL_LOG_DIR', '~/seal_ws/logs'))

from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64, String


# =============================================================================
# CONSTANTS
# =============================================================================
MAX_THRUST = 2000.0          # N - hardware limit
SAFE_THRUST = 800.0          # N - operational limit per thruster
TURN_POWER_LIMIT = 1600.0    # N - max differential thrust (yaw torque proxy)
MAX_GPS_SPEED = 10.0         # m/s - GPS glitch rejection only (not a safety clip)

# Ordered adaptable parameter vector theta
PARAM_KEYS = ['μ_p', 'μ_d', 'μ_i', 'ε_p', 'ε_d', 'ε_i', 'λ', 'boundary_layer', 'robust_gain']
PARAM_ASCII = ['mu_p', 'mu_d', 'mu_i', 'eps_p', 'eps_d', 'eps_i',
               'lambda', 'boundary_layer', 'robust_gain']

# Certified parameter region C_theta (paper eq. 3, extended with lambda, Phi, rho)
PARAM_BOUNDS = {
    'μ_p': (0.1, 0.9), 'μ_d': (0.1, 0.9), 'μ_i': (0.1, 0.9),
    'ε_p': (0.01, 0.5), 'ε_d': (0.01, 0.5), 'ε_i': (0.01, 0.5),
    'λ': (0.01, 0.5), 'boundary_layer': (0.05, 2.0), 'robust_gain': (0.0, 0.2),
}
PARAM_LO = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS])
PARAM_HI = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS])

# Certified region for the velocity (surge) NLPID parameters theta_v (same keys)
PARAM_BOUNDS_V = {
    'μ_p': (0.1, 0.9), 'μ_d': (0.1, 0.9), 'μ_i': (0.1, 0.9),
    'ε_p': (0.01, 0.5), 'ε_d': (0.01, 0.5), 'ε_i': (0.01, 0.5),
    'λ': (0.01, 0.2), 'boundary_layer': (0.5, 5.0), 'robust_gain': (0.0, 1.0),
}
PARAM_LO_V = np.array([PARAM_BOUNDS_V[k][0] for k in PARAM_KEYS])
PARAM_HI_V = np.array([PARAM_BOUNDS_V[k][1] for k in PARAM_KEYS])
PARAM_ASCII_V = ['v_' + k for k in PARAM_ASCII]

# Enlarged proposal sets for the unconstrained-learner experiments (BU / CU): the region an
# unshielded learner may propose from. They contain C_theta and are only bounded where the
# control law would otherwise be undefined (non-positive exponents, widths or boundary layers).
PARAM_BOUNDS_WIDE = {
    'μ_p': (0.02, 1.5), 'μ_d': (0.02, 1.5), 'μ_i': (0.02, 1.5),
    'ε_p': (0.001, 1.0), 'ε_d': (0.001, 1.0), 'ε_i': (0.001, 1.0),
    'λ': (0.001, 2.0), 'boundary_layer': (0.01, 4.0), 'robust_gain': (0.0, 1.0),
}
PARAM_BOUNDS_WIDE_V = {
    'μ_p': (0.02, 1.5), 'μ_d': (0.02, 1.5), 'μ_i': (0.02, 1.5),
    'ε_p': (0.001, 1.0), 'ε_d': (0.001, 1.0), 'ε_i': (0.001, 1.0),
    'λ': (0.001, 1.0), 'boundary_layer': (0.05, 10.0), 'robust_gain': (0.0, 5.0),
}
PARAM_WIDE_LO = np.array([PARAM_BOUNDS_WIDE[k][0] for k in PARAM_KEYS])
PARAM_WIDE_HI = np.array([PARAM_BOUNDS_WIDE[k][1] for k in PARAM_KEYS])
PARAM_WIDE_LO_V = np.array([PARAM_BOUNDS_WIDE_V[k][0] for k in PARAM_KEYS])
PARAM_WIDE_HI_V = np.array([PARAM_BOUNDS_WIDE_V[k][1] for k in PARAM_KEYS])


# =============================================================================
# HELPERS
# =============================================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def theta_to_vec(theta):
    return np.array([theta[k] for k in PARAM_KEYS], dtype=float)


def vec_to_theta(v):
    return {k: float(v[i]) for i, k in enumerate(PARAM_KEYS)}


def project_to_C(v):
    """Euclidean projection onto the heading box C_theta."""
    return np.clip(np.asarray(v, dtype=float), PARAM_LO, PARAM_HI)


def project_to_C_v(v):
    """Euclidean projection onto the velocity box C_theta_v."""
    return np.clip(np.asarray(v, dtype=float), PARAM_LO_V, PARAM_HI_V)


def nl_gain(base, x, mu, eps, lo, hi):
    """Saturation-type nonlinear gain: base*|x|^(mu-1) if |x|>eps else base*eps^(mu-1)."""
    ax = abs(x)
    g = base * (ax ** (mu - 1.0)) if ax > eps else base * (eps ** (mu - 1.0))
    return clamp(g, lo, hi)


# =============================================================================
# NLPID HEADING CONTROLLER
# =============================================================================
class HeadingNLPID:
    """Nonlinear robust PID for heading.

    u = K_p(e) e + K_d(e_dot) e_dot + K_i(int e) int e + rho * tanh(0.7 sigma / Phi)
    sigma = e_dot + lambda e
    `law()` is a pure function of (theta, state) so the shield can preview it.
    """

    def __init__(self, K_p, K_d, K_i, theta, gain_limits,
                 integral_limit=1.0, integral_zone=0.35):
        self.K_p_base = K_p
        self.K_d_base = K_d
        self.K_i_base = K_i
        self.gain_limits = gain_limits          # {'K_p': (lo, hi), 'K_d': ..., 'K_i': ...}
        self.integral_limit = integral_limit    # rad*s
        self.integral_zone = integral_zone      # rad; integrate only near the reference
        self.bounds = PARAM_BOUNDS              # certified box; replaced only by uncertify()
        self.theta = {k: clamp(theta[k], *PARAM_BOUNDS[k]) for k in PARAM_KEYS}
        self.integral = 0.0
        self.sigma = 0.0
        self.current_K_p = K_p
        self.current_K_d = K_d
        self.current_K_i = K_i

    def set_theta(self, theta):
        self.theta = {k: clamp(theta[k], *self.bounds[k]) for k in PARAM_KEYS}

    def uncertify(self, bounds, gain_cap_scale):
        """Unshielded, unconstrained learner (experiment BU): accept theta from the enlarged
        set and lift the certified gain caps by gain_cap_scale. Never used with the shield."""
        self.bounds = bounds
        self.gain_limits = {k: (lo, hi * gain_cap_scale) for k, (lo, hi) in self.gain_limits.items()}

    def reset(self):
        self.integral = 0.0
        self.sigma = 0.0

    def integrate(self, integral, e, dt):
        if abs(e) < self.integral_zone:
            integral = clamp(integral + e * dt, -self.integral_limit, self.integral_limit)
        return integral

    def law(self, theta, e, e_dot, integral):
        Kp = nl_gain(self.K_p_base, e, theta['μ_p'], theta['ε_p'], *self.gain_limits['K_p'])
        Kd = nl_gain(self.K_d_base, e_dot, theta['μ_d'], theta['ε_d'], *self.gain_limits['K_d'])
        Ki = nl_gain(self.K_i_base, integral, theta['μ_i'], theta['ε_i'], *self.gain_limits['K_i'])
        sigma = e_dot + theta['λ'] * e
        robust = theta['robust_gain'] * math.tanh(0.7 * sigma / theta['boundary_layer'])
        u = Kp * e + Kd * e_dot + Ki * integral + robust
        return u, Kp, Kd, Ki, sigma

    def compute_control(self, e, e_dot, dt):
        self.integral = self.integrate(self.integral, e, dt)
        u, Kp, Kd, Ki, sigma = self.law(self.theta, e, e_dot, self.integral)
        self.current_K_p, self.current_K_d, self.current_K_i = Kp, Kd, Ki
        self.sigma = sigma
        return u

    def get_gains(self):
        return self.current_K_p, self.current_K_d, self.current_K_i


# =============================================================================
# NLPID VELOCITY CONTROLLER (fixed parameters, called once per step)
# =============================================================================
class VelocityNLPID:
    """Nonlinear robust PID for surge speed, with RL-adaptable parameters theta_v.

    u = K_p(e_u) e_u + K_d(d) d + K_i(int e_u) int e_u + rho * K_i * tanh(0.8 sigma / Phi)
    d = low-pass filtered d(e_u)/dt,  sigma = d + lambda * e_u
    `law()` is a pure function of (theta_v, state, e_u) so the shield can preview it.
    """

    def __init__(self, K_p, K_d, K_i, theta, gain_limits, integral_limit=200.0,
                 derivative_filter=0.8):
        self.K_p_base, self.K_d_base, self.K_i_base = K_p, K_d, K_i
        self.gain_limits = gain_limits          # {'K_p': (lo, hi), 'K_d': ..., 'K_i': ...}
        self.integral_limit = integral_limit
        self.derivative_filter = derivative_filter
        self.bounds = PARAM_BOUNDS_V            # certified box; replaced only by uncertify()
        self.theta = {k: clamp(theta[k], *PARAM_BOUNDS_V[k]) for k in PARAM_KEYS}
        self.reset()
        self.current_K_p, self.current_K_d, self.current_K_i = K_p, K_d, K_i

    def set_theta(self, theta):
        self.theta = {k: clamp(theta[k], *self.bounds[k]) for k in PARAM_KEYS}

    def uncertify(self, bounds, gain_cap_scale):
        """See HeadingNLPID.uncertify."""
        self.bounds = bounds
        self.gain_limits = {k: (lo, hi * gain_cap_scale) for k, (lo, hi) in self.gain_limits.items()}

    def reset(self):
        self.state = (0.0, 0.0, 0.0)            # (prev_error, filtered derivative, integral)
        self.sliding_surface = 0.0

    def law(self, theta, state, speed_error, dt):
        """Returns (control, new_state, (Kp, Kd, Ki), sigma)."""
        prev_e, prev_d, integ = state
        raw = (speed_error - prev_e) / dt
        a = self.derivative_filter
        d = a * prev_d + (1.0 - a) * raw
        integ = clamp(integ + speed_error * dt, -self.integral_limit, self.integral_limit)
        sigma = d + theta['λ'] * speed_error
        Kp = nl_gain(self.K_p_base, speed_error, theta['μ_p'], theta['ε_p'], *self.gain_limits['K_p'])
        Kd = nl_gain(self.K_d_base, d, theta['μ_d'], theta['ε_d'], *self.gain_limits['K_d'])
        Ki = nl_gain(self.K_i_base, integ, theta['μ_i'], theta['ε_i'], *self.gain_limits['K_i'])
        robust = theta['robust_gain'] * Ki * math.tanh(0.8 * sigma / theta['boundary_layer'])
        u = Kp * speed_error + Kd * d + Ki * integ + robust
        return u, (speed_error, d, integ), (Kp, Kd, Ki), sigma

    def compute_control(self, speed_error, dt):
        if dt < 1e-3:
            dt = 0.05
        u, self.state, gains, sigma = self.law(self.theta, self.state, speed_error, dt)
        self.current_K_p, self.current_K_d, self.current_K_i = gains
        self.sliding_surface = sigma
        return u

    def get_gains(self):
        return self.current_K_p, self.current_K_d, self.current_K_i


# =============================================================================
# SAFETY INDEX (paper Def. 1 / eq. 8)
# =============================================================================
class SafetyIndex:
    """phi = max(phi_h, phi_v), with (paper Def. 1 / eq. 8, plus a speed-tracking term)
        phi_h = |e_psi| / e_max - 1 + delta
        phi_v = max(|u| / u_max, u_d / u_d_max, |u - u_d| / e_u_max) - 1 + delta.
    Safe iff phi <= 0, e.g. |e_psi| <= (1 - delta) * e_max = 27 deg for the defaults.
    Because phi is a max, enforcing the discrete barrier condition separately on
    phi_h (heading shield) and phi_v (velocity shield) enforces it on phi:
    phi_i,k <= c^k phi_i,0 + eps k dt <= c^k phi_0 + eps k dt for every channel i."""

    def __init__(self, e_max_deg=30.0, u_max=5.0, ud_max=2.0, delta=0.1, eu_max=1.0):
        self.e_max = math.radians(e_max_deg)
        self.u_max = u_max
        self.ud_max = ud_max
        self.eu_max = eu_max
        self.delta = delta

    def heading(self, e_rad):
        return abs(e_rad) / self.e_max - 1.0 + self.delta

    def speed(self, u, ud):
        return max(abs(u) / self.u_max, ud / self.ud_max, abs(u - ud) / self.eu_max) - 1.0 + self.delta

    def __call__(self, e_rad, u, ud):
        return max(self.heading(e_rad), self.speed(u, ud))


# =============================================================================
# SURGE MODEL (used by the velocity shield's predictor)
# =============================================================================
class SurgeModel:
    """u_dot = c1 * F - c2 * u|u|,  F = left + right thrust [N].
    Defaults identified from the VRX log of 23/09/2026 (least squares on 1-s
    differences): c1 = 1.07e-4, c2 = 0.161 -> ~0.72 m/s at F = 850 N."""

    def __init__(self, c1=1.07e-4, c2=0.161):
        self.c1, self.c2 = c1, c2

    def step(self, u, F, dt):
        return u + dt * (self.c1 * F - self.c2 * u * abs(u))


# =============================================================================
# ONLINE NOMOTO MODEL IDENTIFICATION (RLS) - used by the shield's predictor
# =============================================================================
class NomotoRLS:
    """First-order Nomoto model with yaw-disturbance bias,
        T*r_dot + r = K*tau_r + d_r,
    identified online in the discrete form
        r[k+1] = a*r[k] + b*(tau[k]/1000) + c,     d_r = c / (1 - a).

    Instrumental-variable RLS with exponential forgetting. Instruments: the
    noise-free simulated yaw rate of the current model, the previous torque and
    a constant; all are uncorrelated with the current IMU noise even in closed
    loop (where tau[k] depends on the noisy r[k]). Plain RLS is biased by IMU
    noise (errors-in-variables) and, without the bias term c, by slowly varying
    yaw disturbances that the controller counteracts."""

    TAU_SCALE = 1000.0

    def __init__(self, dt, K_default, T_default, forgetting=0.999, min_updates=400):
        self.dt = dt
        self.K_default = K_default
        self.T_default = T_default
        self.lam = forgetting
        self.min_updates = min_updates
        a0 = 1.0 - dt / T_default
        b0 = K_default * self.TAU_SCALE * dt / T_default
        self.w = np.array([a0, b0, 0.0])
        self.P = np.diag([1e-2, 1e-2, 1e-4])
        self.n_updates = 0
        self.r_sim = 0.0
        self.u_prev = 0.0

    def update(self, r_prev, tau_prev, r_now):
        u = tau_prev / self.TAU_SCALE
        x = np.array([r_prev, u, 1.0])                 # regressor (noisy, closed loop)
        z = np.array([self.r_sim, self.u_prev, 1.0])   # instruments
        excited = abs(u) >= 0.05 or abs(self.u_prev) >= 0.05
        self.r_sim = clamp(float(self.w @ np.array([self.r_sim, u, 1.0])), -1.0, 1.0)
        self.u_prev = u
        if not excited:
            return  # gate on torque only: gating on the noisy r would correlate
                    # sample selection with the measurement noise
        Pz = self.P @ z
        den = self.lam + x @ Pz
        if abs(den) < 1e-9:
            return
        k = Pz / den
        self.w = self.w + k * (r_now - self.w @ x)
        self.P = (self.P - np.outer(k, x @ self.P)) / self.lam
        tr = np.trace(self.P)
        if tr > 10.0:
            self.P *= 10.0 / tr
        self.n_updates += 1

    def is_valid(self):
        a, b, _ = self.w
        return self.n_updates >= self.min_updates and 0.8 < a < 0.9999 and b > 0.0

    def model(self):
        """Return (K [rad/s per N], T [s], identified_flag)."""
        if not self.is_valid():
            return self.K_default, self.T_default, False
        a, b, _ = self.w
        T = clamp(self.dt / (1.0 - a), 0.3, 20.0)
        K = clamp((b / self.TAU_SCALE) / (1.0 - a), 1e-6, 1e-2)
        return K, T, True

    def yaw_disturbance(self):
        """Steady-state yaw-rate disturbance d_r [rad/s] (0 until identified)."""
        if not self.is_valid():
            return 0.0
        a, _, c = self.w
        return float(c / (1.0 - a))


# =============================================================================
# PARAMETER-SPACE SAFETY SHIELD (paper eq. 6, practical realisation)
# =============================================================================
class ParameterSpaceShield:
    """Minimal-intervention shield acting on theta, not on the control output.

    Candidates along the segment theta(alpha) = theta_nom + alpha*(theta_cand - theta_nom),
    alpha = 1, 0.75, ..., 0, are checked on an H-step prediction (identified
    Nomoto yaw model + actuator model with thrust clipping and slew limit):

      * Barrier mode (nominal prediction satisfies the discrete barrier
        phi_k <= (1 - eta*dt)^k * phi_0 + eps*k*dt): accept the largest alpha
        whose prediction also satisfies it.
      * Recovery mode (even the certified nominal controller cannot satisfy the
        barrier, e.g. right after a reference jump): accept the largest alpha
        whose prediction is no worse than nominal, i.e. worst barrier excess and
        terminal phi within `tol` of the nominal prediction.

    If no alpha > 0 qualifies, theta_nom is returned (certified fallback of
    Theorem 1). `intervened` is True iff alpha < 1.
    """

    def __init__(self, safety_index, eta=0.5, eps=0.1, horizon_steps=20,
                 alphas=(1.0, 0.75, 0.5, 0.25, 0.0), tol=1e-3):
        self.phi = safety_index
        self.eta = eta
        self.eps = eps
        self.H = horizon_steps
        self.alphas = alphas
        self.tol = tol
        self.total_interventions = 0
        self.total_recovery_mode = 0

    def _rollout(self, theta, ctrl, actuator_factory, e0, r0, u, ud, K, T, dt, phi0):
        """Returns (max barrier excess over the horizon, terminal phi)."""
        act = actuator_factory()
        e_k, r_k, integ = e0, r0, ctrl.integral
        decay = 1.0 - self.eta * dt
        worst = -float('inf')
        phi_k = phi0
        for k in range(1, self.H + 1):
            integ = ctrl.integrate(integ, e_k, dt)
            steer, _, _, _, _ = ctrl.law(theta, e_k, -r_k, integ)
            tau = act(steer, e_k)
            r_k = r_k + dt * (K * tau - r_k) / T
            e_k = wrap_angle(e_k - dt * r_k)
            phi_k = self.phi.heading(e_k)
            worst = max(worst, phi_k - ((decay ** k) * phi0 + self.eps * k * dt))
        return worst, phi_k

    def filter(self, theta_cand_vec, theta_nom_vec, ctrl, actuator_factory,
               e, r, u, ud, K, T, dt):
        """Returns (theta_vec, alpha, intervened, recovery_mode, phi0_heading)."""
        phi0 = self.phi.heading(e)
        cand = project_to_C(theta_cand_vec)
        nom = np.asarray(theta_nom_vec, dtype=float)
        if np.allclose(cand, nom):
            return nom.copy(), 1.0, False, False, phi0
        args = (ctrl, actuator_factory, e, r, u, ud, K, T, dt, phi0)
        nom_excess, nom_term = self._rollout(vec_to_theta(nom), *args)
        recovery = nom_excess > 0.0
        if recovery:
            self.total_recovery_mode += 1
        for a in self.alphas[:-1]:
            th_vec = nom + a * (cand - nom)
            excess, term = self._rollout(vec_to_theta(th_vec), *args)
            ok = (excess <= 0.0) if not recovery else \
                 (excess <= nom_excess + self.tol and term <= nom_term + self.tol)
            if ok:
                if a < 1.0:
                    self.total_interventions += 1
                return th_vec, a, a < 1.0, recovery, phi0
        self.total_interventions += 1
        return nom.copy(), 0.0, True, recovery, phi0


class VelocityShield:
    """Same minimal-intervention line search as ParameterSpaceShield, on theta_v.
    Prediction: velocity NLPID law -> tau_u (clipped) -> F = 2 tau_u with the
    per-thruster slew limit -> SurgeModel, over H steps, checked against the
    discrete barrier condition on phi_v."""

    def __init__(self, safety_index, surge_model, eta=0.5, eps=0.1, horizon_steps=40,
                 alphas=(1.0, 0.75, 0.5, 0.25, 0.0), tol=1e-3):
        self.phi = safety_index
        self.model = surge_model
        self.eta, self.eps, self.H = eta, eps, horizon_steps
        self.alphas, self.tol = alphas, tol
        self.total_interventions = 0
        self.total_recovery_mode = 0

    def _rollout(self, theta, ctrl, u0, ud, F0, tau_u_fn, slew, dt, phi0):
        u, F, state = u0, F0, ctrl.state
        decay = 1.0 - self.eta * dt
        worst, phi_k = -float('inf'), phi0
        for k in range(1, self.H + 1):
            cmd, state, _, _ = ctrl.law(theta, state, ud - u, dt)
            F = F + clamp(2.0 * tau_u_fn(cmd) - F, -2.0 * slew, 2.0 * slew)
            u = self.model.step(u, F, dt)
            phi_k = self.phi.speed(u, ud)
            worst = max(worst, phi_k - ((decay ** k) * phi0 + self.eps * k * dt))
        return worst, phi_k

    def filter(self, cand_vec, nom_vec, ctrl, u, ud, F0, tau_u_fn, slew, dt):
        """Returns (theta_v_vec, alpha, intervened, recovery_mode, phi0_speed)."""
        phi0 = self.phi.speed(u, ud)
        cand = project_to_C_v(cand_vec)
        nom = np.asarray(nom_vec, dtype=float)
        if np.allclose(cand, nom):
            return nom.copy(), 1.0, False, False, phi0
        args = (ctrl, u, ud, F0, tau_u_fn, slew, dt, phi0)
        nom_excess, nom_term = self._rollout(vec_to_theta(nom), *args)
        recovery = nom_excess > 0.0
        if recovery:
            self.total_recovery_mode += 1
        for a in self.alphas[:-1]:
            th_vec = nom + a * (cand - nom)
            excess, term = self._rollout(vec_to_theta(th_vec), *args)
            ok = (excess <= 0.0) if not recovery else \
                 (excess <= nom_excess + self.tol and term <= nom_term + self.tol)
            if ok:
                if a < 1.0:
                    self.total_interventions += 1
                return th_vec, a, a < 1.0, recovery, phi0
        self.total_interventions += 1
        return nom.copy(), 0.0, True, recovery, phi0


# =============================================================================
# RL AGENT: linear-tanh policy trained with antithetic evolution strategies
# =============================================================================
class ESLinearPolicyAgent:
    """pi_psi(s) = theta_nom + scale * tanh(W s + b), clipped to the parameter box.
    The action is the stacked vector [theta_heading (9), theta_velocity (9)] when
    velocity adaptation is on, else theta_heading only.

    Modes:
      off    - always theta_nom (baseline NLPID)
      random - theta_nom + scale*tanh(noise*N(0,1)) (reproduces the previous
               unlearned-exploration behaviour, for ablation)
      es     - antithetic ES with centred-rank fitness shaping; one episode per
               perturbation, update after 2*n_pairs episodes
      eval   - mean policy, no exploration, no updates
    With W = b = 0 (initialisation) the policy equals theta_nom.
    """

    MODES = ('off', 'random', 'es', 'eval')

    def __init__(self, state_dim, theta_nom_vec, lo, hi, param_names, mode='es', n_pairs=4,
                 sigma=0.1, lr=0.05, action_scale_frac=0.25, random_noise_std=0.1, seed=None,
                 bias_only=False, turn_weight=0.0):
        self.bias_only = bool(bias_only)
        self.turn_weight = float(turn_weight)     # weight of the mean turn cost in the episode return
        self.turn_cost, self.turn_steps = 0.0, 0
        self.last_turn_steps = 0
        if mode not in self.MODES:
            raise ValueError(f"rl_mode must be one of {self.MODES}, got '{mode}'")
        self.mode = mode
        self.state_dim = state_dim
        self.lo, self.hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        self.param_names = list(param_names)
        self.action_dim = len(self.lo)
        self.theta_nom = np.asarray(theta_nom_vec, dtype=float)
        self.scale = action_scale_frac * (self.hi - self.lo)
        self.n_params = self.action_dim * (state_dim + 1)
        self.params = np.zeros(self.n_params)
        self.n_pairs = n_pairs
        self.sigma = sigma
        self.lr = lr
        self.random_noise_std = random_noise_std
        self.rng = np.random.default_rng(seed)
        self.iteration = 0
        self.episode_count = 0
        self.episode_cost = 0.0
        self.episode_steps = 0
        self.last_return = None
        self.return_history = []
        self._new_population()
        self.begin_episode()

    def _new_population(self):
        self.pop_eps = self.rng.standard_normal((self.n_pairs, self.n_params))
        if getattr(self, 'bias_only', False):
            self.pop_eps[:, :self.action_dim * self.state_dim] = 0.0   # W stays 0: state-independent gains
        self.pop_returns = np.full((self.n_pairs, 2), np.nan)
        self.pop_slot = 0

    def begin_episode(self):
        self.episode_cost = 0.0
        self.episode_steps = 0
        self.turn_cost, self.turn_steps = 0.0, 0
        if self.mode == 'es':
            i, j = divmod(self.pop_slot, 2)
            sign = 1.0 if j == 0 else -1.0
            self.active_params = self.params + sign * self.sigma * self.pop_eps[i]
        else:
            self.active_params = self.params.copy()

    def _policy(self, params, s):
        n_w = self.action_dim * self.state_dim
        W = params[:n_w].reshape(self.action_dim, self.state_dim)
        b = params[n_w:]
        return np.tanh(W @ s + b)

    def act(self, s):
        """Returns (theta_candidate_vec, normalised_action)."""
        if self.mode == 'off':
            a = np.zeros(self.action_dim)
        elif self.mode == 'random':
            a = np.tanh(self.random_noise_std * self.rng.standard_normal(self.action_dim))
        else:
            a = self._policy(self.active_params, np.asarray(s, dtype=float))
        return np.clip(self.theta_nom + self.scale * a, self.lo, self.hi), a

    def record(self, cost):
        self.episode_cost += cost
        self.episode_steps += 1

    def record_turn(self, cost):
        """Cost of one step inside a waypoint-switch transient (used only when turn_weight > 0)."""
        self.turn_cost += cost
        self.turn_steps += 1

    def end_episode(self):
        """Close the current episode. Returns (return, updated_flag) or (None, False)."""
        if self.episode_steps == 0:
            return None, False
        R = -self.episode_cost / self.episode_steps
        if self.turn_weight > 0.0 and self.turn_steps > 0:
            # mean, not sum: an episode is not penalised for containing more or longer turns
            R -= self.turn_weight * self.turn_cost / self.turn_steps
        self.last_turn_steps = self.turn_steps
        self.last_return = R
        self.return_history.append(R)
        self.episode_count += 1
        updated = False
        if self.mode == 'es':
            i, j = divmod(self.pop_slot, 2)
            self.pop_returns[i, j] = R
            self.pop_slot += 1
            if self.pop_slot >= 2 * self.n_pairs:
                self._update()
                updated = True
        self.begin_episode()
        return R, updated

    def _update(self):
        flat = self.pop_returns.ravel()
        ranks = np.empty(len(flat))
        ranks[np.argsort(flat)] = np.arange(len(flat))
        centred = (ranks / (len(flat) - 1) - 0.5).reshape(self.n_pairs, 2)
        grad = ((centred[:, 0] - centred[:, 1])[:, None] * self.pop_eps).sum(0) / (self.n_pairs * self.sigma)
        self.params = self.params + self.lr * grad
        self.iteration += 1
        self._new_population()

    def to_dict(self):
        return {'params': self.params.tolist(), 'iteration': self.iteration,
                'episode_count': self.episode_count, 'state_dim': self.state_dim,
                'action_dim': self.action_dim, 'sigma': self.sigma, 'lr': self.lr,
                'theta_nom': self.theta_nom.tolist(), 'param_keys': self.param_names,
                'return_history': self.return_history[-500:]}

    def load_dict(self, d):
        old_a = d.get('action_dim')
        if d.get('state_dim') != self.state_dim or old_a is None or old_a > self.action_dim:
            raise ValueError("policy file dimensions do not match")
        old = np.asarray(d['params'], dtype=float)
        if old_a == self.action_dim:
            self.params = old
        else:
            # warm start: a heading-only policy (first old_a outputs) is embedded;
            # the new outputs (velocity parameters) start at zero = nominal.
            n_w_old = old_a * self.state_dim
            W = np.zeros((self.action_dim, self.state_dim))
            b = np.zeros(self.action_dim)
            W[:old_a] = old[:n_w_old].reshape(old_a, self.state_dim)
            b[:old_a] = old[n_w_old:]
            self.params = np.concatenate([W.ravel(), b])
        self.iteration = int(d.get('iteration', 0))
        self.return_history = list(d.get('return_history', []))
        self._new_population()
        self.begin_episode()


# =============================================================================
# KALMAN DISTURBANCE (DRIFT) OBSERVER
# =============================================================================
class KalmanDriftEstimator:
    """2D random-walk Kalman filter on the lateral (sideslip/current) drift
    velocity in the world frame. The innovation norm is used as a runtime proxy
    for the disturbance-estimation error d_tilde."""

    def __init__(self, process_noise=0.01, measurement_noise=0.5):
        self.n = 2
        self.x = np.zeros((self.n, 1))
        self.P = np.eye(self.n)
        self.Q = np.eye(self.n) * process_noise
        self.R = np.eye(self.n) * measurement_noise
        self.last_innovation = np.zeros(2)

    def predict(self):
        self.P = self.P + self.Q

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape((self.n, 1))
        y = z - self.x
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K) @ self.P
        self.last_innovation = y.ravel()

    def get_drift(self):
        return float(self.x[0, 0]), float(self.x[1, 0])


# =============================================================================
# RUNTIME LYAPUNOV MONITOR (paper eq. 2)
# =============================================================================
class LyapunovMonitor:
    def __init__(self, k_e=1.0, alpha0=0.5, beta0=1.0, tol=0.02, filt=0.8):
        self.k_e = k_e
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.tol = tol
        self.filt = filt
        self.prev_V = None
        self.V_dot = 0.0
        self.violations = 0
        self.checks = 0

    def reset(self):
        self.prev_V = None
        self.V_dot = 0.0

    def update(self, e, sigma, d_norm, dt, skip=False):
        V = 0.5 * sigma ** 2 + 0.5 * self.k_e * e ** 2
        if self.prev_V is None or skip:
            self.prev_V = V
            self.V_dot = 0.0
            return V, 0.0, False
        raw = (V - self.prev_V) / dt
        self.V_dot = self.filt * self.V_dot + (1.0 - self.filt) * raw
        self.prev_V = V
        violation = self.V_dot > -self.alpha0 * V + self.beta0 * d_norm + self.tol
        self.checks += 1
        if violation:
            self.violations += 1
        return V, self.V_dot, violation


# =============================================================================
# LOG SCHEMA
# =============================================================================
LOG_FIELDS = (
    ['time_elapsed', 'episode', 'step', 'es_iteration', 'rl_mode', 'shield_enabled',
     'heading_error_deg', 'yaw_rate', 'yaw_rate_imu', 'cross_track_error', 'cte_path',
     'velocity_error', 'current_speed', 'desired_speed_eff',
     'step_cost', 'turn_cost', 'ref_transient', 'learn_step', 'episode_return_last',
     'phi', 'phi_heading', 'phi_speed',
     'intervention', 'shield_alpha', 'recovery_mode',
     'intervention_v', 'shield_alpha_v', 'recovery_mode_v',
     'disturbance_gate', 'drift_x', 'drift_y', 'd_tilde_proxy',
     'nomoto_K', 'nomoto_T', 'nomoto_identified', 'yaw_dist_hat',
     'lyap_V', 'lyap_Vdot', 'lyap_violation', 'lyap_v_V', 'lyap_v_violation']
    + [f'cand_{k}' for k in PARAM_ASCII]
    + PARAM_ASCII
    + [f'cand_{k}' for k in PARAM_ASCII_V]
    + PARAM_ASCII_V
    + ['heading_K_p', 'heading_K_d', 'heading_K_i',
       'velocity_K_p', 'velocity_K_d', 'velocity_K_i',
       'tau_r', 'tau_u', 'tau_ff', 'left_thrust', 'right_thrust',
       'distance_to_target', 'obstacle_detected',
       'x', 'y', 'yaw_deg', 'target_x', 'target_y']
)


# =============================================================================
# EXPERIMENT PRESETS (A/B/C shield comparison)
# =============================================================================
# Select the experiment with the ROS parameter `experiment` (preferred):
#     ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=A -p rl_seed:=7
#     A  = nominal NLPID, RL off (reference)
#     B  = aggressive random gains, shield OFF
#     C  = the same aggressive random gains, shield ON
#     ES = ES learning with the shield (normal operation)
# The environment variable BURAN_EXPERIMENT is still accepted if the parameter is not set.
# A preset only changes the *defaults*; a value set explicitly in the launch
# file still wins. B and C use the same rl_seed, so the random gain proposals
# arrive in the same order and the only difference is the shield.
_COMMON = {'surge_priority_deg': 20.0, 'rl_seed': 7}
_AGGRESSIVE = {'rl_mode': 'random', 'rl_random_noise_std': 1.0,
               'rl_action_scale_frac': 1.0, 'rl_rate_limit_frac': 0.2}
EXPERIMENT_PRESETS = {
    'A': {**_COMMON, 'rl_mode': 'off', 'shield_enabled': False},
    'B': {**_COMMON, **_AGGRESSIVE, 'shield_enabled': False},
    'C': {**_COMMON, **_AGGRESSIVE, 'shield_enabled': True},
    'ES': {'surge_priority_deg': 20.0},         # ES learning + shield; same allocation setting as A/B/C
    # Unconstrained learner: random proposals from the enlarged set (PARAM_BOUNDS_WIDE).
    #   BU = applied as proposed: no projection onto C_theta, no barrier, gain caps lifted
    #   CU = the same raw proposals through the full shield (projection + predictive barrier)
    'BU': {**_COMMON, **_AGGRESSIVE, 'shield_enabled': False, 'unconstrained': True},
    'CU': {**_COMMON, **_AGGRESSIVE, 'shield_enabled': True, 'unconstrained': True},
    # Mistuned-nominal learning study: the original heading gain (torque_scale 600), which is
    # oscillatory at 1.2 m/s. AM = that controller with RL off (reference); ESM = ES learning
    # with the shield starting from it, with a wider action range and its own policy file.
    'AM': {**_COMMON, 'rl_mode': 'off', 'shield_enabled': False, 'torque_scale': 600.0},
    'ESM': {'surge_priority_deg': 20.0, 'torque_scale': 600.0, 'rl_action_scale_frac': 0.5,
            'policy_file': os.path.join(LOG_DIR, 'safe_rl_nlpid_policy_esm.json')},
}


# =============================================================================
# BURAN CONTROLLER NODE
# =============================================================================
class BuranController(Node):
    STATE_DIM = 6

    def __init__(self):
        super().__init__('buran_controller_node')
        self.declare_parameter('experiment', '')
        exp_param = str(self.get_parameter('experiment').value).strip().upper()
        self.experiment = exp_param or os.environ.get('BURAN_EXPERIMENT', '').strip().upper()
        if self.experiment and self.experiment not in EXPERIMENT_PRESETS:
            raise ValueError(f"BURAN_EXPERIMENT must be one of {sorted(EXPERIMENT_PRESETS)}, "
                             f"got '{self.experiment}'")
        preset = EXPERIMENT_PRESETS.get(self.experiment, {})

        def dp(name, default):
            return self.declare_parameter(name, preset.get(name, default))

        # ---------- experiment / ablation switches ----------
        dp('rl_mode', 'es')                 # off | random | es | eval
        dp('shield_enabled', True)
        dp('shield_predictive', True)       # False: projection onto C_theta only (ablation, no predictive barrier)
        dp('es_bias_only', False)           # True: ES learns only the offset w0 (W = 0): an optimised fixed-gain baseline
        dp('unconstrained', False)          # True: the learner proposes from the enlarged set (BU / CU)
        dp('unconstrained_gain_cap_scale', 4.0)   # BU only: certified gain caps multiplied by this
        dp('rl_hold_steps', 20)             # decision period (steps) = 1 s at 20 Hz
        dp('rl_episode_length', 400)        # steps per ES episode (20 s)
        dp('rl_rate_limit_frac', 0.05)      # max |delta theta| per decision, fraction of range
        dp('es_n_pairs', 4)
        dp('es_sigma', 0.1)
        dp('es_lr', 0.05)
        dp('rl_action_scale_frac', 0.25)
        dp('rl_random_noise_std', 0.1)
        dp('rl_seed', 0)
        dp('policy_file', os.path.join(LOG_DIR, 'safe_rl_nlpid_policy.json'))
        dp('load_policy', True)
        # Learning cost (per step, settled path-following only):
        #   c = min((e/e_scale)^2, clip) + w_tau*min((tau_r/tau_scale)^2, clip)
        #       + w_cte*(y_e/cte_scale)^2 + w_speed*(e_u/speed_scale)^2 + w_shield*intervened
        # The heading-error and yaw-torque terms are the ones theta acts on and are scaled to be of
        # comparable size (1 deg and 200 N). The cross-track term is off by default: y_e is set by
        # the guidance law and the path geometry, and it used to make up about two thirds of the
        # cost, so episode returns reflected where the episode ran rather than the gains tried.
        dp('cost_e_scale_deg', 1.0)
        dp('cost_tau_scale', 200.0)
        dp('cost_clip', 4.0)
        dp('cost_w_cte', 0.0)
        dp('cost_cte_scale', 10.0)
        dp('cost_w_tau', 1.0)
        dp('cost_w_shield', 0.0)
        dp('cost_skip_after_ref_change', 8.0)  # s; exclude waypoint-switch transients from the return
        # Turn-aware cost (0 = settled-only cost, as in the ECC runs). Inside the transient after a
        # waypoint switch, each step costs min((e_post/scale)^2, clip) + w_sat*[thruster at limit],
        # where e_post = |e| once e has first crossed zero (overshoot and ringing), 0 before that
        # (the approach error is set by the turn angle, not by the gains). The episode return adds
        # cost_w_turn times the MEAN turn cost of the episode.
        dp('cost_w_turn', 0.0)
        dp('cost_turn_scale_deg', 2.0)
        dp('cost_turn_w_sat', 1.0)
        dp('cost_turn_min_steps', 40)       # an episode must contain >= this many turn steps (2 s)
        dp('cost_turn_max_steps', 3000)     # safety cap on settled steps if no switch comes (150 s)
        # Speed in the policy state (needed for speed-dependent gains) and an optional speed schedule.
        dp('rl_state_speed', False)         # True: append u/1.5 to the policy state (state_dim 7)
        dp('speed_schedule_period', 0.0)    # s of sim time; > 0 alternates u_d between low and high
        dp('speed_schedule_low', 0.6)
        dp('speed_schedule_high', 1.5)
        dp('cost_skip_after_speed_change', 6.0)  # s; exclude speed-step transients from the settled cost

        # ---------- heading NLPID ----------
        dp('heading_K_p', 4.0)
        dp('heading_K_d', 2.5)
        dp('heading_K_i', 0.08)
        dp('heading_mu_p', 0.35)
        dp('heading_mu_d', 0.5)
        dp('heading_mu_i', 0.6)
        dp('heading_eps_p', 0.15)
        dp('heading_eps_d', 0.1)
        dp('heading_eps_i', 0.2)
        dp('heading_lambda', 0.1)
        dp('heading_boundary_layer', 0.5)
        dp('heading_robust_gain', 0.05)
        dp('heading_Kp_max', 20.0)          # was 6.0 (pinned K_p for |e| < ~31 deg)
        dp('heading_Kd_max', 10.0)          # was 5.0
        dp('heading_Ki_max', 0.5)           # was 0.15
        dp('heading_integral_limit', 1.0)
        dp('heading_integral_zone_deg', 20.0)
        dp('torque_scale', 200.0)           # N per unit steering (was 1500 with lower caps)
        dp('forced_turn_deg', 60.0)
        dp('forced_turn_min', 600.0)
        dp('yaw_rate_source', 'heading')    # heading: filtered d(psi)/dt from the orientation | imu: angular_velocity.z
        dp('yaw_rate_filter', 0.5)          # low-pass coefficient for the heading-derived yaw rate (0 = none)
        dp('surge_priority_deg', 0.0)       # >0: below this |e| the yaw torque is limited so the surge thrust is kept (0 = off)

        # ---------- velocity NLPID ----------
        dp('velocity_K_p', 80.0)
        dp('velocity_K_d', 15.0)
        dp('velocity_K_i', 0.5)
        dp('velocity_mu_p', 0.5)
        dp('velocity_mu_d', 0.4)
        dp('velocity_mu_i', 0.6)
        dp('velocity_lambda', 0.05)
        dp('velocity_boundary_layer', 2.0)
        dp('velocity_robust_gain', 0.5)
        dp('velocity_eps_p', 0.1)
        dp('velocity_eps_d', 0.05)
        dp('velocity_eps_i', 0.2)
        dp('velocity_Kp_limits', [20.0, 400.0])   # upper cap raised from 200: at 200 the loop could not brake
        dp('velocity_Kd_limits', [5.0, 50.0])
        dp('velocity_Ki_limits', [0.1, 5.0])
        dp('rl_adapt_velocity', True)       # RL also adapts the 9 velocity-NLPID parameters
        dp('cost_w_speed', 0.1)             # cost += w * ((u_d - u) / scale)^2
        dp('cost_speed_scale', 0.5)         # m/s

        # ---------- safety shield ----------
        dp('safety_e_max_deg', 30.0)
        dp('safety_u_max', 5.0)
        dp('safety_ud_max', 2.0)
        dp('safety_delta', 0.1)
        dp('safety_eu_max', 0.5)            # m/s, speed-tracking error bound in phi_v (VRX p99 |u_err| = 0.40)
        dp('shield_v_horizon_steps', 40)    # 2 s: surge is slower than yaw
        dp('surge_c1', 1.07e-4)             # surge model u_dot = c1*(L+R) - c2*u|u| (from VRX log)
        dp('surge_c2', 0.161)
        dp('shield_eta', 0.5)
        dp('shield_eps', 0.1)
        dp('shield_horizon_steps', 20)
        dp('nomoto_K_default', 5.2e-4)      # rad/s per N (identified in VRX, 23/09/2026)
        dp('nomoto_T_default', 1.37)        # s        (identified in VRX, 23/09/2026)
        dp('nomoto_forgetting', 0.999)
        dp('nomoto_online_id', True)        # False: use the defaults above (identified offline)

        # ---------- disturbance observer / certificate ----------
        dp('kalman_process_noise', 0.01)
        dp('kalman_measurement_noise', 0.5)
        dp('d_safe', 0.5)                   # m/s, threshold on d_tilde proxy
        dp('drift_compensation_gain', 0.0)  # ILOS integral already rejects constant drift
        dp('lyap_k_e', 1.0)
        dp('lyap_alpha0', 0.5)
        dp('lyap_beta0', 1.0)
        dp('lyap_tol', 0.02)

        # ---------- guidance ----------
        dp('guidance_mode', 'ilos')         # ilos | pursuit
        dp('los_lookahead', 10.0)           # m
        dp('ilos_kappa', 0.1)
        dp('ilos_integral_limit', 20.0)     # m

        # ---------- speed / mission ----------
        dp('base_speed', 200.0)             # feed-forward surge thrust [N] per thruster at desired_speed (VRX: ~190 N holds 1.2 m/s)
        dp('ff_speed_scaling', True)        # scale the feed-forward with (u_d_eff / u_d)^2 (drag ~ u^2)
        dp('ff_min_thrust', 50.0)           # lower bound on the scaled feed-forward [N]
        dp('surge_brake_thrust', 0.0)       # >0: allow reverse surge down to -this [N] when overspeeding (0 = old behaviour)
        dp('surge_brake_margin', 0.1)       # m/s of overspeed before reverse braking is allowed
        dp('desired_speed', 1.2)
        dp('obstacle_slow_factor', 0.5)
        dp('approach_slow_distance', 10.0)
        dp('approach_slow_factor', 0.2)
        dp('slew_rate_limit', 200.0)
        dp('min_safe_distance', 12.0)
        dp('stuck_timeout', 3.0)
        dp('stuck_threshold', 0.5)

        g = lambda n: self.get_parameter(n).value

        # ---------- read parameters ----------
        self.rl_mode = str(g('rl_mode'))
        self.shield_enabled = bool(g('shield_enabled'))
        self.shield_predictive = bool(g('shield_predictive'))
        self.unconstrained = bool(g('unconstrained'))
        self.hold_steps = max(1, int(g('rl_hold_steps')))
        self.episode_length = max(1, int(g('rl_episode_length')))
        self.cost_w_turn = float(g('cost_w_turn'))
        self.cost_turn_scale = math.radians(float(g('cost_turn_scale_deg')))
        self.cost_turn_w_sat = float(g('cost_turn_w_sat'))
        self.turn_min_steps = int(g('cost_turn_min_steps'))
        self.turn_max_steps = int(g('cost_turn_max_steps'))
        self.rl_state_speed = bool(g('rl_state_speed'))
        self.state_dim = self.STATE_DIM + (1 if self.rl_state_speed else 0)
        self.speed_period = float(g('speed_schedule_period'))
        self.speed_low, self.speed_high = float(g('speed_schedule_low')), float(g('speed_schedule_high'))
        self.cost_skip_speed = float(g('cost_skip_after_speed_change'))
        self.speed_change_time = None
        self._sched_level = None
        self.turn_sign0, self.turn_crossed = None, False
        self.turn_cost_last = 0.0
        self.max_delta = float(g('rl_rate_limit_frac')) * (PARAM_HI - PARAM_LO)
        self.policy_file = g('policy_file')
        self.cost_w_cte = g('cost_w_cte')
        self.cost_cte_scale = g('cost_cte_scale')
        self.cost_w_tau = g('cost_w_tau')
        self.cost_w_shield = g('cost_w_shield')
        self.cost_skip = g('cost_skip_after_ref_change')
        self.cost_e_scale = math.radians(g('cost_e_scale_deg'))
        self.cost_tau_scale = g('cost_tau_scale')
        self.cost_clip = g('cost_clip')
        self.adapt_v = bool(g('rl_adapt_velocity'))
        self.cost_w_speed = g('cost_w_speed')
        self.cost_speed_scale = g('cost_speed_scale')
        self.max_delta_v = float(g('rl_rate_limit_frac')) * (PARAM_HI_V - PARAM_LO_V)

        self.torque_scale = g('torque_scale')
        self.forced_turn = math.radians(g('forced_turn_deg'))
        self.forced_turn_min = g('forced_turn_min')
        self.yaw_rate_source = str(g('yaw_rate_source'))
        self.yaw_rate_filter = clamp(float(g('yaw_rate_filter')), 0.0, 0.99)
        self.surge_priority = math.radians(g('surge_priority_deg'))

        self.base_speed = g('base_speed')
        self.ff_speed_scaling = bool(g('ff_speed_scaling'))
        self.ff_min_thrust = g('ff_min_thrust')
        self.surge_brake_thrust = max(0.0, float(g('surge_brake_thrust')))
        self.surge_brake_margin = g('surge_brake_margin')
        self.desired_speed = g('desired_speed')
        self.obstacle_slow_factor = g('obstacle_slow_factor')
        self.approach_slow_distance = g('approach_slow_distance')
        self.approach_slow_factor = g('approach_slow_factor')
        self.slew_rate_limit = g('slew_rate_limit')
        self.min_safe_distance = g('min_safe_distance')
        self.stuck_timeout = g('stuck_timeout')
        self.stuck_threshold = g('stuck_threshold')

        self.guidance_mode = str(g('guidance_mode'))
        self.los_lookahead = g('los_lookahead')
        self.ilos_kappa = g('ilos_kappa')
        self.ilos_integral_limit = g('ilos_integral_limit')
        self.d_safe = g('d_safe')
        self.drift_compensation_gain = g('drift_compensation_gain')

        self.nomoto_online_id = bool(g('nomoto_online_id'))
        self.dt = 0.05

        # ---------- nominal (certified) theta ----------
        theta_nom = {
            'μ_p': g('heading_mu_p'), 'μ_d': g('heading_mu_d'), 'μ_i': g('heading_mu_i'),
            'ε_p': g('heading_eps_p'), 'ε_d': g('heading_eps_d'), 'ε_i': g('heading_eps_i'),
            'λ': g('heading_lambda'), 'boundary_layer': g('heading_boundary_layer'),
            'robust_gain': g('heading_robust_gain'),
        }
        self.theta_nom_vec = project_to_C(theta_to_vec(theta_nom))
        self.theta_applied_vec = self.theta_nom_vec.copy()
        self.theta_candidate_vec = self.theta_nom_vec.copy()
        self.theta_cand_raw = self.theta_nom_vec.copy()
        theta_nom_v = {
            'μ_p': g('velocity_mu_p'), 'μ_d': g('velocity_mu_d'), 'μ_i': g('velocity_mu_i'),
            'ε_p': g('velocity_eps_p'), 'ε_d': g('velocity_eps_d'), 'ε_i': g('velocity_eps_i'),
            'λ': g('velocity_lambda'), 'boundary_layer': g('velocity_boundary_layer'),
            'robust_gain': g('velocity_robust_gain'),
        }
        self.theta_v_nom_vec = project_to_C_v(theta_to_vec(theta_nom_v))
        self.theta_v_applied_vec = self.theta_v_nom_vec.copy()
        self.theta_v_candidate_vec = self.theta_v_nom_vec.copy()
        self.theta_v_cand_raw = self.theta_v_nom_vec.copy()

        # ---------- components ----------
        self.heading_controller = HeadingNLPID(
            g('heading_K_p'), g('heading_K_d'), g('heading_K_i'),
            vec_to_theta(self.theta_nom_vec),
            gain_limits={'K_p': (0.05, g('heading_Kp_max')),
                         'K_d': (0.1, g('heading_Kd_max')),
                         'K_i': (0.001, g('heading_Ki_max'))},
            integral_limit=g('heading_integral_limit'),
            integral_zone=math.radians(g('heading_integral_zone_deg')))

        self.velocity_controller = VelocityNLPID(
            g('velocity_K_p'), g('velocity_K_d'), g('velocity_K_i'),
            vec_to_theta(self.theta_v_nom_vec),
            gain_limits={'K_p': tuple(g('velocity_Kp_limits')),
                         'K_d': tuple(g('velocity_Kd_limits')),
                         'K_i': tuple(g('velocity_Ki_limits'))},
            integral_limit=200.0, derivative_filter=0.8)

        self.safety_index = SafetyIndex(g('safety_e_max_deg'), g('safety_u_max'),
                                        g('safety_ud_max'), g('safety_delta'), g('safety_eu_max'))
        self.shield = ParameterSpaceShield(self.safety_index, eta=g('shield_eta'),
                                           eps=g('shield_eps'),
                                           horizon_steps=int(g('shield_horizon_steps')))
        self.nomoto = NomotoRLS(self.dt, g('nomoto_K_default'), g('nomoto_T_default'),
                                forgetting=g('nomoto_forgetting'))
        self.drift_kalman = KalmanDriftEstimator(g('kalman_process_noise'),
                                                 g('kalman_measurement_noise'))
        self.lyap = LyapunovMonitor(g('lyap_k_e'), g('lyap_alpha0'), g('lyap_beta0'), g('lyap_tol'))
        self.lyap_v = LyapunovMonitor(g('lyap_k_e'), g('lyap_alpha0'), g('lyap_beta0'), g('lyap_tol'))
        self.surge_model = SurgeModel(g('surge_c1'), g('surge_c2'))
        self.shield_v = VelocityShield(self.safety_index, self.surge_model, eta=g('shield_eta'),
                                       eps=g('shield_eps'),
                                       horizon_steps=int(g('shield_v_horizon_steps')))

        seed = int(g('rl_seed'))
        # proposal set of the learner: the certified box, or the enlarged set (BU / CU)
        self.prop_lo, self.prop_hi = (PARAM_WIDE_LO, PARAM_WIDE_HI) if self.unconstrained else (PARAM_LO, PARAM_HI)
        self.prop_lo_v, self.prop_hi_v = ((PARAM_WIDE_LO_V, PARAM_WIDE_HI_V) if self.unconstrained
                                          else (PARAM_LO_V, PARAM_HI_V))
        if self.unconstrained and not self.shield_enabled:
            cap = float(g('unconstrained_gain_cap_scale'))
            self.heading_controller.uncertify(PARAM_BOUNDS_WIDE, cap)
            self.velocity_controller.uncertify(PARAM_BOUNDS_WIDE_V, cap)
        if self.adapt_v:
            nom_all = np.concatenate([self.theta_nom_vec, self.theta_v_nom_vec])
            lo_all = np.concatenate([self.prop_lo, self.prop_lo_v])
            hi_all = np.concatenate([self.prop_hi, self.prop_hi_v])
            names = PARAM_ASCII + PARAM_ASCII_V
        else:
            nom_all, lo_all, hi_all, names = self.theta_nom_vec, self.prop_lo, self.prop_hi, PARAM_ASCII
        self.agent = ESLinearPolicyAgent(
            self.state_dim, nom_all, lo_all, hi_all, names, mode=self.rl_mode,
            n_pairs=int(g('es_n_pairs')), sigma=g('es_sigma'), lr=g('es_lr'),
            action_scale_frac=g('rl_action_scale_frac'),
            random_noise_std=g('rl_random_noise_std'),
            seed=seed if seed != 0 else None,
            bias_only=bool(g('es_bias_only')), turn_weight=self.cost_w_turn)
        if self.rl_mode in ('es', 'eval') and g('load_policy') and os.path.exists(self.policy_file):
            try:
                with open(self.policy_file) as f:
                    self.agent.load_dict(json.load(f))
                self.get_logger().info(f"Loaded policy (ES iteration {self.agent.iteration}, "
                                       f"{self.agent.action_dim} outputs) from {self.policy_file}")
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as ex:
                self.get_logger().warn(f"Could not load policy: {ex} - starting from nominal")

        # ---------- state ----------
        self.current_yaw = 0.0
        self.yaw_rate = 0.0          # IMU angular_velocity.z (logged for comparison)
        self.r_est = 0.0
        self.prev_yaw_for_rate = None
        self.prev_rate_time = None
        self.target_x = None
        self.target_y = None
        self.last_target = None
        self.segment_start = None
        self.ilos_integral = 0.0
        self.ref_change_time = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.distance_to_target = float('inf')
        self.current_speed = 0.0
        self.gps_vel = (0.0, 0.0)
        self.gps_vel_new = False
        self.all_waypoints = []
        self.waypoints_received = False
        self.cross_track_error = 0.0
        self.cte_path = 0.0
        self.mission_active = False
        self.stop_override = False
        self.obstacle_detected = False
        self.min_obstacle_distance = float('inf')
        self.front_clear = float('inf')
        self.left_clear = float('inf')
        self.right_clear = float('inf')
        self.is_critical = False
        self.urgency = 0.0
        self.obstacle_count = 0
        self.prev_left_thrust = 0.0
        self.prev_right_thrust = 0.0
        self.last_tau_u = 100.0
        self.prev_r = None
        self.prev_tau_applied = None
        self.last_position = None
        self.stuck_check_time = None
        self.is_stuck = False
        self.escape_mode = False
        self.consecutive_stuck_count = 0
        self.start_gps = None
        self.gps_xy_prev = None
        self.gps_raw_xy = None        # latest GPS position in the controller's own local frame
        self._frame_off = (0.0, 0.0)  # planner frame minus controller frame [m] (difference of the two origins)
        self._frame_off_still = []    # offset samples taken while the boat is nearly at rest
        self._frame_off_all = []      # offset samples at any speed (fallback)
        self._frame_off_logged = None
        self.gps_time_prev = None
        self.decision_counter = 0
        self.disturbance_gate = False
        self.d_tilde_proxy = 0.0

        # ---------- logging ----------
        self.logging_enabled = True
        self.log_rows = []
        log_dir = LOG_DIR
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_tag = (f'exp{self.experiment}_s{int(self.get_parameter("rl_seed").value)}_'
                   if self.experiment else '')
        self.log_file = os.path.join(log_dir, f'controller_safe_rl_{exp_tag}{self.rl_mode}_'
                                              f'{"shield" if self.shield_enabled else "noshield"}_{stamp}.csv')
        with open(self.log_file, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()
        self.start_time = self.get_clock().now()
        self.create_timer(30.0, self.flush_log_to_file)

        # ---------- ROS I/O ----------
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.gps_callback, 10)
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, 10)
        self.create_subscription(String, '/planning/current_target', self.target_callback, 10)
        self.create_subscription(String, '/planning/waypoints', self.waypoints_callback, 10)
        self.create_subscription(String, '/perception/obstacle_info', self.obstacle_callback, 10)
        self.create_subscription(String, '/planning/mission_status', self.mission_status_callback, 10)

        self.pub_left = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.pub_right = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.pub_status = self.create_publisher(String, '/control/status', 10)
        self.pub_rl_info = self.create_publisher(String, '/control/rl_info', 10)
        self.pub_skip_request = self.create_publisher(String, '/planning/skip_waypoint', 10)

        self.create_timer(self.dt, self.control_loop)

        log = self.get_logger()
        log.info("=" * 70)
        log.info("BURAN - Safe-RL-NLPID (heading + velocity gains, parameter-space shields, ES, ILOS)")
        if self.experiment:
            log.info("#" * 70)
            try:
                sim_time = self.get_parameter('use_sim_time').value
            except Exception:
                sim_time = '?'
            log.info(f"#   EXPERIMENT {self.experiment}   rl_seed={g('rl_seed')}   sim_time={sim_time}")
            log.info("#" * 70)
            log.info(f"EXPERIMENT {self.experiment}: rl_mode={self.rl_mode} shield={'ON' if self.shield_enabled else 'OFF'} "
                     f"rl_seed={g('rl_seed')} surge_priority_deg={g('surge_priority_deg')}")
        log.info(f"rl_mode={self.rl_mode}  shield={'ON' if self.shield_enabled else 'OFF'}  "
                 f"guidance={self.guidance_mode}")
        log.info("theta_nom (heading) = " + ", ".join(f"{k}={v:.3f}" for k, v in zip(PARAM_ASCII, self.theta_nom_vec)))
        log.info("theta_nom (velocity) = " + ", ".join(f"{k}={v:.3f}" for k, v in zip(PARAM_ASCII_V, self.theta_v_nom_vec))
                 + f"  | RL adapts velocity: {'YES' if self.adapt_v else 'NO'}")
        log.info(f"Safe set: |e_psi| <= {(1 - self.safety_index.delta) * math.degrees(self.safety_index.e_max):.1f} deg")
        log.info(f"Logging to {self.log_file}")
        log.info("=" * 70)

    # ======================================================================
    # SENSOR CALLBACKS
    # ======================================================================
    def latlon_to_meters(self, lat, lon):
        """Local ENU: returns (east, north) in metres relative to the first fix."""
        if self.start_gps is None:
            return 0.0, 0.0
        R = 6371000.0
        d_lat = math.radians(lat - self.start_gps[0])
        d_lon = math.radians(lon - self.start_gps[1])
        north = d_lat * R
        east = d_lon * R * math.cos(math.radians(self.start_gps[0]))
        return east, north

    def _msg_time(self, msg):
        """Time of a sensor message in seconds: its header stamp (exact simulation time set
        by Gazebo); the node clock only if the stamp is empty. Using the clock at callback time
        instead makes dt jitter when the simulation runs slower than real time, and position or
        heading differences divided by a jittery dt overestimate speed and yaw rate."""
        st = getattr(getattr(msg, 'header', None), 'stamp', None)
        if st is not None and (st.sec or st.nanosec):
            return st.sec + st.nanosec * 1e-9
        return self.get_clock().now().nanoseconds * 1e-9

    def gps_callback(self, msg):
        now = self._msg_time(msg)
        if self.start_gps is None:
            self.start_gps = (msg.latitude, msg.longitude)
            self.gps_time_prev = now
            return
        x, y = self.latlon_to_meters(msg.latitude, msg.longitude)
        # Position used for guidance = GPS position moved into the planner's frame, so that it is
        # in the same frame as the waypoints at every sample (see target_callback).
        self.gps_raw_xy = (x, y)
        self.current_x, self.current_y = x + self._frame_off[0], y + self._frame_off[1]
        if self.gps_xy_prev is not None and self.gps_time_prev is not None:
            dt = now - self.gps_time_prev
            if dt > 0.01:
                vx = (x - self.gps_xy_prev[0]) / dt
                vy = (y - self.gps_xy_prev[1]) / dt
                if math.hypot(vx, vy) <= MAX_GPS_SPEED:
                    a = 0.5
                    self.gps_vel = (a * self.gps_vel[0] + (1 - a) * vx,
                                    a * self.gps_vel[1] + (1 - a) * vy)
                    self.current_speed = math.hypot(*self.gps_vel)
                    self.gps_vel_new = True
        self.gps_xy_prev = (x, y)
        self.gps_time_prev = now

    def imu_callback(self, msg):
        self.imu_time = self._msg_time(msg)
        q = msg.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.yaw_rate = msg.angular_velocity.z

    def target_callback(self, msg):
        try:
            data = json.loads(msg.data)
            # The planner reports the position in ITS local frame (origin = its first GPS fix).
            # The controller's frame has another origin (its own first fix), so the two differ by
            # a constant offset. Estimate that offset here and apply it in gps_callback; do not
            # overwrite the position, which would make it jump between the two frames.
            px, py = data['current_position']
            if self.gps_raw_xy is None:
                self.current_x, self.current_y = px, py
            else:
                off = (px - self.gps_raw_xy[0], py - self.gps_raw_xy[1])
                buf = self._frame_off_still if self.current_speed < 0.3 else self._frame_off_all
                buf.append(off)
                if len(buf) > 400:
                    del buf[0]
                use = (self._frame_off_still
                       if (len(self._frame_off_still) >= 5 or not self._frame_off_all)
                       else self._frame_off_all)
                self._frame_off = (float(np.median([o[0] for o in use])),
                                   float(np.median([o[1] for o in use])))
                self.current_x = self.gps_raw_xy[0] + self._frame_off[0]
                self.current_y = self.gps_raw_xy[1] + self._frame_off[1]
                last = self._frame_off_logged
                if last is None or math.hypot(self._frame_off[0] - last[0], self._frame_off[1] - last[1]) > 0.10:
                    self._frame_off_logged = self._frame_off
                    self.get_logger().info(
                        f'Frame offset planner-controller = ({self._frame_off[0]:+.2f}, {self._frame_off[1]:+.2f}) m')
            self.target_x, self.target_y = data['target_waypoint']
            self.distance_to_target = data['distance_to_target']
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            self.get_logger().warn(f"Invalid target: {e}")

    def waypoints_callback(self, msg):
        try:
            data = json.loads(msg.data)
            wps = data.get('waypoints', [])
            self.all_waypoints = [(float(w[0]), float(w[1])) for w in wps if len(w) >= 2]
            self.waypoints_received = len(self.all_waypoints) > 0
            if self.waypoints_received:
                self.get_logger().info(f"Loaded {len(self.all_waypoints)} waypoints")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            self.get_logger().warn(f"Invalid waypoints: {e}")

    def obstacle_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.obstacle_detected = data.get('obstacle_detected', False)
            self.min_obstacle_distance = data.get('min_distance', float('inf'))
            self.front_clear = data.get('front_clear', float('inf'))
            self.left_clear = data.get('left_clear', float('inf'))
            self.right_clear = data.get('right_clear', float('inf'))
            self.is_critical = data.get('is_critical', False)
            self.urgency = data.get('urgency', 0.0)
            self.obstacle_count = data.get('obstacle_count', 0)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.get_logger().warn(f"Invalid obstacle: {e}")

    def mission_status_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.mission_active = (data.get('state', '') == "DRIVING")
            if not self.mission_active:
                self.stop()
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.get_logger().warn(f"Invalid mission status: {e}")

    # ======================================================================
    # GUIDANCE
    # ======================================================================
    def _handle_target_change(self):
        tgt = (self.target_x, self.target_y)
        if self.last_target is not None and math.hypot(tgt[0] - self.last_target[0],
                                                       tgt[1] - self.last_target[1]) < 0.5:
            return
        start = None
        if self.waypoints_received:
            dists = [math.hypot(w[0] - tgt[0], w[1] - tgt[1]) for w in self.all_waypoints]
            idx = int(np.argmin(dists))
            if dists[idx] < 1.0 and idx > 0:
                start = self.all_waypoints[idx - 1]
        if start is None:
            start = self.last_target if self.last_target is not None else (self.current_x, self.current_y)
        self.segment_start = start
        self.last_target = tgt
        self.ilos_integral = 0.0
        self.heading_controller.reset()
        self.lyap.reset()
        self.ref_change_time = self.get_clock().now()
        self.turn_sign0, self.turn_crossed = None, False

    def compute_desired_heading(self):
        """Returns (psi_d, signed cross-track error y_e; y_e > 0 = vehicle left of path)."""
        tx, ty = self.target_x, self.target_y
        cx, cy = self.current_x, self.current_y
        pursuit = math.atan2(ty - cy, tx - cx)
        if self.segment_start is None:
            return pursuit, 0.0
        x1, y1 = self.segment_start
        L = math.hypot(tx - x1, ty - y1)
        if L < 1.0:
            return pursuit, 0.0
        alpha = math.atan2(ty - y1, tx - x1)
        y_e = -(cx - x1) * math.sin(alpha) + (cy - y1) * math.cos(alpha)
        if self.guidance_mode != 'ilos':
            return pursuit, y_e
        # ILOS follows the infinite line through the segment. Once the vehicle is past the end
        # of the segment (the planner skipped a waypoint, or the vehicle overshot it), that line
        # leads AWAY from the waypoint, so steer straight at the waypoint instead.
        s_along = (cx - x1) * math.cos(alpha) + (cy - y1) * math.sin(alpha)
        if s_along >= L:
            return pursuit, y_e
        D = self.los_lookahead
        k = self.ilos_kappa
        denom = D ** 2 + (y_e + k * self.ilos_integral) ** 2
        self.ilos_integral = clamp(self.ilos_integral + self.dt * D * y_e / denom,
                                   -self.ilos_integral_limit, self.ilos_integral_limit)
        psi_d = alpha - math.atan((y_e + k * self.ilos_integral) / D)
        return wrap_angle(psi_d), y_e

    def calculate_path_cte(self):
        """Unsigned distance to the waypoint polyline (kept for comparability with earlier logs)."""
        if not self.waypoints_received or len(self.all_waypoints) < 2:
            return 0.0
        best = float('inf')
        for (x1, y1), (x2, y2) in zip(self.all_waypoints[:-1], self.all_waypoints[1:]):
            dx, dy = x2 - x1, y2 - y1
            l2 = dx * dx + dy * dy
            if l2 < 1e-6:
                d = math.hypot(self.current_x - x1, self.current_y - y1)
            else:
                t = clamp(((self.current_x - x1) * dx + (self.current_y - y1) * dy) / l2, 0.0, 1.0)
                d = math.hypot(self.current_x - (x1 + t * dx), self.current_y - (y1 + t * dy))
            best = min(best, d)
        return min(best, 50.0)

    # ======================================================================
    # DISTURBANCE OBSERVER
    # ======================================================================
    def _update_disturbance_observer(self):
        self.drift_kalman.predict()
        if self.gps_vel_new:
            self.gps_vel_new = False
            hx, hy = math.cos(self.current_yaw), math.sin(self.current_yaw)
            vx, vy = self.gps_vel
            v_along = vx * hx + vy * hy
            lateral = (vx - v_along * hx, vy - v_along * hy)
            self.drift_kalman.update(lateral)
            innov = float(np.linalg.norm(self.drift_kalman.last_innovation))
            self.d_tilde_proxy = 0.9 * self.d_tilde_proxy + 0.1 * innov
        if self.d_tilde_proxy > self.d_safe:
            if not self.disturbance_gate:
                self.get_logger().warn(f"d_tilde proxy {self.d_tilde_proxy:.2f} > d_safe - learning frozen, theta = theta_nom")
            self.disturbance_gate = True
        elif self.d_tilde_proxy < 0.8 * self.d_safe:
            self.disturbance_gate = False

    def _drift_lateral_left(self):
        dx, dy = self.drift_kalman.get_drift()
        return -dx * math.sin(self.current_yaw) + dy * math.cos(self.current_yaw)

    # ======================================================================
    # HELPERS
    # ======================================================================
    def steering_to_torque(self, steering, e):
        tau = clamp(steering * self.torque_scale, -TURN_POWER_LIMIT, TURN_POWER_LIMIT)
        if abs(e) > self.forced_turn:
            if e > 0.0:
                tau = max(tau, self.forced_turn_min)     # CCW / left turn
            else:
                tau = min(tau, -self.forced_turn_min)    # CW / right turn
        return tau

    def _estimate_yaw_rate(self):
        """Yaw rate used by the controller.

        'heading' (default): low-pass-filtered derivative of the yaw angle from the
        IMU orientation. On the VRX WAM-V, angular_velocity.z was observed to be
        dominated by noise (+-2 rad/s sign flips while the heading changed at
        ~0.2 rad/s); used as e_dot it made the derivative term chatter the yaw
        torque between +-1600 N so the boat stalled/span in place.
        'imu': angular_velocity.z directly (only if it is clean)."""
        if self.yaw_rate_source == 'imu':
            return self.yaw_rate
        now = getattr(self, 'imu_time', None)          # stamp of the IMU message that gave current_yaw
        if now is None:
            return self.r_est
        if self.prev_yaw_for_rate is None:
            self.prev_yaw_for_rate, self.prev_rate_time = self.current_yaw, now
            return self.r_est
        dt = now - self.prev_rate_time
        if dt < 1e-3:
            return self.r_est
        raw = wrap_angle(self.current_yaw - self.prev_yaw_for_rate) / dt
        a = self.yaw_rate_filter
        self.r_est = a * self.r_est + (1.0 - a) * clamp(raw, -3.0, 3.0)
        self.prev_yaw_for_rate, self.prev_rate_time = self.current_yaw, now
        return self.r_est

    def _limit_for_surge(self, tau_r, tau_u, e):
        """Surge-preserving allocation: for |e| below surge_priority, limit the yaw
        torque so neither thruster clips and the commanded forward thrust tau_u is
        kept (a clipped L=-800/R=+800 pair gives zero forward thrust)."""
        if abs(e) >= self.surge_priority:
            return tau_r
        lim = max(0.0, SAFE_THRUST - abs(tau_u))
        return clamp(tau_r, -lim, lim)

    def _make_actuator_model(self):
        """Fresh actuator model for one shield rollout: steering -> effective yaw
        torque 0.5*(R - L) after per-thruster clipping and slew limiting, starting
        from the last commanded thrusts and the last surge command."""
        state = {'L': self.prev_left_thrust, 'R': self.prev_right_thrust}
        tau_u = self.last_tau_u

        def act(steering, e):
            tau = self._limit_for_surge(self.steering_to_torque(steering, e), tau_u, e)
            L = self._slew_limit(state['L'], clamp(tau_u - tau, -SAFE_THRUST, SAFE_THRUST))
            R = self._slew_limit(state['R'], clamp(tau_u + tau, -SAFE_THRUST, SAFE_THRUST))
            state['L'], state['R'] = L, R
            return 0.5 * (R - L)
        return act

    def _build_state(self, e, e_dot, y_e, ud_eff):
        sigma = e_dot + self.theta_nom_vec[PARAM_KEYS.index('λ')] * e
        s = np.array([
            e / math.pi,
            clamp(e_dot, -2.0, 2.0),
            clamp(sigma, -2.0, 2.0),
            clamp(y_e / 20.0, -3.0, 3.0),
            clamp(self._drift_lateral_left(), -2.0, 2.0),
            clamp(self.current_speed - ud_eff, -2.0, 2.0),
        ])
        if self.rl_state_speed:
            s = np.append(s, clamp(self.current_speed / 1.5, 0.0, 2.0))
        return s

    def _scheduled_speed(self):
        """Desired speed: the parameter value, or an alternating low/high schedule in sim time."""
        if self.speed_period <= 0.0:
            return self.desired_speed
        t = self.get_clock().now().nanoseconds / 1e9
        level = int(t // self.speed_period) % 2
        if level != self._sched_level:
            if self._sched_level is not None:
                self.speed_change_time = self.get_clock().now()
                self.get_logger().info(f"Speed schedule: u_d -> {self.speed_high if level else self.speed_low:.2f} m/s")
            self._sched_level = level
        return self.speed_high if level else self.speed_low

    def _feedforward_thrust(self, ud_eff):
        """Surge feed-forward. With ff_speed_scaling it follows the effective
        desired speed (drag ~ u^2), so the boat can actually slow down when the
        planner lowers the speed near waypoints or obstacles. Without it the
        constant base_speed dominated the surge command and, with tailwind, the
        boat stayed ~0.3-0.5 m/s too fast whatever the velocity gains were."""
        if not self.ff_speed_scaling or self.desired_speed <= 1e-3:
            return self.base_speed
        ratio = clamp(ud_eff / self.desired_speed, 0.0, 1.5)
        return max(self.ff_min_thrust, self.base_speed * ratio * ratio)

    def _slew_limit(self, previous, target):
        return previous + clamp(target - previous, -self.slew_rate_limit, self.slew_rate_limit)

    def send_thrust(self, left, right):
        if not self.mission_active or self.stop_override:
            left = right = 0.0
        self.pub_left.publish(Float64(data=float(left)))
        self.pub_right.publish(Float64(data=float(right)))

    def stop(self):
        self.send_thrust(0.0, 0.0)

    # ======================================================================
    # ANTI-STUCK (unchanged behaviour)
    # ======================================================================
    def check_stuck_condition(self):
        if self.escape_mode or not self.mission_active:
            return
        if self.last_position is None:
            self.last_position = (self.current_x, self.current_y)
            self.stuck_check_time = self.get_clock().now()
            return
        moved = math.hypot(self.current_x - self.last_position[0], self.current_y - self.last_position[1])
        elapsed = (self.get_clock().now() - self.stuck_check_time).nanoseconds / 1e9
        if elapsed >= self.stuck_timeout:
            if moved < self.stuck_threshold:
                if not self.is_stuck:
                    self.is_stuck = True
                    self.escape_mode = True
                    self.consecutive_stuck_count += 1
                    self.get_logger().warn(f"STUCK - escape attempt {self.consecutive_stuck_count}")
                    if self.consecutive_stuck_count >= 3:
                        self.get_logger().error("Requesting waypoint skip")
                        self.pub_skip_request.publish(String(data=json.dumps({'type': 'skip', 'reason': 'stuck'})))
                        self.consecutive_stuck_count = 0
                        self.is_stuck = False
                        self.escape_mode = False
            else:
                if self.is_stuck:
                    self.get_logger().info("Unstuck")
                self.is_stuck = False
                self.escape_mode = False
                if moved > self.stuck_threshold * 2.0:
                    self.consecutive_stuck_count = 0
            self.last_position = (self.current_x, self.current_y)
            self.stuck_check_time = self.get_clock().now()

    def execute_smart_escape(self):
        if not self.mission_active:
            self.escape_mode = False
            self.stop()
            return
        if self.front_clear > self.min_safe_distance:
            self.get_logger().info("Path clear - escape complete")
            self.escape_mode = False
            self.is_stuck = False
            self.heading_controller.reset()
            self.velocity_controller.reset()
        else:
            p = 450.0
            if self.right_clear > self.left_clear:
                self.send_thrust(p, -p)     # clockwise
            else:
                self.send_thrust(-p, p)     # counter-clockwise

    # ======================================================================
    # MAIN CONTROL LOOP
    # ======================================================================
    def control_loop(self):
        if self.stop_override or not self.mission_active or self.target_x is None or self.target_y is None:
            self.stop()
            self.prev_r = None
            return

        if self.last_position is None:
            self.last_position = (self.current_x, self.current_y)
            self.stuck_check_time = self.get_clock().now()
        if not self.escape_mode:
            self.check_stuck_condition()
        if self.escape_mode:
            self.execute_smart_escape()
            self.prev_r = None
            return

        r = self._estimate_yaw_rate()
        if self.prev_r is not None and self.nomoto_online_id:
            self.nomoto.update(self.prev_r, self.prev_tau_applied, r)

        self._handle_target_change()
        if (math.hypot(self.target_x - self.current_x, self.target_y - self.current_y) < 0.5
                and self.distance_to_target < 2.0):
            self.stop()
            self.prev_r = None
            self.get_logger().info("Reached target waypoint", throttle_duration_sec=2.0)
            return

        # ---------------- guidance, observer, error ----------------
        psi_d, y_e = self.compute_desired_heading()
        self._update_disturbance_observer()
        if self.drift_compensation_gain > 0.0:
            crab = math.atan2(self._drift_lateral_left(), max(self.current_speed, 0.3))
            psi_d = wrap_angle(psi_d - self.drift_compensation_gain * crab)
        e = wrap_angle(psi_d - self.current_yaw)
        e_dot = -r                                   # derivative on measurement
        u = self.current_speed
        ud = self._scheduled_speed()
        self.cross_track_error = y_e
        self.cte_path = self.calculate_path_cte()

        # ---------------- speed reference ----------------
        factor = 1.0
        if math.isfinite(self.distance_to_target) and self.distance_to_target < self.approach_slow_distance:
            frac = clamp(self.distance_to_target / self.approach_slow_distance, 0.0, 1.0)
            factor = min(factor, self.approach_slow_factor + (1.0 - self.approach_slow_factor) * frac)
        if self.obstacle_detected and self.urgency > 0:
            factor = min(factor, 1.0 - self.urgency * (1.0 - self.obstacle_slow_factor))
        ud_eff = ud * factor
        speed_error = ud_eff - u
        tau_u_lo = 50.0 if factor < 1.0 else 100.0
        if self.surge_brake_thrust > 0.0 and speed_error < -self.surge_brake_margin:
            tau_u_lo = -self.surge_brake_thrust     # overspeed: allow reverse thrust to brake
        tau_ff = self._feedforward_thrust(ud_eff)

        # ---------------- RL proposal (rate-limited, held) ----------------
        s = self._build_state(e, e_dot, y_e, ud_eff)
        rl_active = self.rl_mode != 'off' and not self.disturbance_gate
        if rl_active:
            if self.decision_counter % self.hold_steps == 0:
                raw, _ = self.agent.act(s)
                self.theta_cand_raw = raw[:len(PARAM_KEYS)]
                if self.adapt_v:
                    self.theta_v_cand_raw = raw[len(PARAM_KEYS):]
            self.decision_counter += 1
            step = np.clip(self.theta_cand_raw - self.theta_applied_vec, -self.max_delta, self.max_delta)
            self.theta_candidate_vec = np.clip(self.theta_applied_vec + step, self.prop_lo, self.prop_hi)
            if self.adapt_v:
                step_v = np.clip(self.theta_v_cand_raw - self.theta_v_applied_vec,
                                 -self.max_delta_v, self.max_delta_v)
                self.theta_v_candidate_vec = np.clip(self.theta_v_applied_vec + step_v,
                                                     self.prop_lo_v, self.prop_hi_v)
            else:
                self.theta_v_candidate_vec = self.theta_v_nom_vec.copy()
        else:
            self.theta_candidate_vec = self.theta_nom_vec.copy()
            self.theta_v_candidate_vec = self.theta_v_nom_vec.copy()

        # ---------------- parameter-space safety shield ----------------
        K_n, T_n, nomoto_ok = self.nomoto.model()
        if self.shield_enabled and rl_active and not self.shield_predictive:
            theta_vec, theta_v_vec = project_to_C(self.theta_candidate_vec), project_to_C_v(self.theta_v_candidate_vec)
            alpha, intervened, recovery_mode = 1.0, False, False
            alpha_v, intervened_v, recovery_mode_v = 1.0, False, False
        elif self.shield_enabled and rl_active:
            theta_vec, alpha, intervened, recovery_mode, _ = self.shield.filter(
                self.theta_candidate_vec, self.theta_nom_vec, self.heading_controller,
                self._make_actuator_model, e, r, u, ud, K_n, T_n, self.dt)
            theta_v_vec, alpha_v, intervened_v, recovery_mode_v, _ = self.shield_v.filter(
                self.theta_v_candidate_vec, self.theta_v_nom_vec, self.velocity_controller,
                u, ud_eff, self.prev_left_thrust + self.prev_right_thrust,
                lambda cmd: clamp(tau_ff + cmd, tau_u_lo, SAFE_THRUST),
                self.slew_rate_limit, self.dt)
        else:
            theta_vec, theta_v_vec = self.theta_candidate_vec, self.theta_v_candidate_vec
            alpha, intervened, recovery_mode = 1.0, False, False
            alpha_v, intervened_v, recovery_mode_v = 1.0, False, False
        phi_h = self.safety_index.heading(e)
        phi_v = self.safety_index.speed(u, ud_eff)
        phi = max(phi_h, phi_v)
        self.theta_applied_vec = np.asarray(theta_vec, dtype=float)
        self.theta_v_applied_vec = np.asarray(theta_v_vec, dtype=float)
        self.heading_controller.set_theta(vec_to_theta(self.theta_applied_vec))
        self.velocity_controller.set_theta(vec_to_theta(self.theta_v_applied_vec))

        # ---------------- heading control ----------------
        steering = self.heading_controller.compute_control(e, e_dot, self.dt)
        tau_r = self.steering_to_torque(steering, e)

        # ---------------- velocity control (single evaluation) ----------------
        vel_ctrl = self.velocity_controller.compute_control(speed_error, self.dt)
        tau_u = clamp(tau_ff + vel_ctrl, tau_u_lo, SAFE_THRUST)
        self.last_tau_u = tau_u

        # ---------------- thrust allocation ----------------
        tau_r = self._limit_for_surge(tau_r, tau_u, e)
        left = clamp(tau_u - tau_r, -SAFE_THRUST, SAFE_THRUST)
        right = clamp(tau_u + tau_r, -SAFE_THRUST, SAFE_THRUST)
        left = self._slew_limit(self.prev_left_thrust, left)
        right = self._slew_limit(self.prev_right_thrust, right)
        self.prev_left_thrust, self.prev_right_thrust = left, right
        self.prev_r = r
        self.prev_tau_applied = 0.5 * (right - left)
        self.send_thrust(left, right)

        # ---------------- cost, learning bookkeeping ----------------
        cost = (min((e / self.cost_e_scale) ** 2, self.cost_clip)
                + self.cost_w_tau * min((tau_r / self.cost_tau_scale) ** 2, self.cost_clip)
                + self.cost_w_cte * (y_e / self.cost_cte_scale) ** 2
                + self.cost_w_speed * (speed_error / self.cost_speed_scale) ** 2
                + self.cost_w_shield * (1.0 if (intervened or intervened_v) else 0.0))
        t_since_ref = ((self.get_clock().now() - self.ref_change_time).nanoseconds / 1e9
                       if self.ref_change_time is not None else float('inf'))
        t_since_speed = ((self.get_clock().now() - self.speed_change_time).nanoseconds / 1e9
                         if self.speed_change_time is not None else float('inf'))
        in_turn = t_since_ref < self.cost_skip
        in_transient = in_turn or t_since_speed < self.cost_skip_speed
        # turn-aware cost: overshoot/ringing after the first zero crossing, and thruster saturation
        self.turn_cost_last = 0.0
        if (self.cost_w_turn > 0.0 and rl_active and in_turn
                and not self.obstacle_detected and not self.escape_mode):
            if self.turn_sign0 is None and abs(e) > math.radians(1.0):
                self.turn_sign0 = 1.0 if e > 0 else -1.0
            if self.turn_sign0 is not None and not self.turn_crossed and e * self.turn_sign0 < 0.0:
                self.turn_crossed = True
            e_post = abs(e) if self.turn_crossed else 0.0
            at_limit = 1.0 if (abs(left) >= SAFE_THRUST - 1.0 or abs(right) >= SAFE_THRUST - 1.0) else 0.0
            self.turn_cost_last = (min((e_post / self.cost_turn_scale) ** 2, self.cost_clip)
                                   + self.cost_turn_w_sat * at_limit)
            self.agent.record_turn(self.turn_cost_last)
        # Episodes are scored on settled path following only, so that every episode covers the
        # same regime: no waypoint-switch transient, no slow-down before a waypoint, no obstacle
        # avoidance, and the vehicle actually under way (not stuck or escaping).
        learn_step = (rl_active and not in_transient and factor >= 0.999
                      and not self.obstacle_detected and not self.escape_mode
                      and u > 0.5 * max(ud_eff, 0.1))
        if learn_step:
            self.agent.record(cost)
            # with the turn-aware cost, an episode also has to contain a turn, so that every
            # return is scored on both regimes. Long transects (80-120 s without a switch) must not
            # end an episode without a turn, so the safety cap is cost_turn_max_steps (150 s).
            turn_ok = (self.cost_w_turn <= 0.0 or self.agent.turn_steps >= self.turn_min_steps
                       or self.agent.episode_steps >= self.turn_max_steps)
            if self.agent.episode_steps >= self.episode_length and turn_ok:
                R, updated = self.agent.end_episode()
                self.get_logger().info(
                    f"Episode {self.agent.episode_count} done | return={R:.4f} | "
                    f"ES iter={self.agent.iteration} | turn steps={self.agent.last_turn_steps} | shield interventions heading={self.shield.total_interventions} "
                    f"velocity={self.shield_v.total_interventions}")
                if updated:
                    self.save_policy()

        # ---------------- Lyapunov monitor ----------------
        recent_ref_change = t_since_ref < 1.0
        d_norm = float(np.hypot(*self.drift_kalman.get_drift()))
        V, V_dot, lyap_violation = self.lyap.update(e, self.heading_controller.sigma, d_norm,
                                                    self.dt, skip=recent_ref_change)
        V_v, _, lyap_v_violation = self.lyap_v.update(speed_error, self.velocity_controller.sliding_surface,
                                                      d_norm, self.dt, skip=recent_ref_change)

        # ---------------- console ----------------
        self.get_logger().info(
            f"[{self.rl_mode}|{'S' if self.shield_enabled else '-'}] ep{self.agent.episode_count} "
            f"e={math.degrees(e):.1f}° r={r:.2f} r_imu={self.yaw_rate:.2f} y_e={y_e:.1f}m phi={phi:.2f} "
            f"alpha={alpha:.2f}/{alpha_v:.2f} tau_r={tau_r:.0f} tau_u={tau_u:.0f} u={u:.2f}/{ud_eff:.2f} "
            f"gate={'Y' if self.disturbance_gate else 'N'}",
            throttle_duration_sec=0.5)

        # ---------------- logging ----------------
        if self.logging_enabled:
            Kp, Kd, Ki = self.heading_controller.get_gains()
            vKp, vKd, vKi = self.velocity_controller.get_gains()
            dx, dy = self.drift_kalman.get_drift()
            row = {
                'time_elapsed': round((self.get_clock().now() - self.start_time).nanoseconds / 1e9, 3),
                'episode': self.agent.episode_count, 'step': self.agent.episode_steps,
                'es_iteration': self.agent.iteration, 'rl_mode': self.rl_mode,
                'shield_enabled': int(self.shield_enabled),
                'heading_error_deg': round(math.degrees(e), 3), 'yaw_rate': round(r, 4), 'yaw_rate_imu': round(self.yaw_rate, 4),
                'cross_track_error': round(y_e, 3), 'cte_path': round(self.cte_path, 3),
                'velocity_error': round(speed_error, 3), 'current_speed': round(u, 3),
                'desired_speed_eff': round(ud_eff, 3), 'step_cost': round(cost, 5),
                'turn_cost': round(self.turn_cost_last, 5),
                'ref_transient': int(in_transient), 'learn_step': int(learn_step),
                'episode_return_last': (round(self.agent.last_return, 5)
                                        if self.agent.last_return is not None else ''),
                'phi': round(phi, 4), 'phi_heading': round(phi_h, 4), 'phi_speed': round(phi_v, 4),
                'intervention': int(intervened), 'shield_alpha': alpha,
                'recovery_mode': int(recovery_mode),
                'intervention_v': int(intervened_v), 'shield_alpha_v': alpha_v,
                'recovery_mode_v': int(recovery_mode_v),
                'disturbance_gate': int(self.disturbance_gate),
                'drift_x': round(dx, 4), 'drift_y': round(dy, 4),
                'd_tilde_proxy': round(self.d_tilde_proxy, 4),
                'nomoto_K': K_n, 'nomoto_T': round(T_n, 3), 'nomoto_identified': int(nomoto_ok),
                'yaw_dist_hat': round(self.nomoto.yaw_disturbance(), 5),
                'lyap_V': round(V, 5), 'lyap_Vdot': round(V_dot, 5),
                'lyap_violation': int(lyap_violation),
                'lyap_v_V': round(V_v, 5), 'lyap_v_violation': int(lyap_v_violation),
                'heading_K_p': round(Kp, 3), 'heading_K_d': round(Kd, 3), 'heading_K_i': round(Ki, 4),
                'velocity_K_p': round(vKp, 3), 'velocity_K_d': round(vKd, 3), 'velocity_K_i': round(vKi, 4),
                'tau_r': round(tau_r, 1), 'tau_u': round(tau_u, 1), 'tau_ff': round(tau_ff, 1),
                'left_thrust': round(left, 1), 'right_thrust': round(right, 1),
                'distance_to_target': round(self.distance_to_target, 2),
                'obstacle_detected': int(bool(self.obstacle_detected)),
                'x': round(self.current_x, 2), 'y': round(self.current_y, 2),
                'yaw_deg': round(math.degrees(self.current_yaw), 2),
                'target_x': round(self.target_x, 2), 'target_y': round(self.target_y, 2),
            }
            for i, k in enumerate(PARAM_ASCII):
                row[f'cand_{k}'] = round(float(self.theta_candidate_vec[i]), 5)
                row[k] = round(float(self.theta_applied_vec[i]), 5)
            for i, k in enumerate(PARAM_ASCII_V):
                row[f'cand_{k}'] = round(float(self.theta_v_candidate_vec[i]), 5)
                row[k] = round(float(self.theta_v_applied_vec[i]), 5)
            self.log_rows.append(row)
            if len(self.log_rows) >= 1000:
                self.flush_log_to_file()

        # ---------------- publish ----------------
        self.pub_status.publish(String(data=json.dumps({
            'mode': f'SAFE_RL_{self.rl_mode.upper()}',
            'shield_enabled': self.shield_enabled,
            'heading_error_deg': round(math.degrees(e), 1),
            'cross_track_error': round(y_e, 2),
            'current_speed': round(u, 2),
            'desired_speed': round(ud_eff, 2),
            'left_thrust': round(left, 1), 'right_thrust': round(right, 1),
            'distance_to_target': round(self.distance_to_target, 1),
            'tau_r': round(tau_r, 1), 'phi': round(phi, 3),
            'shield_alpha': alpha, 'shield_alpha_v': alpha_v, 'disturbance_gate': self.disturbance_gate,
        })))
        self.pub_rl_info.publish(String(data=json.dumps({
            'episode': self.agent.episode_count, 'step': self.agent.episode_steps,
            'es_iteration': self.agent.iteration,
            'last_return': self.agent.last_return,
            'theta_applied': {k: round(float(v), 4) for k, v in zip(PARAM_ASCII, self.theta_applied_vec)},
            'theta_v_applied': {k: round(float(v), 4) for k, v in zip(PARAM_ASCII_V, self.theta_v_applied_vec)},
            'shield_interventions': self.shield.total_interventions,
            'shield_v_interventions': self.shield_v.total_interventions,
            'recovery_mode_total': self.shield.total_recovery_mode,
            'lyap_violation_rate': (self.lyap.violations / self.lyap.checks) if self.lyap.checks else 0.0,
            'nomoto': {'K': K_n, 'T': T_n, 'identified': nomoto_ok},
        })))

    # ======================================================================
    # PERSISTENCE
    # ======================================================================
    def save_policy(self):
        try:
            os.makedirs(os.path.dirname(self.policy_file), exist_ok=True)
            with open(self.policy_file, 'w') as f:
                json.dump(self.agent.to_dict(), f, indent=2)
            self.get_logger().info(f"Policy saved (ES iteration {self.agent.iteration})")
        except OSError as ex:
            self.get_logger().warn(f"Could not save policy: {ex}")

    def flush_log_to_file(self):
        if not self.log_rows:
            return
        with open(self.log_file, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            w.writerows(self.log_rows)
        self.get_logger().info(f"Saved {len(self.log_rows)} rows to {self.log_file}")
        self.log_rows = []


# =============================================================================
# MAIN
# =============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = BuranController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down BURAN controller...")
    finally:
        node.flush_log_to_file()
        if node.rl_mode == 'es':
            node.save_policy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
