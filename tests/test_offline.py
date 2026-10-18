"""Offline tests: import the controller without ROS (stub packages in tests/stubs) and check
the pieces that the shield and the learner rely on.

  python -m pytest tests/
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'stubs'))
sys.path.insert(0, os.path.join(HERE, '..', 'ros2_ws', 'src', 'safe_rl_nlpid', 'safe_rl_nlpid'))

import buran_controller as bc  # noqa: E402


def test_projection_stays_in_certified_box():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        v = rng.normal(0, 5, len(bc.PARAM_LO))
        p = bc.project_to_C(v)
        assert np.all(p >= bc.PARAM_LO - 1e-12) and np.all(p <= bc.PARAM_HI + 1e-12)
    inside = (bc.PARAM_LO + bc.PARAM_HI) / 2
    assert np.allclose(bc.project_to_C(inside), inside)


def test_theta_vector_round_trip():
    v = (bc.PARAM_LO + bc.PARAM_HI) / 2
    assert np.allclose(bc.theta_to_vec(bc.vec_to_theta(v)), v)


def test_nonlinear_gain_is_capped():
    for x in (1e-6, 1e-3, 0.1, 1.0, 10.0):
        g = bc.nl_gain(4.0, x, 0.1, 0.01, 0.05, 20.0)
        assert 0.05 <= g <= 20.0


def test_es_agent_starts_at_nominal_and_saves():
    nom = (bc.PARAM_LO + bc.PARAM_HI) / 2
    ag = bc.ESLinearPolicyAgent(6, nom, bc.PARAM_LO, bc.PARAM_HI, bc.PARAM_KEYS, mode='eval', seed=1)
    theta, _ = ag.act(np.zeros(6))          # W = b = 0 at initialisation -> theta_nom
    assert np.allclose(theta, nom)
    d = ag.to_dict()
    assert 'params' in d and 'return_history' in d


def test_wq_probe_lag():
    sys.path.insert(0, os.path.join(HERE, '..', 'ros2_ws', 'src', 'safe_rl_nlpid', 'safe_rl_nlpid'))
    from wq_sim_node import ProbeSensor
    s = ProbeSensor(tau_s=5.0, sigma0=0.0, k_r=0.0, k_dF=0.0, seed=0)
    s.step(0.0, 0.05)              # the first reading initialises the probe state
    for _ in range(100):           # 5 s of a unit step at 20 Hz -> about 63 % of the step
        s.step(1.0, 0.05)
    assert 0.55 < s.c_m < 0.70
