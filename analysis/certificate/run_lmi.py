import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import math, json, itertools, numpy as np
from lmi_common_P import *
e_max = math.radians(30); out = {}
def uniq(V): return list({A.tobytes(): A for A in V}.values())
print('--- nominal model, unsaturated, by yaw-rate-error bound')
for ed_max in (0.5, 1.0, 1.5, 2.0):
    V, kp, kd, ph = vertices(e_max, ed_max, (1.5e-3, 1.5e-3), (0.68, 0.68), 1.0); V = uniq(V); a, P = best_P(V)
    print(f'ed_max {ed_max}: k_p {kp[0]:.2f}-{kp[1]:.0f} k_d {kd[0]:.2f}-{kd[1]:.0f} phi 0-{ph[1]:.1f} | {len(V)} vertices | alpha {a:.3f} P {P.round(4).tolist()}')
    out[f'nominal|{ed_max}'] = dict(alpha=a, P=P.tolist(), kp=kp, kd=kd, phi=ph, n=len(V))
print('--- fixed (K, T) on a grid of the identified range: P may depend on (K, T), common over theta')
res = []
for K in np.linspace(1.27e-3, 1.92e-3, 5):
    for T in np.linspace(0.59, 0.91, 5):
        V, *_ = vertices(e_max, 1.0, (K, K), (T, T), 1.0); a, P = best_P(uniq(V)); res.append((K, T, a))
res = np.array(res); print(np.round(res[:, 2].reshape(5, 5), 3)); print('min alpha', res[:, 2].min(), 'at', res[res[:, 2].argmin()])
out['grid_identified'] = res.tolist()
resb = []
for K in np.linspace(1.27e-3, 2.2e-3, 5):
    for T in np.linspace(0.56, 1.32, 5):
        V, *_ = vertices(e_max, 1.0, (K, K), (T, T), 1.0); a, P = best_P(uniq(V)); resb.append((K, T, a))
resb = np.array(resb); print('online range'); print(np.round(resb[:, 2].reshape(5, 5), 3)); out['grid_online'] = resb.tolist()
print('--- nominal model with saturation factor s in [s_min, 1]')
for s_min in (0.8, 0.6, 0.5, 0.4, 0.3, 0.2):
    V, *_ = vertices(e_max, 1.0, (1.5e-3, 1.5e-3), (0.68, 0.68), s_min); a, P = best_P(uniq(V))
    print(f's_min {s_min}: alpha {a:.3f} P {P.round(3).tolist()}'); out[f'sat|{s_min}'] = dict(alpha=a, P=P.tolist())
json.dump(out, open(_os.path.join(_HERE, 'lmi_results.json'), 'w'), indent=1)
