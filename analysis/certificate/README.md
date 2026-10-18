# Certificate checks (Section IV of the paper)

| Script | What it checks |
|---|---|
| `lmi_common_P.py` | vertex matrices of the heading loop over C_theta and the box of states; search for a common P |
| `run_lmi.py` | continuous time: common P for the nominal model, over the identified (K, T) range, and with a torque fraction s in [s_min, 1] |
| `verify.py` | the rounded P of the paper at the 12 vertices, plus 200 000 random samples of the real nonlinear law (imports the controller with ROS stubs) |
| `dt_check.py`, `dt_lmi.py`, `dt_lmi2.py` | 20 Hz discrete-time loop with the implemented yaw-rate filter, with 0, 1 and 2 samples of actuation delay, for k_tau = 200 N (nominal) and 600 N (mistuned) |
| `dt_box.py`, `ct24.py` | box and 24-vertex variants |

Run from anywhere, e.g. `python3 analysis/certificate/dt_lmi2.py`. Expected output (paper):
common P at 20 Hz without delay (decay about 0.24 1/s); no common quadratic P with one sample of
delay; for k_tau = 600 N with one sample of delay a vertex is unstable (mode near 2 Hz).
