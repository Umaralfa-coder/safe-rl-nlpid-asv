# Safe-RL-NLPID: safety-shielded online tuning of robust ASV control

Code for the paper

> **Safety-Shielded Reinforcement Learning-Based Robust Control for Autonomous Surface Vehicles in Water-Quality Monitoring**
> Umar Alfa, Auwal shehu Tijjani, Raymond Houe Ngouna, Luc Fabresse and Eric Duviella.
> Center for Digital Systems, IMT Nord Europe, Univ. Lille; LGP-ENIT, Tarbes. *Manuscript in preparation.*

An evolution-strategies (ES) policy tunes the nonlinear parameters of a heading and a surge
nonlinear PID (NLPID) online. Every proposal passes a **parameter-space shield**:

1. a projection onto a box of gains, C_θ, on which the heading loop satisfies a
   conditional Lyapunov decrease for any gain sequence (checked by vertex LMIs, also at 20 Hz
   with the implemented yaw-rate filter);
2. a predictive line-search filter that rolls an online-identified Nomoto model forward
   with the exact control law and the thrust limits, and pulls a proposal back toward the
   certified nominal gains if a barrier condition on the safety index would fail.

The learner changes gains, never thrust, so the vehicle is always driven by an NLPID from
the certified set. The certificate is **conditional**: it holds while the gains stay in C_θ,
the disturbance-estimation error stays below an assumed bound and the state stays in a set
whose forward invariance is not proven. It is not a safety guarantee; see the paper.

Everything runs in **ROS 2 Jazzy + Gazebo Harmonic + VRX** (WAM-V, `sydney_regatta` world).

## Main results (simulation)

| Experiment | Result |
|---|---|
| E1, shield ablation (3 seeds, 101 paired legs) | Unconstrained random gains: +36 % thrust per metre, ×2 yaw-torque activity, Lyapunov decrease violated on 1.2 % of steps. The same proposals through the shield: no detectable difference from nominal at the seed level. |
| E2b, turn-aware learning | 38 % lower cross-track error than the hand-tuned NLPID, same turn overshoot, 8 % less thrust, no saturation. |
| E2d, doubled wind and waves | The frozen learned policy loses its advantage; the disturbance monitor never fires. Resumed learning: 28 % lower heading error than hand-tuned, 2.5× less torque activity, first state-dependent gains (extra damping in some states), but not the cross-track gain (not in the cost) and +11 % thrust. |
| E3, water-quality map | Raw 5-s probe: the lag dominates, all controllers equal. Lag-corrected probe: the shield prevents a 25 % loss of map accuracy, and the turn-aware controller improves it by 18 %. |

All results are from simulation, partly from single runs; see the limitations in the paper.

## Repository layout

```
ros2_ws/src/safe_rl_nlpid/      ROS 2 package (ament_python)
  safe_rl_nlpid/buran_controller.py   controller node: NLPID, shield, ES agent, observers, logging
  safe_rl_nlpid/wq_sim_node.py        water-quality probe simulator + online GP map
analysis/
  legs.py, compare_runs.py      leg-paired comparison of runs (Tables III, IV)
  wq_replay.py                  probe + GP map replay on logged trajectories (raw / no lag / lag-corrected)
  gains.py                      applied parameters: constant or state-dependent?
  learning_curve.py             episode returns from a saved policy file
  certificate/                  vertex LMI (continuous time) and 20 Hz / delay checks
  figures/                      turn-in + per-leg figure
scripts/set_weather.sh          switch the VRX world between normal and doubled wind/waves
docs/RUN_EXPERIMENTS.md         every command for every experiment
results/                        per-leg and map-error tables behind the paper's numbers
tests/                          offline tests (ROS stubs, no simulator needed)
```

## Installation

Requirements: Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic and
[VRX](https://github.com/osrf/vrx) built in a colcon workspace, plus a mission planner (see
*Interfaces*).

```bash
cd ~/seal_ws/src
git clone https://github.com/<your-account>/safe-rl-nlpid-asv.git   # colcon finds the package inside
cd ~/seal_ws && colcon build --merge-install --packages-select safe_rl_nlpid
source install/setup.bash
ros2 pkg executables safe_rl_nlpid       # -> buran_controller, wq_sim
pip install -r safe-rl-nlpid-asv/requirements.txt     # analysis tools
```

Offline checks (no simulator): `python3 -m pytest tests/`.

## Quick start

```bash
# terminal 1
ros2 launch vrx_gz competition.launch.py world:=sydney_regatta
# terminal 2: hand-tuned NLPID (no learning)
ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=A -p rl_seed:=7
# terminal 3: start your planner and the mission (docs/RUN_EXPERIMENTS.md)
```

Experiment presets (`-p experiment:=...`): `A` nominal; `BU`/`CU` unconstrained random gains
without/with shield; `ES` learning with shield; `AM`/`ESM` mistuned controller without/with
learning. Learning, evaluation, turn-aware cost (`cost_w_turn`), fixed-gain learner
(`es_bias_only`) and the other options are parameters; see `docs/RUN_EXPERIMENTS.md`.

## Interfaces

The controller follows waypoints given by an external planner (we used the `sputnik` planner
of our team's VRX stack). Any planner works if it publishes these topics:

| Topic | Type | Direction | Content |
|---|---|---|---|
| `/wamv/sensors/gps/gps/fix` | `sensor_msgs/NavSatFix` | in | GPS |
| `/wamv/sensors/imu/imu/data` | `sensor_msgs/Imu` | in | orientation (heading), gyro |
| `/planning/current_target` | `std_msgs/String` (JSON) | in | `current_position`, `target_waypoint`, `distance_to_target` |
| `/planning/waypoints` | `std_msgs/String` (JSON) | in | `waypoints: [[x, y], ...]` |
| `/planning/mission_status` | `std_msgs/String` (JSON) | in | `state` (`"DRIVING"` = active) |
| `/perception/obstacle_info` | `std_msgs/String` (JSON) | in | `obstacle_detected`, `min_distance`, ... (optional) |
| `/wamv/thrusters/{left,right}/thrust` | `std_msgs/Float64` | out | thrust commands (N), limit ±800 |
| `/control/status`, `/control/rl_info` | `std_msgs/String` (JSON) | out | controller and learner state |
| `/planning/skip_waypoint` | `std_msgs/String` (JSON) | out | request to skip a waypoint when stuck |

Logs: one CSV per controller start (about 90 columns: errors, gains, safety index, shield,
Lyapunov monitor, thrusts, positions) and the policy JSON (parameters + every episode return)
in `$SAFE_RL_LOG_DIR` (default `~/seal_ws/logs`).

## Citation

```bibtex
@unpublished{alfa2026saferl,
  author = {Alfa, Umar and Tijjani, Auwal shehu and Houe Ngouna, Raymond and Fabresse, Luc and Duviella, Eric},
  title  = {Safety-Shielded Reinforcement Learning-Based Robust Control for Autonomous Surface Vehicles in Water-Quality Monitoring},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```

## License

MIT (see `LICENSE`). VRX and Gazebo are separate projects under their own licenses; this
repository contains no VRX code, only a script that edits the wind and wave values of the
installed `sydney_regatta` world.
