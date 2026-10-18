# Running the experiments

Every experiment is one mission in the VRX `sydney_regatta` world, with three terminals:
Gazebo, the controller, and your mission planner. Each terminal starts with

```bash
source ~/seal_ws/install/setup.bash          # your workspace with VRX + this package
export GZ_IP=127.0.0.1                        # keeps Gazebo/ROS on this machine (avoids network drops)
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
```

Logs and policies go to `$SAFE_RL_LOG_DIR` (default `~/seal_ws/logs`).

> Lines marked **(reconstructed)** were rebuilt from the settings in the paper, not copied
> from a terminal. Check them against your shell history (`history | grep buran_controller`)
> before re-running.

## 1. One mission, step by step

**Terminal 1, Gazebo**
```bash
ros2 launch vrx_gz competition.launch.py world:=sydney_regatta
```

**Terminal 2, controller** (about 30 s later)
```bash
touch /tmp/run_start                         # marks the start, used to collect the log afterwards
ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=A -p rl_seed:=7
```

**Terminal 3, planner** (start your planner, then)
```bash
ros2 topic pub --once /sputnik/mission_command std_msgs/msg/String "{data: '{\"command\": \"generate_waypoints\"}'}"
ros2 topic pub --once /sputnik/mission_command std_msgs/msg/String "{data: '{\"command\": \"confirm_waypoints\"}'}"
ros2 topic pub --once /sputnik/mission_command std_msgs/msg/String "{data: '{\"command\": \"start_mission\"}'}"
```

**After the mission** (Ctrl+C in terminal 2; the controller saves the log and the policy)
```bash
NAME=A_s7
mkdir -p ~/seal_ws/logs/$NAME && find ~/seal_ws/logs -maxdepth 1 -name "controller_*.csv" \
  -newer /tmp/run_start -exec cp {} ~/seal_ws/logs/$NAME/ \; && ls -l ~/seal_ws/logs/$NAME
```

If the controller is restarted during a learning mission with `load_policy:=true`, learning
continues from the saved policy, but each restart writes a new CSV.

## 2. Status line

```
[eval|S] ep16 e=0.2° r=0.04 r_imu=1.07 y_e=2.4m phi=-0.30 alpha=1.00/1.00 tau_r=-50 tau_u=188 u=1.25/1.20 gate=N
```

| field | meaning |
|---|---|
| `[mode\|S]` | `off` (no learning, controller A), `random`, `es` (learning), `eval` (learned policy, no exploration); `S` = shield on |
| `ep` | episode number |
| `e` | heading error (deg) |
| `r` | yaw rate from the filtered heading derivative (rad/s), used by the controller |
| `r_imu` | raw gyro yaw rate, logged only (noisy in VRX) |
| `y_e` | cross-track error (m); reads 0 close to a waypoint |
| `phi` | safety index; below 0 is safe |
| `alpha` | shield factor heading/speed; 1 = proposal accepted, 0 = nominal gains |
| `tau_r`, `tau_u` | yaw-torque and surge-force commands |
| `u` | speed actual/desired (m/s) |
| `gate` | `Y` = disturbance monitor fired, learning frozen |

## 3. Experiments of the paper

All in normal weather (`scripts/set_weather.sh normal`) except E2d.

### E1: shield ablation (seeds 7, 8, 9)
```bash
# A: nominal gains, no learning
ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=A  -p rl_seed:=7
# BU: random proposals from the enlarged set, applied without shield     (reconstructed)
ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=BU -p rl_seed:=7
# CU: the same proposals through the full shield                          (reconstructed)
ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=CU -p rl_seed:=7
```
The `BU`/`CU` presets set the mode, the enlarged proposal set and the shield; BU and CU of one
seed receive identical proposals.

### E2a: straight-leg cost, large steps (seeds 7, 8, 9) (reconstructed)
```bash
ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=ES -p rl_seed:=7 \
  -p load_policy:=false -p es_sigma:=0.5 -p es_lr:=0.5 -p rl_action_scale_frac:=1.0 -p rl_rate_limit_frac:=0.25 \
  -p rl_adapt_velocity:=false -p rl_episode_length:=200 -p policy_file:=$HOME/seal_ws/logs/safe_rl_nlpid_policy_est_s7.json
# evaluation of the learned policy
ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=ES -p rl_seed:=7 \
  -p rl_mode:=eval -p load_policy:=true -p rl_action_scale_frac:=1.0 -p rl_rate_limit_frac:=0.25 \
  -p rl_adapt_velocity:=false -p policy_file:=$HOME/seal_ws/logs/safe_rl_nlpid_policy_est_s7.json
```
Fixed-gain learner: add `-p es_bias_only:=true` (W = 0) and use its own policy file.

### E2b: turn-aware cost (three learning missions, then evaluation)
```bash
# mission 1 (start from nominal); missions 2 and 3: rl_seed 17 and 27 with load_policy:=true
ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=ES -p rl_mode:=es -p rl_seed:=7 \
  -p load_policy:=false -p cost_w_turn:=5.0 -p es_n_pairs:=2 -p es_sigma:=0.5 -p es_lr:=0.5 \
  -p rl_action_scale_frac:=1.0 -p rl_rate_limit_frac:=0.25 -p rl_adapt_velocity:=false -p rl_episode_length:=100 \
  -p policy_file:=$HOME/seal_ws/logs/policy_turnL.json
# evaluation (L_t)
ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=ES -p rl_seed:=7 -p rl_mode:=eval \
  -p load_policy:=true -p cost_w_turn:=5.0 -p rl_action_scale_frac:=1.0 -p rl_rate_limit_frac:=0.25 \
  -p rl_adapt_velocity:=false -p rl_episode_length:=100 -p policy_file:=$HOME/seal_ws/logs/policy_turnL.json
```
Check after about 2 minutes that the learned policy is the one in use:
`python3 analysis/gains.py --run Lt=$HOME/seal_ws/logs` should show `mu_d` 0.10,
`boundary_layer` 2.0, `heading_K_p` about 20.

### E2c: from the mistuned controller (reconstructed)
```bash
ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=AM  -p rl_seed:=7   # no learning
ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=ESM -p rl_seed:=7   # ES from AM
```

### E2d: doubled wind and waves
```bash
scripts/set_weather.sh strong          # restart Gazebo afterwards
# S1: controller A
ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=A -p rl_seed:=7
# S2b: L_t frozen: the E2b evaluation command above
# R1: resume learning from L_t, on a copy of the policy
cp ~/seal_ws/logs/policy_turnL.json ~/seal_ws/logs/policy_turnL_strong.json
ros2 run safe_rl_nlpid buran_controller --ros-args -p use_sim_time:=true -p experiment:=ES -p rl_mode:=es -p rl_seed:=37 \
  -p load_policy:=true -p cost_w_turn:=5.0 -p es_n_pairs:=2 -p es_sigma:=0.5 -p es_lr:=0.5 \
  -p rl_action_scale_frac:=1.0 -p rl_rate_limit_frac:=0.25 -p rl_adapt_velocity:=false -p rl_episode_length:=100 \
  -p policy_file:=$HOME/seal_ws/logs/policy_turnL_strong.json
# R3: evaluation of the resumed policy (L_t+): the E2b evaluation command with policy_turnL_strong.json
scripts/set_weather.sh normal          # when done
```

### E3: water-quality sampling (offline, on the logged trajectories)
```bash
python3 analysis/wq_replay.py --run A=logs/A_s7 --run BU=logs/BU_s7 --run CU=logs/CU_s7 --fields 10
```
The probe simulator can also run live next to the controller: `ros2 run safe_rl_nlpid wq_sim`.

## 4. Analysis

```bash
pip install -r requirements.txt
# leg-paired comparison (Tables III and IV); NAME=p1,p2,p3 averages seeds per leg
python3 analysis/compare_runs.py --ref A --run A=logs/A_s7,logs/A_s8,logs/A_s9 --run Lt=logs/TURN_eval --out results/calm_legs.csv
python3 analysis/compare_runs.py --ref A --run A=logs/S1_strong_A --run Lt=logs/S2b_strong_LT --run Lt+=logs/R3_strong_eval
# applied gains (constant or state-dependent?)
python3 analysis/gains.py --run Lt=logs/S2b_strong_LT --run Lt+=logs/R3_strong_eval
# learning curve from a policy file
python3 analysis/learning_curve.py logs/policy_turnL_strong.json --split 53 --png results/curve.png
# certificate: vertex LMI (continuous time) and the 20 Hz / delay check
python3 analysis/certificate/verify.py
python3 analysis/certificate/dt_lmi2.py
```
