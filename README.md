# E-puck2 Obstacle Avoidance Simulation

Simple Braitenberg-style obstacle avoidance for the e-puck2 in Webots.

## Quick Start

```bash
# Open in Webots
webots /home/jack/swarmlab/projects/epuck2-sim/worlds/epuck2_obstacle_avoidance.wbt
```

Or: Webots → File → Open World → select the `.wbt` file.

## What It Does

The e-puck2 navigates a 1m×1m arena with 6 wooden box obstacles.
Uses a **Braitenberg vehicle** approach — direct sensor-motor coupling
with no explicit planning. 8 IR proximity sensors drive differential
wheel speeds through a weight matrix.

## Files

Obstacle avoidance demo:
- `worlds/epuck2_obstacle_avoidance.wbt` — single-robot Webots world file
- `controllers/epuck2_obstacle_avoidance/epuck2_obstacle_avoidance.py` — single-robot Python controller

Basic 10-robot swarm experiment:
- `worlds/epuck2_swarm_basic_10.wbt` — 10 e-puck2 robots in a larger arena
- `controllers/epuck2_swarm_basic/epuck2_swarm_basic.py` — shared multi-robot controller

Dreamer-inspired 4-robot virtual-body experiment:
- `worlds/epuck2_dreamer_team_4.wbt` — four e-puck2 robots arranged as one cooperative body
- `controllers/epuck2_dreamer_team_supervisor/epuck2_dreamer_team_supervisor.py` — centralized team planner / slot controller
- `controllers/epuck2_dreamer_team_member/epuck2_dreamer_team_member.py` — per-robot wheel execution with local avoidance
- `controllers/dreamer_team_common.py` — testable formation and tracking math shared by both controllers

## Basic 10-Robot Swarm Experiment

Open in Webots:

```bash
webots /home/jack/swarmlab/projects/epuck2-sim/worlds/epuck2_swarm_basic_10.wbt
```

What it does:
- Spawns 10 e-puck2 robots (`version "2"`)
- Uses a simple shared controller for all robots
- Drives forward by default
- Uses proximity sensors for reactive obstacle avoidance
- Adds a small per-robot asymmetry so motion is less synchronized/clumped

## Dreamer-Inspired 4-Robot Virtual Body

Open in Webots:

```bash
webots /home/jack/swarmlab/projects/epuck2-sim/worlds/epuck2_dreamer_team_4.wbt
```

What it does:
- Spawns four e-puck2 robots in a compact 2×2 formation
- Uses a centralized supervisor as the team-level planner
- Assigns each robot a persistent slot (`front_left`, `front_right`, `rear_left`, `rear_right`)
- Computes wheel-speed commands so the four robots move like one larger virtual robot body
- Keeps local proximity-based avoidance active so members can still react to walls or boxes
- Cycles through arena waypoints to demonstrate coordinated group locomotion

### Dreamer-compatible interface scaffold

Files:
- `src/dreamer_env.py` — pure Python Dreamer-facing environment/state-transition scaffold
- `src/dreamer_gym.py` — Gymnasium-style wrapper with task randomization and episode truncation
- `src/dreamer_webots_live.py` — live Webots file-bridge wrapper and snapshot conversion layer
- `tests/test_dreamer_env.py` — unit tests for observation shape, reward, and step behavior
- `tests/test_dreamer_gym.py` — unit tests for reset/step tuple shape and randomized waypoint resets
- `tests/test_dreamer_webots_live.py` — unit tests for the live bridge wrapper using a fake bridge
- `scripts/check_dreamer_gym.py` — quick rollout sanity-check script
- `scripts/run_dreamer_webots_live.py` — end-to-end live rollout script that launches Webots and drives the bridge
- `scripts/launch_webots_headless.sh` — dedicated headless Webots launcher used by the live bridge when no display is available

Interface design:
- Action = `(forward_cmd, turn_cmd)`
  - both normalized to `[-1, 1]`
  - `forward_cmd` controls team translation speed
  - `turn_cmd` controls team yaw rate
- Observation = 19 floats
  - team centroid `(x, y)`
  - waypoint delta `(dx, dy)`
  - team heading
  - previous action `(forward_cmd, turn_cmd)`
  - for each of 4 robots: relative slot position `(x_rel, y_rel)` and heading error
- Reward includes:
  - progress toward waypoint
  - heading alignment bonus
  - small alive bonus
  - action penalty
  - reach bonus near target
- Reset randomization:
  - samples a waypoint inside the arena by default
  - supports deterministic seeding
  - supports explicit waypoint override via reset options

Live Webots bridge:
- The supervisor now supports `DREAMER_TEAM_MODE=bridge`
- In bridge mode it reads:
  - `action.json`
  - `reset.json`
- And writes:
  - `state.json`
- Bridge directory is controlled by `DREAMER_TEAM_BRIDGE_DIR`
- Python side uses:
  - `BridgePaths`
  - `FileBridgeClient`
  - `DreamerTeamLiveWebotsEnv`
- The bridge is sequence-synchronized, so each env reset/step waits for the matching supervisor snapshot instead of re-reading stale state
- In headless sessions without `DISPLAY`, the launcher automatically falls back to a dedicated `Xvfb`-backed wrapper script and runs Webots with `--batch --no-rendering --mode=realtime`
- The launcher also picks a free Webots TCP port automatically to avoid 1234 collisions with other simulator instances
- The client now launches Webots in its own process group and tears down the full group on `close()` so repeated runs don't leave orphaned controllers or port conflicts

Quick checks:

```bash
cd /home/jack/swarmlab/projects/epuck2-sim
python3 -m unittest tests.test_dreamer_webots_live tests.test_dreamer_gym tests.test_dreamer_env tests.test_dreamer_team_logic -v
python3 scripts/check_dreamer_gym.py
python3 scripts/train_team_cem.py --env abstract --iterations 2 --population 4 --eval-episodes 1 --episode-steps 20 --policy-kind mlp --hidden-sizes 32,32 --num-workers 2 --run-name smoke-parallel-mlp
```

Manual live-bridge launch example:

```bash
cd /home/jack/swarmlab/projects/epuck2-sim
DREAMER_TEAM_MODE=bridge \
DREAMER_TEAM_BRIDGE_DIR=/tmp/dreamer_team_bridge \
webots /home/jack/swarmlab/projects/epuck2-sim/worlds/epuck2_dreamer_team_4.wbt
```

Headless end-to-end rollout example:

```bash
cd /home/jack/swarmlab/projects/epuck2-sim
python3 scripts/run_dreamer_webots_live.py --steps 12 --launch-mode headless
```

That script launches Webots automatically, runs a short scripted rollout through `DreamerTeamLiveWebotsEnv`, prints JSON results, and cleans up the simulator process group when it exits.

Parallel CEM trainer:
- `scripts/train_team_cem.py` now supports:
  - larger MLP policies (`--policy-kind mlp --hidden-sizes 128,64` etc.)
  - parallel candidate evaluation across multiple worker processes (`--num-workers`)
  - isolated live Webots worker bridges so limited live parallelism remains stable
- Recommended high-throughput abstract CPU run on this PC:

```bash
cd /home/jack/swarmlab/projects/epuck2-sim
python3 scripts/train_team_cem.py \
  --env abstract \
  --iterations 250 \
  --population 96 \
  --eval-episodes 2 \
  --episode-steps 100 \
  --policy-kind mlp \
  --hidden-sizes 128,64 \
  --init-sigma 0.28 \
  --min-sigma 0.02 \
  --num-workers 14 \
  --run-name cem-abstract-mlp-parallel-v1
```

GPU abstract trainer:
- `src/dreamer_torch.py` provides batched PyTorch dynamics, reward, and observation code for the abstract environment
- `scripts/train_team_cem_torch.py` runs a GPU-backed batched CEM loop on the abstract task
- Environment setup used here:

```bash
cd /home/jack/swarmlab/projects/epuck2-sim
/usr/bin/python3 -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Example GPU run:

```bash
cd /home/jack/swarmlab/projects/epuck2-sim
. .venv-gpu/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_team_cem_torch.py \
  --iterations 300 \
  --population 64 \
  --eval-episodes 4 \
  --episode-steps 120 \
  --hidden-sizes 128,64 \
  --init-sigma 0.20 \
  --min-sigma 0.012 \
  --run-name torch-cem-gpu-v1-small
```

Note: on this machine, other GPU workloads (for example Ollama) can occupy most VRAM, so the safe batch size may need to be reduced unless that workload is stopped.

## Controller Details

- **Algorithm:** Braitenberg obstacle avoidance
- **Sensors:** 8 IR proximity (ps0-ps7), range 0-4096
- **Actuators:** 2 wheel motors, 10 LEDs
- **Max speed:** 6.28 rad/s
- **Time step:** 16ms
- **Logging:** Prints front proximity + wheel speeds every 2s
