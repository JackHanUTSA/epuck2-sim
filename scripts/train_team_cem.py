#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dreamer_gym import DreamerTeamGymEnv  # noqa: E402
from src.dreamer_webots_live import BridgePaths, DreamerTeamLiveWebotsEnv, FileBridgeClient  # noqa: E402


_WORKER_ENV = None
_WORKER_ENV_CONFIG: dict[str, Any] | None = None


@dataclass(frozen=True)
class PolicySpec:
    kind: str
    obs_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...] = ()

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        return (self.obs_dim, *self.hidden_sizes, self.action_dim)

    @property
    def param_dim(self) -> int:
        total = 0
        sizes = self.layer_sizes
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            total += in_dim * out_dim + out_dim
        return total

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "param_dim": self.param_dim,
        }


@dataclass
class MLPPolicy:
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        for layer_idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = weight @ x + bias
            x = np.tanh(x)
            if layer_idx == len(self.weights) - 1:
                x = np.clip(x, -1.0, 1.0)
        return x

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "weights": [weight.tolist() for weight in self.weights],
            "biases": [bias.tolist() for bias in self.biases],
        }


@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    final_distance: float


@dataclass
class IterationResult:
    iteration: int
    mean_return: float
    best_return: float
    elite_mean_return: float
    sigma: float
    best_final_distance: float
    elapsed_sec: float


@dataclass
class CandidateEvaluation:
    vector: np.ndarray
    result: EpisodeResult


@dataclass(frozen=True)
class CandidateTask:
    vector_list: list[float]
    seed_base: int
    eval_episodes: int
    policy_spec: dict[str, Any]


@dataclass(frozen=True)
class WorkerEnvConfig:
    env: str
    episode_steps: int
    snapshot_timeout: float
    world: str
    bridge_root: str | None
    launch_mode: str


@dataclass(frozen=True)
class WorkerInitConfig:
    env_config: WorkerEnvConfig
    worker_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a higher-capacity team controller with CEM.")
    parser.add_argument("--env", choices=["abstract", "live"], default="abstract")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--episode-steps", type=int, default=80)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--init-sigma", type=float, default=0.35)
    parser.add_argument("--min-sigma", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot-timeout", type=float, default=25.0)
    parser.add_argument("--world", type=Path, default=ROOT / "worlds" / "epuck2_dreamer_team_4.wbt")
    parser.add_argument("--bridge-dir", type=Path, default=None)
    parser.add_argument("--launch-mode", choices=["auto", "gui", "headless"], default="headless")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--init-policy",
        type=Path,
        default=None,
        help="Optional .npy vector to use as the initial CEM mean.",
    )
    parser.add_argument(
        "--policy-kind",
        choices=["linear", "mlp"],
        default="mlp",
        help="Policy architecture. MLP enables larger training matrices than the original linear controller.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="64,64",
        help="Comma-separated hidden sizes for MLP policy, e.g. 64,64 or 128,64. Ignored for linear.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel evaluation workers. 0 = auto (uses CPU count for abstract, 2 for live).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Safety cap for auto worker selection on this PC.",
    )
    return parser.parse_args()


def parse_hidden_sizes(text: str, policy_kind: str) -> tuple[int, ...]:
    if policy_kind == "linear":
        return ()
    text = text.strip()
    if not text:
        return ()
    values = tuple(int(chunk) for chunk in text.split(",") if chunk.strip())
    if any(value <= 0 for value in values):
        raise ValueError(f"hidden sizes must be positive: {values}")
    return values


def build_policy_spec(args: argparse.Namespace, *, obs_dim: int, action_dim: int) -> PolicySpec:
    return PolicySpec(
        kind=args.policy_kind,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes, args.policy_kind),
    )


def policy_from_flat(vector: np.ndarray, policy_spec: PolicySpec) -> MLPPolicy:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.shape != (policy_spec.param_dim,):
        raise ValueError(f"policy vector has shape {vector.shape}, expected {(policy_spec.param_dim,)}")
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    offset = 0
    for in_dim, out_dim in zip(policy_spec.layer_sizes[:-1], policy_spec.layer_sizes[1:]):
        weight_count = in_dim * out_dim
        weight = vector[offset: offset + weight_count].reshape(out_dim, in_dim)
        offset += weight_count
        bias = vector[offset: offset + out_dim]
        offset += out_dim
        weights.append(weight)
        biases.append(bias)
    return MLPPolicy(weights=weights, biases=biases)


def worker_env_config_from_args(args: argparse.Namespace, run_dir: Path) -> WorkerEnvConfig:
    bridge_root = None
    if args.env == "live":
        bridge_root = str((args.bridge_dir or (run_dir / "worker-bridges")).resolve())
    return WorkerEnvConfig(
        env=args.env,
        episode_steps=args.episode_steps,
        snapshot_timeout=args.snapshot_timeout,
        world=str(args.world.resolve()),
        bridge_root=bridge_root,
        launch_mode=args.launch_mode,
    )


def make_env_from_config(config: WorkerEnvConfig, worker_index: int = 0):
    if config.env == "abstract":
        return DreamerTeamGymEnv(max_steps=config.episode_steps)
    bridge_root = Path(config.bridge_root or (RUNS_DIR / "live-bridge"))
    bridge_dir = bridge_root / f"worker-{worker_index:02d}"
    bridge = FileBridgeClient(
        paths=BridgePaths.from_dir(bridge_dir),
        world_path=Path(config.world),
        mode=config.launch_mode,
    )
    return DreamerTeamLiveWebotsEnv(
        bridge=bridge,
        max_steps=config.episode_steps,
        snapshot_timeout=config.snapshot_timeout,
    )


def rollout(env, policy: MLPPolicy, *, seed: int) -> EpisodeResult:
    obs, _info = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float64)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    final_distance = float("nan")

    while True:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        obs = np.asarray(obs, dtype=np.float64)
        total_reward += float(reward)
        steps += 1
        final_distance = float(info.get("distance_to_waypoint", float("nan")))
        if terminated or truncated:
            break

    return EpisodeResult(
        total_reward=total_reward,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        final_distance=final_distance,
    )


def evaluate_candidate(env, policy: MLPPolicy, *, eval_episodes: int, seed_base: int) -> EpisodeResult:
    rewards = []
    steps = []
    final_distances = []
    terminations = 0
    truncations = 0
    for episode_idx in range(eval_episodes):
        result = rollout(env, policy, seed=seed_base + episode_idx)
        rewards.append(result.total_reward)
        steps.append(result.steps)
        final_distances.append(result.final_distance)
        terminations += int(result.terminated)
        truncations += int(result.truncated)
    return EpisodeResult(
        total_reward=float(np.mean(rewards)),
        steps=int(np.mean(steps)),
        terminated=terminations > 0,
        truncated=truncations > 0,
        final_distance=float(np.mean(final_distances)),
    )


def ensure_run_dir(run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = run_name or f"cem-{timestamp}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2))


def _worker_initializer(init_config: WorkerInitConfig) -> None:
    global _WORKER_ENV, _WORKER_ENV_CONFIG
    _WORKER_ENV_CONFIG = {"worker_index": init_config.worker_index, "env": asdict(init_config.env_config)}
    _WORKER_ENV = make_env_from_config(init_config.env_config, worker_index=init_config.worker_index)


def _evaluate_task(task: CandidateTask) -> tuple[list[float], dict[str, Any]]:
    global _WORKER_ENV
    if _WORKER_ENV is None:
        raise RuntimeError("worker env is not initialized")
    spec = PolicySpec(
        kind=task.policy_spec["kind"],
        obs_dim=int(task.policy_spec["obs_dim"]),
        action_dim=int(task.policy_spec["action_dim"]),
        hidden_sizes=tuple(task.policy_spec.get("hidden_sizes", [])),
    )
    vector = np.asarray(task.vector_list, dtype=np.float64)
    policy = policy_from_flat(vector, spec)
    result = evaluate_candidate(_WORKER_ENV, policy, eval_episodes=task.eval_episodes, seed_base=task.seed_base)
    return task.vector_list, asdict(result)


def choose_num_workers(args: argparse.Namespace) -> int:
    if args.num_workers > 0:
        return args.num_workers
    cpu_count = os.cpu_count() or 4
    if args.env == "abstract":
        return max(1, min(args.max_workers, max(2, cpu_count - 2)))
    return max(1, min(2, args.max_workers))


def sample_population(rng: np.random.Generator, *, mean: np.ndarray, sigma: float, population: int) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=sigma, size=(population, mean.shape[0]))
    return mean[None, :] + noise


def evaluate_population_parallel(
    *,
    vectors: np.ndarray,
    policy_spec: PolicySpec,
    eval_episodes: int,
    iteration: int,
    seed: int,
    executor: ProcessPoolExecutor,
) -> list[CandidateEvaluation]:
    tasks = []
    for member_idx, vector in enumerate(vectors):
        tasks.append(
            CandidateTask(
                vector_list=vector.tolist(),
                seed_base=seed * 100000 + iteration * 1000 + member_idx * 10,
                eval_episodes=eval_episodes,
                policy_spec=policy_spec.to_jsonable(),
            )
        )
    results = list(executor.map(_evaluate_task, tasks, chunksize=1))
    evaluated = []
    for vector_list, result_dict in results:
        evaluated.append(
            CandidateEvaluation(
                vector=np.asarray(vector_list, dtype=np.float64),
                result=EpisodeResult(**result_dict),
            )
        )
    return evaluated


def main() -> int:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_name)
    rng = np.random.default_rng(args.seed)

    obs_dim = 19
    action_dim = 2
    policy_spec = build_policy_spec(args, obs_dim=obs_dim, action_dim=action_dim)
    env_config = worker_env_config_from_args(args, run_dir)
    num_workers = choose_num_workers(args)

    mean = np.zeros(policy_spec.param_dim, dtype=np.float64)
    if args.init_policy is not None:
        loaded = np.load(args.init_policy)
        loaded = np.asarray(loaded, dtype=np.float64).reshape(-1)
        if loaded.shape != (policy_spec.param_dim,):
            raise ValueError(f"init policy has wrong shape {loaded.shape}, expected {(policy_spec.param_dim,)}")
        mean = loaded.copy()
    sigma = float(args.init_sigma)
    elite_count = max(1, math.ceil(args.population * args.elite_frac))
    best_score = -float("inf")
    history_path = run_dir / "history.jsonl"

    save_json(
        run_dir / "config.json",
        {
            "args": vars(args),
            "policy_spec": policy_spec.to_jsonable(),
            "elite_count": elite_count,
            "num_workers": num_workers,
            "worker_env_config": asdict(env_config),
        },
    )

    start_time = time.time()

    # Use one persistent single-worker process per worker index so live Webots workers
    # keep isolated bridge directories and stable simulator state across evaluations.
    executors = [
        ProcessPoolExecutor(
            max_workers=1,
            initializer=_worker_initializer,
            initargs=(WorkerInitConfig(env_config=env_config, worker_index=worker_index),),
        )
        for worker_index in range(num_workers)
    ]

    try:
        for iteration in range(1, args.iterations + 1):
            vectors = sample_population(rng, mean=mean, sigma=sigma, population=args.population)
            shards = [[] for _ in range(num_workers)]
            for idx, vector in enumerate(vectors):
                shards[idx % num_workers].append((idx, vector))

            candidate_slots: list[CandidateEvaluation | None] = [None] * args.population
            for worker_index, shard in enumerate(shards):
                if not shard:
                    continue
                tasks = [
                    CandidateTask(
                        vector_list=vector.tolist(),
                        seed_base=args.seed * 100000 + iteration * 1000 + idx * 10,
                        eval_episodes=args.eval_episodes,
                        policy_spec=policy_spec.to_jsonable(),
                    )
                    for idx, vector in shard
                ]
                for (idx, _vector), (vector_list, result_dict) in zip(
                    shard,
                    executors[worker_index].map(_evaluate_task, tasks, chunksize=1),
                ):
                    candidate_slots[idx] = CandidateEvaluation(
                        vector=np.asarray(vector_list, dtype=np.float64),
                        result=EpisodeResult(**result_dict),
                    )

            candidates = [candidate for candidate in candidate_slots if candidate is not None]
            candidates.sort(key=lambda item: item.result.total_reward, reverse=True)
            elites = candidates[:elite_count]
            elite_vectors = np.stack([item.vector for item in elites], axis=0)
            mean = elite_vectors.mean(axis=0)
            sample_std = elite_vectors.std(axis=0).mean()
            sigma = max(args.min_sigma, float(0.9 * sigma + 0.1 * sample_std))

            best_candidate = candidates[0]
            best_policy = policy_from_flat(best_candidate.vector, policy_spec)
            if best_candidate.result.total_reward > best_score:
                best_score = best_candidate.result.total_reward
                save_json(
                    run_dir / "best_policy.json",
                    {
                        "score": best_score,
                        "final_distance": best_candidate.result.final_distance,
                        "steps": best_candidate.result.steps,
                        "policy_spec": policy_spec.to_jsonable(),
                        "policy": best_policy.to_jsonable(),
                    },
                )
                np.save(run_dir / "best_policy.npy", best_candidate.vector)

            record = IterationResult(
                iteration=iteration,
                mean_return=float(np.mean([item.result.total_reward for item in candidates])),
                best_return=float(best_candidate.result.total_reward),
                elite_mean_return=float(np.mean([item.result.total_reward for item in elites])),
                sigma=float(sigma),
                best_final_distance=float(best_candidate.result.final_distance),
                elapsed_sec=float(time.time() - start_time),
            )
            with history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(record)) + "\n")
            print(
                f"iter={record.iteration} mean={record.mean_return:.4f} "
                f"elite={record.elite_mean_return:.4f} best={record.best_return:.4f} "
                f"dist={record.best_final_distance:.4f} sigma={record.sigma:.4f}",
                flush=True,
            )

        summary = {
            "best_score": float(best_score),
            "run_dir": str(run_dir),
            "history_path": str(history_path),
            "best_policy_path": str(run_dir / "best_policy.json"),
            "policy_spec": policy_spec.to_jsonable(),
            "num_workers": num_workers,
        }
        save_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
