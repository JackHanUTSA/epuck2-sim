#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dreamer_env import DreamerTeamEnvConfig  # noqa: E402
from src.dreamer_torch import (  # noqa: E402
    TorchPolicySpec,
    TorchTeamState,
    build_observation_batch,
    compute_reward_batch,
    step_batch,
)
from src.dreamer_torch_selection import rank_candidates_by_robust_score  # noqa: E402


@dataclass
class IterationResult:
    iteration: int
    mean_return: float
    best_return: float
    elite_mean_return: float
    sigma: float
    best_final_distance: float
    terminated_fraction: float
    elapsed_sec: float


class GroupedMLPPolicy:
    def __init__(self, weight_batches: list[torch.Tensor], bias_batches: list[torch.Tensor]):
        self.weight_batches = weight_batches
        self.bias_batches = bias_batches

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        for idx, (weight, bias) in enumerate(zip(self.weight_batches, self.bias_batches)):
            x = torch.einsum("bi,boi->bo", x, weight) + bias
            x = torch.tanh(x)
            if idx == len(self.weight_batches) - 1:
                x = torch.clamp(x, -1.0, 1.0)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-accelerated abstract CEM trainer using batched PyTorch simulation.")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--elite-frac", type=float, default=0.125)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=120)
    parser.add_argument("--hidden-sizes", type=str, default="256,128")
    parser.add_argument("--init-sigma", type=float, default=0.22)
    parser.add_argument("--min-sigma", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--init-policy", type=Path, default=None)
    parser.add_argument("--waypoint-margin", type=float, default=0.12)
    parser.add_argument("--robustness-top-k", type=int, default=8)
    parser.add_argument("--robustness-eval-episodes", type=int, default=12)
    parser.add_argument("--robustness-quantile", type=float, default=0.25)
    parser.add_argument("--robustness-risk-penalty", type=float, default=0.5)
    return parser.parse_args()


def ensure_run_dir(run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = run_name or f"torch-cem-{timestamp}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def parse_hidden_sizes(text: str) -> tuple[int, ...]:
    text = text.strip()
    if not text:
        return ()
    values = tuple(int(chunk) for chunk in text.split(",") if chunk.strip())
    if any(v <= 0 for v in values):
        raise ValueError(f"hidden sizes must be positive: {values}")
    return values


def sample_waypoints(batch_size: int, *, config: DreamerTeamEnvConfig, margin: float, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    radius = max(0.0, config.arena_half_extent - margin)
    return (torch.rand((batch_size, 2), generator=generator, device=device) * 2.0 - 1.0) * radius


def initial_state_batch(waypoints: torch.Tensor, *, device: torch.device) -> TorchTeamState:
    batch_size = waypoints.shape[0]
    return TorchTeamState(
        centroid=torch.zeros((batch_size, 2), dtype=torch.float32, device=device),
        waypoint=waypoints,
        team_heading=torch.zeros((batch_size,), dtype=torch.float32, device=device),
        prev_action=torch.zeros((batch_size, 2), dtype=torch.float32, device=device),
        step_count=torch.zeros((batch_size,), dtype=torch.int64, device=device),
    )


def decode_population(vectors: torch.Tensor, spec: TorchPolicySpec, *, eval_episodes: int) -> GroupedMLPPolicy:
    population = vectors.shape[0]
    offset = 0
    weights: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    for in_dim, out_dim in zip(spec.layer_sizes[:-1], spec.layer_sizes[1:]):
        weight_count = in_dim * out_dim
        weight = vectors[:, offset: offset + weight_count].reshape(population, out_dim, in_dim)
        offset += weight_count
        bias = vectors[:, offset: offset + out_dim].reshape(population, out_dim)
        offset += out_dim
        weights.append(weight.repeat_interleave(eval_episodes, dim=0))
        biases.append(bias.repeat_interleave(eval_episodes, dim=0))
    return GroupedMLPPolicy(weights, biases)


def evaluate_population_episode_returns(
    vectors: torch.Tensor,
    *,
    spec: TorchPolicySpec,
    config: DreamerTeamEnvConfig,
    eval_episodes: int,
    episode_steps: int,
    margin: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    population = vectors.shape[0]
    batch_size = population * eval_episodes
    grouped_policy = decode_population(vectors, spec, eval_episodes=eval_episodes)
    waypoints = sample_waypoints(batch_size, config=config, margin=margin, generator=generator, device=device)
    state = initial_state_batch(waypoints, device=device)
    total_reward = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    terminated = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    final_distance = torch.linalg.norm(state.waypoint - state.centroid, dim=1)

    for _ in range(episode_steps):
        obs = build_observation_batch(state, config=config)
        action = grouped_policy.act(obs)
        next_state = step_batch(state, action, config=config)
        reward = compute_reward_batch(state, next_state)
        distance = torch.linalg.norm(next_state.waypoint - next_state.centroid, dim=1)
        done_now = distance < config.reach_radius
        active = ~terminated
        total_reward = total_reward + reward * active.float()
        final_distance = torch.where(active, distance, final_distance)
        terminated = terminated | done_now
        state = next_state
        if bool(torch.all(terminated).item()):
            break

    episode_returns = total_reward.reshape(population, eval_episodes)
    distances = final_distance.reshape(population, eval_episodes)
    terminated_fraction = terminated.reshape(population, eval_episodes).float().mean(dim=1)
    return episode_returns, distances, terminated_fraction


def evaluate_population(
    vectors: torch.Tensor,
    *,
    spec: TorchPolicySpec,
    config: DreamerTeamEnvConfig,
    eval_episodes: int,
    episode_steps: int,
    margin: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    episode_returns, distances, terminated_fraction = evaluate_population_episode_returns(
        vectors,
        spec=spec,
        config=config,
        eval_episodes=eval_episodes,
        episode_steps=episode_steps,
        margin=margin,
        generator=generator,
        device=device,
    )
    returns = episode_returns.mean(dim=1)
    mean_distances = distances.mean(dim=1)
    return returns, mean_distances, terminated_fraction


def main() -> int:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_name)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config = DreamerTeamEnvConfig()
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    spec = TorchPolicySpec(kind="mlp", obs_dim=19, action_dim=2, hidden_sizes=hidden_sizes)
    param_dim = spec.param_dim
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    mean = torch.zeros((param_dim,), dtype=torch.float32, device=device)
    if args.init_policy is not None:
        loaded = torch.from_numpy(__import__("numpy").load(args.init_policy)).to(device=device, dtype=torch.float32).reshape(-1)
        if loaded.shape != (param_dim,):
            raise ValueError(f"init policy has wrong shape {tuple(loaded.shape)}, expected {(param_dim,)}")
        mean = loaded.clone()
    sigma = float(args.init_sigma)
    elite_count = max(1, math.ceil(args.population * args.elite_frac))
    best_score = -float("inf")
    history_path = run_dir / "history.jsonl"

    save_json(
        run_dir / "config.json",
        {
            "args": vars(args),
            "policy_spec": {
                "kind": spec.kind,
                "obs_dim": spec.obs_dim,
                "action_dim": spec.action_dim,
                "hidden_sizes": list(spec.hidden_sizes),
                "param_dim": spec.param_dim,
            },
            "device": str(device),
            "elite_count": elite_count,
        },
    )

    start_time = time.time()
    for iteration in range(1, args.iterations + 1):
        noise = torch.randn((args.population, param_dim), generator=generator, device=device, dtype=torch.float32)
        vectors = mean.unsqueeze(0) + sigma * noise
        returns, distances, terminated_fraction = evaluate_population(
            vectors,
            spec=spec,
            config=config,
            eval_episodes=args.eval_episodes,
            episode_steps=args.episode_steps,
            margin=args.waypoint_margin,
            generator=generator,
            device=device,
        )
        order = torch.argsort(returns, descending=True)
        elite_idx = order[:elite_count]
        elite_vectors = vectors[elite_idx]
        mean = elite_vectors.mean(dim=0)
        sample_std = elite_vectors.std(dim=0).mean().item()
        sigma = max(args.min_sigma, 0.9 * sigma + 0.1 * sample_std)

        robust_top_k = max(0, min(args.robustness_top_k, args.population))
        robust_eval_episodes = max(0, args.robustness_eval_episodes)
        robust_best_idx = int(order[0].item())
        robust_best_score = float(returns[robust_best_idx].item())
        robust_best_distance = float(distances[robust_best_idx].item())
        robust_best_terminated_fraction = float(terminated_fraction[robust_best_idx].item())
        robust_summary: dict[str, Any] | None = None

        if robust_top_k > 1 and robust_eval_episodes > 0:
            candidate_idx = order[:robust_top_k]
            candidate_vectors = vectors[candidate_idx]
            robust_episode_returns, robust_distances, robust_terminated_fraction = evaluate_population_episode_returns(
                candidate_vectors,
                spec=spec,
                config=config,
                eval_episodes=robust_eval_episodes,
                episode_steps=args.episode_steps,
                margin=args.waypoint_margin,
                generator=generator,
                device=device,
            )
            robust_ranking = rank_candidates_by_robust_score(
                robust_episode_returns,
                quantile=args.robustness_quantile,
                risk_penalty=args.robustness_risk_penalty,
            )
            chosen_offset = int(robust_ranking.indices[0].item())
            robust_best_idx = int(candidate_idx[chosen_offset].item())
            robust_best_score = float(robust_ranking.robust_score[chosen_offset].item())
            robust_best_distance = float(robust_distances[chosen_offset].mean().item())
            robust_best_terminated_fraction = float(robust_terminated_fraction[chosen_offset].item())
            robust_summary = {
                "candidate_count": int(candidate_idx.shape[0]),
                "eval_episodes": robust_eval_episodes,
                "quantile": float(args.robustness_quantile),
                "risk_penalty": float(args.robustness_risk_penalty),
                "selected_population_index": robust_best_idx,
                "selected_rank_within_top_k": chosen_offset,
                "mean_return": float(robust_ranking.summary.mean_return[chosen_offset].item()),
                "return_std": float(robust_ranking.summary.return_std[chosen_offset].item()),
                "lower_quantile": float(robust_ranking.summary.lower_quantile[chosen_offset].item()),
                "robust_score": robust_best_score,
            }

        best_idx = robust_best_idx
        best_return = robust_best_score
        best_distance = robust_best_distance
        if best_return > best_score:
            best_score = best_return
            best_vector = vectors[best_idx].detach().cpu()
            torch.save(best_vector, run_dir / "best_policy.pt")
            payload = {
                "score": best_score,
                "final_distance": best_distance,
                "terminated_fraction": robust_best_terminated_fraction,
                "policy_spec": {
                    "kind": spec.kind,
                    "obs_dim": spec.obs_dim,
                    "action_dim": spec.action_dim,
                    "hidden_sizes": list(spec.hidden_sizes),
                    "param_dim": spec.param_dim,
                },
            }
            if robust_summary is not None:
                payload["robust_validation"] = robust_summary
            save_json(run_dir / "best_policy.json", payload)

        record = IterationResult(
            iteration=iteration,
            mean_return=float(returns.mean().item()),
            best_return=best_return,
            elite_mean_return=float(returns[elite_idx].mean().item()),
            sigma=float(sigma),
            best_final_distance=best_distance,
            terminated_fraction=float(terminated_fraction.mean().item()),
            elapsed_sec=float(time.time() - start_time),
        )
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")
        print(
            f"iter={record.iteration} mean={record.mean_return:.4f} elite={record.elite_mean_return:.4f} "
            f"best={record.best_return:.4f} dist={record.best_final_distance:.4f} "
            f"term={record.terminated_fraction:.3f} sigma={record.sigma:.4f}",
            flush=True,
        )

    summary = {
        "best_score": float(best_score),
        "run_dir": str(run_dir),
        "history_path": str(history_path),
        "best_policy_path": str(run_dir / "best_policy.json"),
        "device": str(device),
        "policy_spec": {
            "kind": spec.kind,
            "obs_dim": spec.obs_dim,
            "action_dim": spec.action_dim,
            "hidden_sizes": list(spec.hidden_sizes),
            "param_dim": spec.param_dim,
        },
    }
    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
