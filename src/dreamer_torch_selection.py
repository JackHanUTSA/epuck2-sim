from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CandidateReturnSummary:
    mean_return: torch.Tensor
    return_std: torch.Tensor
    lower_quantile: torch.Tensor


@dataclass(frozen=True)
class RobustRanking:
    indices: torch.Tensor
    robust_score: torch.Tensor
    summary: CandidateReturnSummary


def summarize_candidate_returns(episode_returns: torch.Tensor, *, quantile: float = 0.25) -> CandidateReturnSummary:
    returns = torch.as_tensor(episode_returns, dtype=torch.float32)
    if returns.ndim != 2:
        raise ValueError(f"episode_returns must have shape (candidates, episodes), got {tuple(returns.shape)}")
    if returns.shape[1] <= 0:
        raise ValueError("episode_returns must include at least one episode per candidate")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")

    mean_return = returns.mean(dim=1)
    return_std = returns.std(dim=1, unbiased=False)
    lower_quantile = torch.quantile(returns, quantile, dim=1)
    return CandidateReturnSummary(
        mean_return=mean_return,
        return_std=return_std,
        lower_quantile=lower_quantile,
    )


def rank_candidates_by_robust_score(
    episode_returns: torch.Tensor,
    *,
    quantile: float = 0.25,
    risk_penalty: float = 0.5,
) -> RobustRanking:
    summary = summarize_candidate_returns(episode_returns, quantile=quantile)
    robust_score = summary.mean_return + summary.lower_quantile - risk_penalty * summary.return_std
    indices = torch.argsort(robust_score, descending=True)
    return RobustRanking(indices=indices, robust_score=robust_score, summary=summary)
