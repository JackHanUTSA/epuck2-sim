import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.dreamer_torch_selection import (  # noqa: E402
    rank_candidates_by_robust_score,
    summarize_candidate_returns,
)


class TorchSelectionTests(unittest.TestCase):
    def test_summarize_candidate_returns_reports_mean_std_and_quantile(self):
        returns = torch.tensor(
            [
                [3.0, 3.0, 3.0, 3.0],
                [5.0, 1.0, 5.0, 1.0],
            ],
            dtype=torch.float32,
        )

        summary = summarize_candidate_returns(returns, quantile=0.25)

        self.assertTrue(torch.allclose(summary.mean_return, torch.tensor([3.0, 3.0])))
        self.assertTrue(torch.allclose(summary.return_std, torch.tensor([0.0, 2.0])))
        self.assertTrue(torch.allclose(summary.lower_quantile, torch.tensor([3.0, 1.0])))

    def test_rank_candidates_by_robust_score_penalizes_high_variance_candidates(self):
        returns = torch.tensor(
            [
                [3.0, 3.0, 3.0, 3.0],
                [5.0, 1.0, 5.0, 1.0],
                [2.5, 2.5, 2.5, 2.5],
            ],
            dtype=torch.float32,
        )

        ranking = rank_candidates_by_robust_score(returns, quantile=0.25, risk_penalty=0.5)

        self.assertEqual(ranking.indices.tolist(), [0, 2, 1])
        self.assertGreater(ranking.robust_score[0].item(), ranking.robust_score[2].item())
        self.assertGreater(ranking.robust_score[2].item(), ranking.robust_score[1].item())


if __name__ == "__main__":
    unittest.main()
