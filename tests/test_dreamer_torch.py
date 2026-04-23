import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.dreamer_env import (  # noqa: E402
    DreamerTeamEnvConfig,
    build_observation,
    compute_reward,
    initial_state,
    step_team_state,
)
from src.dreamer_torch import (  # noqa: E402
    TorchPolicySpec,
    build_observation_batch,
    compute_reward_batch,
    policy_from_flat,
    state_from_team_states,
    step_batch,
)


class DreamerTorchTests(unittest.TestCase):
    def test_batched_step_matches_python_reference_for_single_transition(self):
        config = DreamerTeamEnvConfig()
        previous = initial_state(waypoint=(0.31, -0.17), config=config)
        action = (0.42, -0.33)
        current = step_team_state(previous, action=action, config=config)

        batch_state = state_from_team_states([previous], device="cpu")
        batch_action = torch.tensor([action], dtype=torch.float32)
        next_state = step_batch(batch_state, batch_action, config=config)
        next_obs = build_observation_batch(next_state)[0].tolist()
        next_reward = compute_reward_batch(batch_state, next_state)[0].item()

        self.assertEqual(len(next_obs), 19)
        self.assertAlmostEqual(next_reward, compute_reward(previous, current), places=5)
        for actual, expected in zip(next_obs, build_observation(current)):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_policy_from_flat_produces_bounded_actions(self):
        spec = TorchPolicySpec(kind="mlp", obs_dim=19, action_dim=2, hidden_sizes=(8, 4))
        vector = torch.linspace(-0.3, 0.3, spec.param_dim)
        policy = policy_from_flat(vector, spec, device="cpu")
        obs = torch.linspace(-1.0, 1.0, 19).unsqueeze(0)
        action = policy.act(obs)

        self.assertEqual(tuple(action.shape), (1, 2))
        self.assertTrue(torch.all(action <= 1.0).item())
        self.assertTrue(torch.all(action >= -1.0).item())


if __name__ == "__main__":
    unittest.main()
