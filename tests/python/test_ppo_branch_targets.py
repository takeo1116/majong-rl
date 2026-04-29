"""CQ-0275: PPO advantage / return / weight の branch 元順 scatter テスト"""
import pytest
import numpy as np
import torch

pytestmark = pytest.mark.smoke

from mahjong_rl.stage2a_learner import Stage2aLearner
from mahjong_rl.models.stage2a_model import Stage2aModel
from mahjong_rl.call_shard import DecisionSample, CandidateRecord


def _make_learner(tmp_path, mixed_mode="separated"):
    model = Stage2aModel(input_dim=10, discard_hidden_dims=[8],
                          optional_hidden_dims=[8])
    return Stage2aLearner(
        config={"training": {
            "algorithm": "ppo",
            "rule_mix_learner": {"ppo_mode": mixed_mode,
                                  "baseline_sample_weight": 0.3},
        }},
        model=model, run_dir=tmp_path / "run",
        device=torch.device("cpu"),
    )


def _mk_discard_sample(step_id, player_id, episode_id="ep0",
                        reward=0.0, value=0.0, terminated=False,
                        actor_type="policy"):
    return DecisionSample(
        decision_type="discard",
        observation=np.zeros(10, dtype=np.float32),
        legal_mask=np.ones(34, dtype=np.float32),
        action=0, reward=reward, log_prob=-0.5, value=value,
        terminated=terminated, round_over=False,
        player_id=player_id, episode_id=episode_id,
        step_id=step_id, actor_type=actor_type,
        experiment_id="t", run_id="r", worker_id="w",
    )


def _mk_call_sample(step_id, player_id, episode_id="ep0",
                    reward=0.0, value=0.0, terminated=False,
                    actor_type="policy"):
    return DecisionSample(
        decision_type="call",
        observation=np.zeros(10, dtype=np.float32),
        reward=reward, log_prob=-0.5, value=value,
        terminated=terminated, round_over=False,
        selected_candidate_index=0, candidate_count=1,
        candidates=[CandidateRecord(action_type=8)],
        player_id=player_id, episode_id=episode_id,
        step_id=step_id, actor_type=actor_type,
        experiment_id="t", run_id="r", worker_id="w",
    )


class TestBranchTargetsScatter:
    """CQ-0275: branch 元順への scatter 整合性"""

    def test_non_monotonic_step_ids(self, tmp_path):
        """branch 内 step_id が非単調でも、各 sample に正しい return/adv が付く"""
        learner = _make_learner(tmp_path)
        # discard branch within: step_ids [10, 5, 20, 15] (non-monotonic)
        # all same player to ensure same trajectory
        d = [
            _mk_discard_sample(10, 0, reward=1.0, value=0.5),
            _mk_discard_sample(5, 0, reward=2.0, value=0.3),
            _mk_discard_sample(20, 0, reward=4.0, value=0.7, terminated=True),
            _mk_discard_sample(15, 0, reward=3.0, value=0.6),
        ]
        c = []
        out = learner._compute_ppo_branch_targets(d, c, is_mixed=False)

        # The step_id order is 5, 10, 15, 20 → indexed 1, 0, 3, 2
        # GAE is computed in this temporal order.
        # We need d_ret/d_adv to map back to original branch order.
        # Build expected directly via _compute_returns_advantages on sorted samples.
        # Use id() to avoid numpy __eq__ ambiguity
        d_id_to_branch = {id(s): i for i, s in enumerate(d)}
        sorted_samples = sorted(d, key=lambda s: s.step_id)
        sorted_idx = [d_id_to_branch[id(s)] for s in sorted_samples]
        ret_sorted, adv_sorted = learner._compute_returns_advantages(sorted_samples)

        # Apply same advantage normalization
        if adv_sorted.numel() > 1:
            adv_sorted = (adv_sorted - adv_sorted.mean()) / (adv_sorted.std() + 1e-8)

        # Expected: branch_idx position should hold sorted_idx position's value
        for sorted_pos, branch_idx in enumerate(sorted_idx):
            assert out["d_ret"][branch_idx].item() == pytest.approx(
                ret_sorted[sorted_pos].item(), abs=1e-5)
            assert out["d_adv"][branch_idx].item() == pytest.approx(
                adv_sorted[sorted_pos].item(), abs=1e-5)

    def test_interleaved_discard_call(self, tmp_path):
        """discard / call が interleaved した trajectory で同 player GAE が一致する"""
        learner = _make_learner(tmp_path)
        # Same player, alternating discard/call
        d = [
            _mk_discard_sample(0, 0, reward=1.0, value=0.1),
            _mk_discard_sample(2, 0, reward=2.0, value=0.2),
            _mk_discard_sample(4, 0, reward=3.0, value=0.3, terminated=True),
        ]
        c = [
            _mk_call_sample(1, 0, reward=10.0, value=1.0),
            _mk_call_sample(3, 0, reward=20.0, value=2.0),
        ]
        out = learner._compute_ppo_branch_targets(d, c, is_mixed=False)

        # Verify each branch sample's value matches what's in all_sorted at its sorted position
        all_sorted = out["all_sorted"]
        sorted_id_to_pos = {id(s): i for i, s in enumerate(all_sorted)}
        ret, adv = learner._compute_returns_advantages(all_sorted)
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # For each original branch sample, find its position in all_sorted
        # and verify that d_ret[branch_idx] / c_ret[branch_idx] match
        for i, sample in enumerate(d):
            sorted_pos = sorted_id_to_pos[id(sample)]
            assert out["d_ret"][i].item() == pytest.approx(
                ret[sorted_pos].item(), abs=1e-5)
            assert out["d_adv"][i].item() == pytest.approx(
                adv[sorted_pos].item(), abs=1e-5)
        for i, sample in enumerate(c):
            sorted_pos = sorted_id_to_pos[id(sample)]
            assert out["c_ret"][i].item() == pytest.approx(
                ret[sorted_pos].item(), abs=1e-5)

    def test_mixed_baseline_weight_branch_order(self, tmp_path):
        """mixed PPO の weight も元 branch 順に一致する"""
        learner = _make_learner(tmp_path, mixed_mode="mixed")
        d = [
            _mk_discard_sample(10, 0, actor_type="baseline"),  # weight=0.3
            _mk_discard_sample(5, 0, actor_type="policy"),     # weight=1.0
            _mk_discard_sample(20, 0, actor_type="baseline"),  # weight=0.3
        ]
        c = [
            _mk_call_sample(15, 0, actor_type="policy"),       # weight=1.0
        ]
        out = learner._compute_ppo_branch_targets(d, c, is_mixed=True)
        # d_weights は元 branch 順 (d[0]=baseline, d[1]=policy, d[2]=baseline)
        assert out["d_weights"][0].item() == pytest.approx(0.3)
        assert out["d_weights"][1].item() == pytest.approx(1.0)
        assert out["d_weights"][2].item() == pytest.approx(0.3)
        assert out["c_weights"][0].item() == pytest.approx(1.0)

    def test_empty_call_branch(self, tmp_path):
        """call_samples が空でも crash しない"""
        learner = _make_learner(tmp_path)
        d = [_mk_discard_sample(0, 0, reward=1.0, terminated=True)]
        c = []
        out = learner._compute_ppo_branch_targets(d, c)
        assert out["d_ret"] is not None
        assert out["c_ret"] is None

    def test_empty_discard_branch(self, tmp_path):
        """discard_samples が空でも crash しない"""
        learner = _make_learner(tmp_path)
        d = []
        c = [_mk_call_sample(0, 0, reward=1.0, terminated=True)]
        out = learner._compute_ppo_branch_targets(d, c)
        assert out["d_ret"] is None
        assert out["c_ret"] is not None

    def test_branch_idx_positions(self, tmp_path):
        """各 sample の reward が元 index 位置の return に反映される"""
        learner = _make_learner(tmp_path)
        # 異なる player → 異なる group なので独立 GAE
        d = [
            _mk_discard_sample(0, 0, reward=10.0, value=0.0, terminated=True),
            _mk_discard_sample(1, 1, reward=20.0, value=0.0, terminated=True),
            _mk_discard_sample(2, 2, reward=30.0, value=0.0, terminated=True),
        ]
        out = learner._compute_ppo_branch_targets(d, [])
        # Each player's last-and-only sample → return == reward
        assert out["d_ret"][0].item() == pytest.approx(10.0, abs=1e-4)
        assert out["d_ret"][1].item() == pytest.approx(20.0, abs=1e-4)
        assert out["d_ret"][2].item() == pytest.approx(30.0, abs=1e-4)
