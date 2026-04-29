"""CQ-0277: terminal player-round 正規化の discard/call 横断テスト"""
import pytest
import numpy as np
import torch

pytestmark = pytest.mark.smoke

from mahjong_rl.stage2a_learner import Stage2aLearner
from mahjong_rl.models.stage2a_model import Stage2aModel
from mahjong_rl.call_shard import DecisionSample, CandidateRecord


def _make_learner(tmp_path):
    model = Stage2aModel(input_dim=10, discard_hidden_dims=[8],
                          optional_hidden_dims=[8])
    return Stage2aLearner(
        config={"training": {"algorithm": "ppo"}},
        model=model, run_dir=tmp_path / "run",
        device=torch.device("cpu"),
    )


class TestCrossBranchWeights:
    """_compute_terminal_weights_cross_branch の挙動"""

    def test_discard_only_matches_legacy(self, tmp_path):
        """call_samples=空 → 旧 _compute_terminal_weights と一致"""
        learner = _make_learner(tmp_path)
        eps = ["e0", "e0", "e1"]
        rds = [0, 0, 1]
        pids = [0, 0, 0]

        d_w, c_w = learner._compute_terminal_weights_cross_branch(
            eps, rds, pids, None, None, None, torch.device("cpu"))
        legacy = learner._compute_terminal_weights(
            eps, rds, pids, 3, torch.device("cpu"))
        assert c_w is None
        assert torch.allclose(d_w, legacy)

    def test_call_only_matches_legacy(self, tmp_path):
        """discard_samples=空 → 旧 _compute_terminal_weights と一致"""
        learner = _make_learner(tmp_path)
        eps = ["e0", "e0", "e1", "e1"]
        rds = [0, 0, 0, 0]
        pids = [0, 1, 0, 1]
        d_w, c_w = learner._compute_terminal_weights_cross_branch(
            None, None, None, eps, rds, pids, torch.device("cpu"))
        legacy = learner._compute_terminal_weights(
            eps, rds, pids, 4, torch.device("cpu"))
        assert d_w is None
        assert torch.allclose(c_w, legacy)

    def test_cross_branch_sums_to_one(self, tmp_path):
        """同 (episode, round, player) に discard 3 + call 2 → 合計 1.0"""
        learner = _make_learner(tmp_path)
        # 同じ (e0, 0, 0) に discard 3, call 2 → 全部で 5 件
        d_eps = ["e0", "e0", "e0"]
        d_rds = [0, 0, 0]
        d_pids = [0, 0, 0]
        c_eps = ["e0", "e0"]
        c_rds = [0, 0]
        c_pids = [0, 0]

        d_w, c_w = learner._compute_terminal_weights_cross_branch(
            d_eps, d_rds, d_pids, c_eps, c_rds, c_pids,
            torch.device("cpu"))
        # 各 row weight = 1/5 = 0.2
        for i in range(3):
            assert d_w[i].item() == pytest.approx(0.2)
        for i in range(2):
            assert c_w[i].item() == pytest.approx(0.2)
        # 合計
        total = d_w.sum().item() + c_w.sum().item()
        assert total == pytest.approx(1.0)

    def test_cross_branch_separate_groups(self, tmp_path):
        """異なる (episode, round, player) は別 group"""
        learner = _make_learner(tmp_path)
        # group A: (e0,0,0) → discard 1
        # group B: (e0,1,0) → discard 1, call 1 (round 違い)
        # group C: (e0,0,1) → call 1 (player 違い)
        # group D: (e1,0,0) → call 1 (episode 違い)
        d_eps = ["e0", "e0"]
        d_rds = [0, 1]
        d_pids = [0, 0]
        c_eps = ["e0", "e0", "e1"]
        c_rds = [1, 0, 0]
        c_pids = [0, 1, 0]

        d_w, c_w = learner._compute_terminal_weights_cross_branch(
            d_eps, d_rds, d_pids, c_eps, c_rds, c_pids,
            torch.device("cpu"))
        # group A (e0,0,0): only d[0] → weight 1.0
        assert d_w[0].item() == pytest.approx(1.0)
        # group B (e0,1,0): d[1] + c[0] → each 0.5
        assert d_w[1].item() == pytest.approx(0.5)
        assert c_w[0].item() == pytest.approx(0.5)
        # group C (e0,0,1): only c[1] → 1.0
        assert c_w[1].item() == pytest.approx(1.0)
        # group D (e1,0,0): only c[2] → 1.0
        assert c_w[2].item() == pytest.approx(1.0)

    def test_branch_order_preserved(self, tmp_path):
        """branch 元順がそのまま返る"""
        learner = _make_learner(tmp_path)
        d_eps = ["e0", "e1", "e0"]
        d_rds = [0, 0, 0]
        d_pids = [0, 0, 0]
        c_eps = ["e0", "e0"]
        c_rds = [0, 0]
        c_pids = [0, 0]

        d_w, c_w = learner._compute_terminal_weights_cross_branch(
            d_eps, d_rds, d_pids, c_eps, c_rds, c_pids,
            torch.device("cpu"))
        # group (e0,0,0): d[0], d[2], c[0], c[1] → 4 件 → each 0.25
        # group (e1,0,0): d[1] のみ → 1.0
        assert d_w[0].item() == pytest.approx(0.25)
        assert d_w[1].item() == pytest.approx(1.0)
        assert d_w[2].item() == pytest.approx(0.25)
        assert c_w[0].item() == pytest.approx(0.25)
        assert c_w[1].item() == pytest.approx(0.25)

    def test_both_empty(self, tmp_path):
        """両 branch 空 → どちらも None"""
        learner = _make_learner(tmp_path)
        d_w, c_w = learner._compute_terminal_weights_cross_branch(
            None, None, None, None, None, None, torch.device("cpu"))
        assert d_w is None
        assert c_w is None


class TestPPOSmokeWithCrossBranch:
    """PPO path で cross-branch normalization が crash しない"""

    def test_ppo_smoke_semantic_aux(self, tmp_path):
        """semantic_aux=on PPO smoke (cross-branch path を踏む)"""
        from mahjong_rl.call_shard import DecisionShardWriter

        # write a small mixed shard
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        # 同 (ep0, round 0, player 0) に discard 2 + call 1
        for i in range(2):
            writer.add(DecisionSample(
                decision_type="discard",
                observation=np.zeros(10, dtype=np.float32),
                legal_mask=np.ones(34, dtype=np.float32),
                action=0, reward=0.1 * i, log_prob=-0.5, value=0.0,
                terminated=False, round_over=False,
                round_terminal_label="win_menzen",
                eventual_win_yaku_ids=[1],
                player_id=0, episode_id="ep0",
                round_id=0, step_id=i,
                actor_type="policy",
                experiment_id="t", run_id="r", worker_id="w",
            ))
        writer.add(DecisionSample(
            decision_type="call",
            observation=np.zeros(10, dtype=np.float32),
            reward=0.0, log_prob=-0.5, value=0.0,
            terminated=True, round_over=True,
            selected_candidate_index=0, candidate_count=1,
            candidates=[CandidateRecord(action_type=8)],
            round_terminal_label="win_menzen",
            eventual_win_yaku_ids=[1],
            player_id=0, episode_id="ep0",
            round_id=0, step_id=2,
            actor_type="policy",
            experiment_id="t", run_id="r", worker_id="w",
        ))
        writer.close()

        model = Stage2aModel(
            input_dim=10, discard_hidden_dims=[8],
            optional_hidden_dims=[8], value_hidden_dims=[8],
            semantic_aux_config={"enabled": True, "policy_projection_dim": 4})
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "ppo", "epochs": 1, "batch_size": 4,
                "semantic_aux": {"enabled": True,
                                  "terminal_loss_coef": 0.2,
                                  "yaku_loss_coef": 0.1},
            }},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        assert metrics["num_updates"] > 0
        assert metrics["terminal_loss"] is not None
