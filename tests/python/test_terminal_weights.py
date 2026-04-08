"""CQ-0268: terminal semantic loss player-round 正規化テスト"""
import pytest
import numpy as np
import torch

pytestmark = pytest.mark.smoke

from mahjong_rl.stage2a_learner import Stage2aLearner
from mahjong_rl.models.stage2a_model import Stage2aModel
from mahjong_rl.call_shard import DecisionSample, CandidateRecord, DecisionShardWriter


# ========== weight computation ==========

class TestTerminalWeightComputation:
    """player-round weight の計算"""

    def test_uniform_group(self):
        """1 group 4 samples → 各 0.25"""
        w = Stage2aLearner._compute_terminal_weights(
            episode_ids=["e0", "e0", "e0", "e0"],
            round_ids=[0, 0, 0, 0],
            player_ids=[0, 0, 0, 0],
            n=4, device=torch.device("cpu"),
        )
        assert w.shape == (4,)
        assert torch.allclose(w, torch.tensor([0.25, 0.25, 0.25, 0.25]))
        assert w.sum().item() == pytest.approx(1.0)

    def test_two_groups_different_length(self):
        """group A (2 samples) + group B (4 samples)"""
        w = Stage2aLearner._compute_terminal_weights(
            episode_ids=["e0", "e0", "e1", "e1", "e1", "e1"],
            round_ids=[0, 0, 0, 0, 0, 0],
            player_ids=[0, 0, 0, 0, 0, 0],
            n=6, device=torch.device("cpu"),
        )
        # group e0: 2 samples → 0.5 each
        assert w[0].item() == pytest.approx(0.5)
        assert w[1].item() == pytest.approx(0.5)
        # group e1: 4 samples → 0.25 each
        assert w[2].item() == pytest.approx(0.25)
        # sum per group
        assert w[:2].sum().item() == pytest.approx(1.0)
        assert w[2:].sum().item() == pytest.approx(1.0)

    def test_round_id_separates_groups(self):
        """same episode/player but different round → separate groups"""
        w = Stage2aLearner._compute_terminal_weights(
            episode_ids=["e0", "e0", "e0", "e0"],
            round_ids=[0, 0, 1, 1],
            player_ids=[0, 0, 0, 0],
            n=4, device=torch.device("cpu"),
        )
        # round 0: 2 → 0.5 each
        assert w[0].item() == pytest.approx(0.5)
        assert w[1].item() == pytest.approx(0.5)
        # round 1: 2 → 0.5 each
        assert w[2].item() == pytest.approx(0.5)
        assert w[3].item() == pytest.approx(0.5)

    def test_player_id_separates_groups(self):
        """same episode/round but different player → separate groups"""
        w = Stage2aLearner._compute_terminal_weights(
            episode_ids=["e0", "e0", "e0"],
            round_ids=[0, 0, 0],
            player_ids=[0, 0, 1],
            n=3, device=torch.device("cpu"),
        )
        # player 0: 2 → 0.5 each
        assert w[0].item() == pytest.approx(0.5)
        assert w[1].item() == pytest.approx(0.5)
        # player 1: 1 → 1.0
        assert w[2].item() == pytest.approx(1.0)

    def test_short_group_heavier_than_long(self):
        """短い group の各 row 重みが長い group より大きい"""
        w = Stage2aLearner._compute_terminal_weights(
            episode_ids=["e0", "e0", "e1"] * 3 + ["e1"],
            round_ids=[0] * 10,
            player_ids=[0] * 10,
            n=10, device=torch.device("cpu"),
        )
        # e0 group: indices 0,1,3,4,6,7 → 6 samples → 1/6 each
        # e1 group: indices 2,5,8,9 → 4 samples → 1/4 each
        assert w[2].item() > w[0].item()  # 1/4 > 1/6


# ========== semantic loss integration ==========

class TestTerminalLossWeighted:
    """terminal CE にだけ weight が掛かり、yaku は変わらない"""

    def test_weighted_vs_unweighted_differs(self):
        """weighted terminal CE ≠ unweighted"""
        from mahjong_rl.outcome_vocab import NUM_TERMINAL_CLASSES, NUM_YAKU
        logits = torch.randn(4, NUM_TERMINAL_CLASSES)
        targets = torch.tensor([0, 1, 2, 3])
        yaku_logits = torch.randn(4, NUM_YAKU)
        yaku_targets = torch.zeros(4, NUM_YAKU)
        is_winner = torch.zeros(4)
        semantic = {
            "terminal_logits": logits,
            "yaku_logits": yaku_logits,
            "semantic_summary": torch.zeros(4, 10),
        }
        model = Stage2aModel(
            input_dim=50, discard_hidden_dims=[16],
            optional_hidden_dims=[16], value_hidden_dims=[16],
            semantic_aux_config={"enabled": True, "policy_projection_dim": 4})
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "imitation",
                "semantic_aux": {"enabled": True,
                                  "terminal_loss_coef": 1.0,
                                  "yaku_loss_coef": 0.0},
            }},
            model=model, run_dir=torch.device("cpu"),
            device=torch.device("cpu"),
        )

        # uniform weights (same as unweighted mean)
        w_uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])
        loss_u, tl_u, _ = learner._compute_semantic_aux_loss(
            semantic, targets, yaku_targets, is_winner,
            terminal_weights=w_uniform)

        # skewed weights (different)
        w_skewed = torch.tensor([0.5, 0.5, 0.0, 0.0])
        loss_s, tl_s, _ = learner._compute_semantic_aux_loss(
            semantic, targets, yaku_targets, is_winner,
            terminal_weights=w_skewed)

        assert tl_u != pytest.approx(tl_s, abs=1e-5)

    def test_yaku_unchanged_by_terminal_weights(self):
        """terminal weights は yaku loss に影響しない"""
        from mahjong_rl.outcome_vocab import NUM_TERMINAL_CLASSES, NUM_YAKU
        semantic = {
            "terminal_logits": torch.randn(4, NUM_TERMINAL_CLASSES),
            "yaku_logits": torch.randn(4, NUM_YAKU),
            "semantic_summary": torch.zeros(4, 10),
        }
        targets = torch.tensor([0, 1, 2, 3])
        yaku_targets = torch.zeros(4, NUM_YAKU)
        yaku_targets[0, 0] = 1.0
        is_winner = torch.tensor([1.0, 0.0, 0.0, 0.0])

        model = Stage2aModel(
            input_dim=50, discard_hidden_dims=[16],
            optional_hidden_dims=[16], value_hidden_dims=[16],
            semantic_aux_config={"enabled": True, "policy_projection_dim": 4})
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "imitation",
                "semantic_aux": {"enabled": True,
                                  "terminal_loss_coef": 0.0,
                                  "yaku_loss_coef": 1.0},
            }},
            model=model, run_dir=torch.device("cpu"),
            device=torch.device("cpu"),
        )

        w1 = torch.tensor([0.25, 0.25, 0.25, 0.25])
        w2 = torch.tensor([0.5, 0.5, 0.0, 0.0])
        _, _, yl1 = learner._compute_semantic_aux_loss(
            semantic, targets, yaku_targets, is_winner, terminal_weights=w1)
        _, _, yl2 = learner._compute_semantic_aux_loss(
            semantic, targets, yaku_targets, is_winner, terminal_weights=w2)
        assert yl1 == pytest.approx(yl2)


# ========== integration smokes ==========

def _write_shards_with_rounds(shard_dir, obs_dim=50, n_per_round=5):
    """2 round × 2 player の labeled shard"""
    writer = DecisionShardWriter(shard_dir, max_samples=10000)
    labels = ["win_menzen", "deal_in", "draw_tenpai", "other_non_dealin"]
    step = 0
    for round_id in range(2):
        for pid in range(2):
            for i in range(n_per_round):
                mask = np.ones(34, dtype=np.float32)
                label = labels[(round_id * 2 + pid) % len(labels)]
                writer.add(DecisionSample(
                    decision_type="discard",
                    observation=np.random.randn(obs_dim).astype(np.float32),
                    reward=0.0, log_prob=-0.5, value=0.0,
                    terminated=(i == n_per_round - 1), round_over=False,
                    action=0, legal_mask=mask,
                    player_id=pid, episode_id="ep0", round_id=round_id,
                    step_id=step,
                    round_terminal_label=label,
                    experiment_id="test", run_id="r0", worker_id="w0",
                ))
                step += 1
    writer.close()


class TestImitationSmoke:
    """imitation path で crash しない"""

    def test_imitation_with_terminal_weights(self, tmp_path):
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_shards_with_rounds(shard_dir, obs_dim=obs_dim)

        model = Stage2aModel(
            input_dim=obs_dim, discard_hidden_dims=[16],
            optional_hidden_dims=[16], value_hidden_dims=[16],
            semantic_aux_config={"enabled": True, "policy_projection_dim": 4})
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "imitation", "epochs": 1, "batch_size": 8,
                "semantic_aux": {"enabled": True,
                                  "terminal_loss_coef": 0.2,
                                  "yaku_loss_coef": 0.1},
            }},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["num_updates"] > 0
        assert metrics["terminal_loss"] is not None


class TestPPOSmoke:
    """PPO path で crash しない"""

    def test_ppo_with_terminal_weights(self, tmp_path):
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_shards_with_rounds(shard_dir, obs_dim=obs_dim)

        model = Stage2aModel(
            input_dim=obs_dim, discard_hidden_dims=[16],
            optional_hidden_dims=[16], value_hidden_dims=[16],
            semantic_aux_config={"enabled": True, "policy_projection_dim": 4})
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "ppo", "epochs": 1, "batch_size": 8,
                "semantic_aux": {"enabled": True,
                                  "terminal_loss_coef": 0.2,
                                  "yaku_loss_coef": 0.1},
            }},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["num_updates"] > 0
        assert metrics["terminal_loss"] is not None
