"""CQ-0281: Stage2a PPO diagnostics quantile expansion テスト"""
import pytest
import numpy as np
import torch
from pathlib import Path

pytestmark = pytest.mark.smoke

from mahjong_rl.stage2a_learner import Stage2aLearner
from mahjong_rl.models.stage2a_model import Stage2aModel
from mahjong_rl.call_shard import (
    DecisionSample, CandidateRecord, DecisionShardWriter,
)


# ========== helper unit tests ==========


class TestSafeNpQuantiles:
    """_safe_np_quantiles の挙動"""

    def test_normal(self):
        out = Stage2aLearner._safe_np_quantiles(
            np.arange(101, dtype=np.float64), [0.01, 0.5, 0.99])
        assert out["p01"] == pytest.approx(1.0)
        assert out["p50"] == pytest.approx(50.0)
        assert out["p99"] == pytest.approx(99.0)

    def test_empty(self):
        out = Stage2aLearner._safe_np_quantiles(np.array([]), [0.5])
        assert out["p50"] is None

    def test_none(self):
        out = Stage2aLearner._safe_np_quantiles(None, [0.5])
        assert out["p50"] is None


class TestWeightedHelpers:
    """_weighted_mean / _weighted_fraction"""

    def test_weighted_mean_unweighted(self):
        m = Stage2aLearner._weighted_mean(np.array([1.0, 2.0, 3.0]), None)
        assert m == pytest.approx(2.0)

    def test_weighted_mean_with_weights(self):
        m = Stage2aLearner._weighted_mean(
            np.array([1.0, 2.0]), np.array([3.0, 1.0]))
        assert m == pytest.approx((3.0 + 2.0) / 4.0)

    def test_weighted_mean_empty(self):
        assert Stage2aLearner._weighted_mean(np.array([]), None) is None

    def test_weighted_fraction(self):
        f = Stage2aLearner._weighted_fraction(
            np.array([True, False, True, False]),
            np.array([1.0, 2.0, 3.0, 4.0]))
        assert f == pytest.approx((1.0 + 3.0) / 10.0)


class TestComputePpoDiagStats:
    """_compute_ppo_diag_stats の主要 key を確認"""

    def test_full_input_emits_all_keys(self):
        rng = np.random.RandomState(42)
        n = 100
        log_ratios = rng.randn(n) * 0.1
        advantages = rng.randn(n)  # 正負混在
        max_probs = np.clip(rng.rand(n), 0.0, 1.0)
        weights = np.ones(n)
        stats = Stage2aLearner._compute_ppo_diag_stats(
            log_ratios, advantages, max_probs, weights, clip_epsilon=0.2)
        # log_ratio quantiles
        for k in ("log_ratio_mean", "log_ratio_std", "log_ratio_min",
                  "log_ratio_max", "log_ratio_p01", "log_ratio_p05",
                  "log_ratio_p50", "log_ratio_p95", "log_ratio_p99"):
            assert k in stats and stats[k] is not None
        # ratio quantiles
        for k in ("ratio_p01", "ratio_p50", "ratio_p99", "ratio_max",
                  "clip_fraction"):
            assert k in stats
        # advantage sign / quantiles / cross
        for k in ("advantage_pos_frac", "advantage_neg_frac",
                  "advantage_zero_frac", "advantage_abs_mean",
                  "advantage_p01", "advantage_p50", "advantage_p99",
                  "num_adv_pos", "num_adv_neg",
                  "log_ratio_mean_adv_pos", "log_ratio_mean_adv_neg",
                  "ratio_mean_adv_pos", "ratio_mean_adv_neg",
                  "clip_fraction_adv_pos", "clip_fraction_adv_neg"):
            assert k in stats
        # max_prob quantiles
        for k in ("max_prob_mean", "max_prob_p50", "max_prob_p90",
                  "max_prob_p95", "max_prob_p99"):
            assert k in stats and stats[k] is not None

    def test_advantage_sign_fractions_sum_to_one(self):
        a = np.array([1.0, -1.0, 2.0, 0.0, -3.0, 0.0])
        stats = Stage2aLearner._compute_ppo_diag_stats(
            np.zeros(6), a, np.zeros(6), None, clip_epsilon=0.2)
        total = (stats["advantage_pos_frac"]
                 + stats["advantage_neg_frac"]
                 + stats["advantage_zero_frac"])
        assert total == pytest.approx(1.0)

    def test_max_prob_quantiles_in_unit_range(self):
        rng = np.random.RandomState(0)
        # softmax 由来を模擬 (0..1 範囲)
        mp = rng.rand(50)
        stats = Stage2aLearner._compute_ppo_diag_stats(
            np.zeros(50), np.zeros(50), mp, None, clip_epsilon=0.2)
        for k in ("max_prob_p50", "max_prob_p90", "max_prob_p95",
                  "max_prob_p99"):
            assert 0.0 <= stats[k] <= 1.0

    def test_clip_fraction_split(self):
        # ratio = 1.5 (clipped) for adv_pos / 0.5 (clipped) for adv_neg
        n = 4
        log_ratios = np.array([np.log(1.5), np.log(0.5),
                                np.log(1.5), np.log(0.5)])
        advantages = np.array([1.0, -1.0, 1.0, -1.0])
        stats = Stage2aLearner._compute_ppo_diag_stats(
            log_ratios, advantages, None, None, clip_epsilon=0.2)
        # clip_epsilon=0.2 で |1.5-1|=0.5>0.2, |0.5-1|=0.5>0.2
        assert stats["clip_fraction_adv_pos"] == pytest.approx(1.0)
        assert stats["clip_fraction_adv_neg"] == pytest.approx(1.0)
        assert stats["num_adv_pos"] == 2
        assert stats["num_adv_neg"] == 2

    def test_one_sided_advantage_pos_only(self):
        """advantage が全 positive"""
        log_ratios = np.array([0.1, 0.2, -0.1])
        advantages = np.array([1.0, 2.0, 3.0])
        stats = Stage2aLearner._compute_ppo_diag_stats(
            log_ratios, advantages, None, None, clip_epsilon=0.2)
        assert stats["log_ratio_mean_adv_pos"] is not None
        assert stats["log_ratio_mean_adv_neg"] is None
        assert stats["ratio_mean_adv_neg"] is None
        assert stats["clip_fraction_adv_neg"] is None
        assert stats["num_adv_pos"] == 3
        assert stats["num_adv_neg"] == 0

    def test_one_sided_advantage_neg_only(self):
        log_ratios = np.array([0.1, 0.2])
        advantages = np.array([-1.0, -2.0])
        stats = Stage2aLearner._compute_ppo_diag_stats(
            log_ratios, advantages, None, None, clip_epsilon=0.2)
        assert stats["log_ratio_mean_adv_pos"] is None
        assert stats["log_ratio_mean_adv_neg"] is not None
        assert stats["num_adv_pos"] == 0
        assert stats["num_adv_neg"] == 2

    def test_empty(self):
        stats = Stage2aLearner._compute_ppo_diag_stats(
            None, None, None, None, clip_epsilon=0.2)
        assert stats == {}


# ========== PPO smoke ==========


def _mk_discard(step_id, version=3, **kw):
    base = dict(
        decision_type="discard",
        observation=np.zeros(10, dtype=np.float32),
        legal_mask=np.ones(34, dtype=np.float32),
        action=0, reward=0.0, log_prob=-0.5, value=0.0,
        terminated=False, round_over=False,
        player_id=0, episode_id="ep0", round_id=0,
        step_id=step_id, actor_type="policy",
        experiment_id="t", run_id="r", worker_id="w",
    )
    base.update(kw)
    s = DecisionSample(**base)
    s.sample_semantics_version = version
    return s


def _mk_call(step_id, version=3, **kw):
    base = dict(
        decision_type="call",
        observation=np.zeros(10, dtype=np.float32),
        reward=0.0, log_prob=-0.5, value=0.0,
        terminated=False, round_over=False,
        selected_candidate_index=0, candidate_count=2,
        candidates=[CandidateRecord(action_type=8),
                    CandidateRecord(action_type=4, tile_type=10,
                                     target_rel_seat=2)],
        player_id=0, episode_id="ep0", round_id=0,
        step_id=step_id, actor_type="policy",
        experiment_id="t", run_id="r", worker_id="w",
    )
    base.update(kw)
    s = DecisionSample(**base)
    s.sample_semantics_version = version
    return s


def _make_learner(tmp_path):
    model = Stage2aModel(input_dim=10, discard_hidden_dims=[8],
                          optional_hidden_dims=[8])
    return Stage2aLearner(
        config={"training": {
            "algorithm": "ppo", "epochs": 1, "batch_size": 4,
        }},
        model=model, run_dir=tmp_path / "run",
        device=torch.device("cpu"),
    )


class TestPpoDiagSmoke:
    """実際に PPO を 1 epoch 通して ppo_diag を見る"""

    def _write_mixed_shard(self, shard_dir, n_d=8, n_c=6):
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        rng = np.random.RandomState(42)
        for i in range(n_d):
            writer.add(_mk_discard(
                step_id=i,
                reward=float(rng.randn() * 0.1),
                value=float(rng.randn() * 0.1),
                terminated=(i == n_d - 1),
            ))
        for i in range(n_c):
            writer.add(_mk_call(
                step_id=n_d + i,
                reward=float(rng.randn() * 0.1),
                value=float(rng.randn() * 0.1),
                terminated=(i == n_c - 1),
            ))
        writer.close()

    def test_ppo_diag_has_new_keys(self, tmp_path):
        shard_dir = tmp_path / "shards"
        self._write_mixed_shard(shard_dir)
        learner = _make_learner(tmp_path)
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        diag = metrics["ppo_diag"]
        # 既存 key
        assert "ratio_mean" in diag
        assert "ratio_std" in diag
        assert "clip_fraction" in diag
        # 新規 top-level
        assert "log_ratio_p50" in diag
        assert "log_ratio_p95" in diag
        assert "advantage_pos_frac" in diag
        assert "advantage_neg_frac" in diag
        assert "max_prob_mean" in diag
        assert "ratio_max" in diag
        # branch 別
        assert "discard" in diag
        assert "call" in diag
        for branch in ("discard", "call"):
            b = diag[branch]
            assert "log_ratio_mean" in b
            assert "log_ratio_p95" in b
            assert "advantage_pos_frac" in b
            assert "clip_fraction" in b
            assert "max_prob_mean" in b
            assert "max_prob_p95" in b

    def test_ppo_diag_json_serializable(self, tmp_path):
        import json
        shard_dir = tmp_path / "shards"
        self._write_mixed_shard(shard_dir)
        learner = _make_learner(tmp_path)
        metrics = learner.train(shard_dir)
        # raise しないこと
        json.dumps(metrics["ppo_diag"])

    def test_ppo_discard_only(self, tmp_path):
        """call サンプル無しでも ppo_diag が動く"""
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        for i in range(8):
            writer.add(_mk_discard(
                step_id=i, reward=0.1, value=0.05,
                terminated=(i == 7),
            ))
        writer.close()
        learner = _make_learner(tmp_path)
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        assert "discard" in diag
        assert "call" not in diag
        assert "log_ratio_p50" in diag

    def test_ppo_existing_keys_still_present(self, tmp_path):
        """既存の ratio_mean / ratio_std / clip_fraction / advantage_mean /
        advantage_std を削除していないこと"""
        shard_dir = tmp_path / "shards"
        self._write_mixed_shard(shard_dir)
        learner = _make_learner(tmp_path)
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        for k in ("ratio_mean", "ratio_std", "clip_fraction",
                  "advantage_mean", "advantage_std"):
            assert k in diag, f"既存 diagnostics key {k} が消えている"


class TestBranchSplitDiag:
    """branch split test: 同じ helper を独立 buffer で呼んで両 branch dict が出る"""

    def test_helper_separates_branches(self):
        rng = np.random.RandomState(0)
        d = Stage2aLearner._compute_ppo_diag_stats(
            rng.randn(20) * 0.1, rng.randn(20),
            np.clip(rng.rand(20), 0, 1), np.ones(20),
            clip_epsilon=0.2)
        c = Stage2aLearner._compute_ppo_diag_stats(
            rng.randn(15) * 0.1, rng.randn(15),
            np.clip(rng.rand(15), 0, 1), np.ones(15),
            clip_epsilon=0.2)
        diag = {}
        diag["discard"] = d
        diag["call"] = c
        assert diag["discard"]["log_ratio_p50"] != diag["call"]["log_ratio_p50"]
        # 両方とも独立に key を持つ
        assert "max_prob_p95" in diag["discard"]
        assert "max_prob_p95" in diag["call"]
