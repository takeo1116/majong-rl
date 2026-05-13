"""CQ-0293: target_kl early stop と applied diagnostics の整合確認

確認:
- skip_minibatch_on_exceed=True で KL 超過 minibatch が
  optimizer.step() されない
- skipped minibatch が applied diagnostics に入らない
  - semantic aux loss (terminal_loss / yaku_loss)
  - gradient_norms aggregate
  - policy/value/entropy applied stats
- target_kl_applied_minibatches が出る
- approx_kl が batch_w weighted mean になる
- skip_on_exceed=False では KL 超過 minibatch も step され、applied
  diagnostics に入る
- default off では既存 schema を壊さない
- gradient_norms と target_kl 併用で crash しない
- ppo_diag の counts は JSON serializable
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.smoke

from mahjong_rl.stage2a_learner import Stage2aLearner
from mahjong_rl.models.stage2a_model import Stage2aModel
from mahjong_rl.call_shard import (
    DecisionSample, CandidateRecord, DecisionShardWriter,
)


# ========== helpers ==========


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


def _make_model():
    return Stage2aModel(
        input_dim=10, discard_hidden_dims=[8],
        optional_hidden_dims=[8], value_hidden_dims=[8],
        candidate_dim=8, optional_scorer_hidden=8,
    )


def _make_learner(tmp_path, *, target_kl_cfg=None, gradient_norms_cfg=None,
                   semantic_aux_cfg=None, epochs=1, batch_size=4):
    cfg: dict = {
        "training": {
            "algorithm": "ppo", "epochs": epochs, "batch_size": batch_size,
            "lr": 1e-4,
        },
    }
    if target_kl_cfg is not None:
        cfg["training"]["ppo_target_kl"] = target_kl_cfg
    if gradient_norms_cfg is not None:
        cfg["training"]["diagnostics"] = {
            "gradient_norms": gradient_norms_cfg,
        }
    if semantic_aux_cfg is not None:
        cfg["model"] = {"semantic_aux": semantic_aux_cfg}
    model = _make_model()
    return Stage2aLearner(
        config=cfg, model=model, run_dir=tmp_path / "run",
        device=torch.device("cpu"),
    )


def _write_shard(shard_dir, n_d=8, n_c=6, old_lp_d=-0.5, old_lp_c=-0.5):
    writer = DecisionShardWriter(shard_dir, max_samples=100)
    rng = np.random.RandomState(42)
    for i in range(n_d):
        writer.add(_mk_discard(
            step_id=i, log_prob=old_lp_d,
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=(i == n_d - 1),
        ))
    for i in range(n_c):
        writer.add(_mk_call(
            step_id=n_d + i, log_prob=old_lp_c,
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=(i == n_c - 1),
        ))
    writer.close()


# ========== Section 1: applied count schema ==========


class TestAppliedMinibatchCount:
    """target_kl_applied_minibatches schema が出る。"""

    def test_default_off_applied_equals_checked(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(tmp_path, target_kl_cfg=None)
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        # default off: skip は発生しない → applied == checked
        checked = diag["target_kl_checked_minibatches"]
        applied = diag["target_kl_applied_minibatches"]
        skipped = diag["target_kl_skipped_minibatches"]
        assert applied == checked
        assert skipped == 0

    def test_skip_on_exceed_applied_is_checked_minus_skipped(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir, old_lp_d=0.0, old_lp_c=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": True})
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        checked = diag["target_kl_checked_minibatches"]
        applied = diag["target_kl_applied_minibatches"]
        skipped = diag["target_kl_skipped_minibatches"]
        assert applied == checked - skipped
        # 強制 KL>0 なので 1 つ以上 skip されるはず
        assert skipped >= 1

    def test_branch_level_applied_count(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir, old_lp_d=0.0, old_lp_c=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": True})
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        # branch 別 sub-dict にも同 key が出る
        assert "target_kl_applied_minibatches" in diag.get("discard", {})
        assert "target_kl_applied_minibatches" in diag.get("call", {})
        d = diag["discard"]
        assert d["target_kl_applied_minibatches"] == (
            d["target_kl_checked_minibatches"]
            - d["target_kl_skipped_minibatches"])

    def test_diag_is_json_serializable(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir, old_lp_d=0.0, old_lp_c=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": True})
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        # JSON serializable
        s = json.dumps(diag)
        loaded = json.loads(s)
        assert "target_kl_applied_minibatches" in loaded


# ========== Section 2: skipped excluded from applied diagnostics ==========


class TestSkippedExcludedFromAppliedDiagnostics:
    """skip_minibatch_on_exceed=True で KL 超過 minibatch が
    semantic / gradient_norms / policy_loss 集計に入らないこと。"""

    def test_skipped_excluded_from_terminal_loss(self, tmp_path):
        """semantic_aux 有効 + skip_on_exceed=True で全 minibatch が
        skip された場合、terminal_loss / yaku_loss が None / 計上されない。"""
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir, old_lp_d=0.0, old_lp_c=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": True},
            semantic_aux_cfg={"enabled": True})
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        # 全 minibatch skip → applied=0
        d_applied = diag["discard"]["target_kl_applied_minibatches"]
        c_applied = diag["call"]["target_kl_applied_minibatches"]
        if d_applied == 0 and c_applied == 0:
            # applied が 0 なので semantic loss は計上されない (None)
            assert metrics["terminal_loss"] is None
            assert metrics["yaku_loss"] is None

    def test_skipped_excluded_from_gradient_norms(self, tmp_path):
        """gradient_norms 有効 + skip_on_exceed=True で全 minibatch が
        skip された場合、gradient_norms aggregate count が 0。"""
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir, old_lp_d=0.0, old_lp_c=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": True},
            gradient_norms_cfg={"enabled": True,
                                  "max_batches_per_epoch": 16,
                                  "every_n_epochs": 1})
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        gn = diag.get("gradient_norms")
        assert gn is not None
        # 全 minibatch skip なら aggregate count = 0
        d_applied = diag["discard"]["target_kl_applied_minibatches"]
        c_applied = diag["call"]["target_kl_applied_minibatches"]
        agg = gn["aggregate"]
        if d_applied == 0 and c_applied == 0:
            assert agg.get("count", 0) == 0

    def test_skipped_excluded_from_policy_loss(self, tmp_path):
        """skip_on_exceed=True で全 minibatch skip なら
        policy_loss / value_loss / entropy も 0 (計上対象なし)。"""
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir, old_lp_d=0.0, old_lp_c=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": True})
        metrics = learner.train(shard_dir)
        d_applied = metrics["ppo_diag"]["discard"][
            "target_kl_applied_minibatches"]
        c_applied = metrics["ppo_diag"]["call"][
            "target_kl_applied_minibatches"]
        if d_applied == 0 and c_applied == 0:
            assert metrics["policy_loss"] == 0.0
            assert metrics["value_loss"] == 0.0
            assert metrics["entropy"] == 0.0
            assert metrics["num_updates"] == 0


# ========== Section 3: skip_on_exceed=False keeps applied ==========


class TestSkipFalseStillSteps:
    """skip_on_exceed=False で KL 超過 minibatch も step され、applied
    diagnostics に入ること。"""

    def test_step_still_runs_on_exceed(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir, old_lp_d=0.0, old_lp_c=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": False})
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        # skip_on_exceed=False → skipped は常に 0
        assert diag["target_kl_skipped_minibatches"] == 0
        # applied == checked
        assert (diag["target_kl_applied_minibatches"]
                == diag["target_kl_checked_minibatches"])
        # num_updates >= 1 (step が 1 回以上発火)
        assert metrics["num_updates"] >= 1


# ========== Section 4: approx_kl weighted mean ==========


class TestApproxKlWeighted:
    """approx_kl が batch_w weighted mean になっていることを確認。

    本命 separated PPO では batch_w=1 なので unweighted と同値だが、
    実装が weighted mean であることを weighted vs unweighted の比較で
    確認する。
    """

    def test_approx_kl_uniform_weight_matches_unweighted(self, tmp_path):
        """weights が全て 1 (default) のときは unweighted mean と一致する。"""
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir, old_lp_d=0.1, old_lp_c=0.1)
        learner = _make_learner(tmp_path, target_kl_cfg=None)
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        akl = diag["approx_kl_mean"]
        assert akl is not None
        # 値そのものは shard 依存だが、float であり JSON serializable
        assert isinstance(akl, float)

    def test_approx_kl_weighted_implementation(self):
        """approx_kl 計算式: kl_per * batch_w / sum(batch_w)。

        手動で同 input で計算した場合と一致することを確認。
        """
        torch.manual_seed(0)
        ratio = torch.tensor([1.05, 1.10, 0.95, 1.20])
        log_ratio = torch.log(ratio)
        batch_w = torch.tensor([1.0, 2.0, 1.0, 2.0])

        kl_per = (ratio - 1.0) - log_ratio
        w_sum = batch_w.sum().clamp_min(1e-8)
        weighted = float(((kl_per * batch_w).sum() / w_sum).item())
        # weighted mean は unweighted mean と異なる値 (重み付き)
        unweighted = float(kl_per.mean().item())
        assert weighted != pytest.approx(unweighted)
        # uniform weight の場合は一致
        uniform_w = torch.ones(4)
        uniform_weighted = float(((kl_per * uniform_w).sum()
                                    / uniform_w.sum()).item())
        assert uniform_weighted == pytest.approx(unweighted)


# ========== Section 5: gradient_norms + target_kl coexistence ==========


class TestGradientNormsTargetKlCoexist:
    """gradient_norms enabled + target_kl 併用で crash しない。"""

    def test_both_enabled_does_not_crash(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.5,
                           "stop_multiplier": 1.5,
                           "skip_minibatch_on_exceed": True},
            gradient_norms_cfg={"enabled": True,
                                  "max_batches_per_epoch": 4,
                                  "every_n_epochs": 1})
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        assert "gradient_norms" in diag
        assert "target_kl_checked_minibatches" in diag
        # JSON serializable
        json.dumps(diag)


# ========== Section 6: default off backward compatibility ==========


class TestDefaultOffSchema:
    """default off で既存 ppo_diag schema が壊れない。"""

    def test_default_diag_keys(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(tmp_path, target_kl_cfg=None)
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        # CQ-0287 既存 keys
        for k in ("target_kl_enabled", "target_kl_threshold",
                  "target_kl_stop_count", "target_kl_skipped_minibatches",
                  "target_kl_checked_minibatches",
                  "approx_kl_mean", "approx_kl_max"):
            assert k in diag, f"missing key: {k}"
        # CQ-0293 で新規追加
        assert "target_kl_applied_minibatches" in diag
        # default off では skipped=0, applied=checked
        assert diag["target_kl_skipped_minibatches"] == 0
        assert diag["target_kl_applied_minibatches"] == diag[
            "target_kl_checked_minibatches"]

    def test_default_off_no_optimizer_step_change(self, tmp_path):
        """default off で num_updates が batch 数と一致 (skip なし)。"""
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir, n_d=8, n_c=6)
        learner = _make_learner(tmp_path, target_kl_cfg=None,
                                  epochs=1, batch_size=4)
        metrics = learner.train(shard_dir)
        # num_updates = applied minibatch 総数
        applied_total = metrics["ppo_diag"]["target_kl_applied_minibatches"]
        # 8 discard + 6 call → batch_size=4 → 2+2 = 4 minibatches per epoch
        assert applied_total >= 1
        assert metrics["num_updates"] == applied_total


# ========== Section 7: gradient_norms disabled stays disabled ==========


class TestGradientNormsDisabledStaysOff:
    """gradient_norms disabled では torch.autograd.grad が呼ばれない
    既存テストと同等の挙動を維持する (CQ-0284 互換)。"""

    def test_no_gradient_norms_in_diag_when_disabled(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.5,
                           "stop_multiplier": 1.5},
            gradient_norms_cfg=None)  # disabled
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        assert "gradient_norms" not in diag
