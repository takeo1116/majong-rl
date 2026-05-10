"""CQ-0287: Stage2a PPO target_kl early stop

確認:
- default off (`ppo_target_kl.enabled=false` または未指定) で既存と完全互換
  (optimizer step 回数が変わらない)
- forced high KL で early stop が発火し、その minibatch では
  `optimizer.step()` が呼ばれない (`skip_minibatch_on_exceed=true`)
- `skip_minibatch_on_exceed=false` では step してから break
- discard branch / call branch の両方で early stop が動く
- branch 片側空 (discard-only / call-only) で crash しない
- `ppo_diag` に approx_kl / stop_count / skipped / checked / threshold /
  enabled が JSON serializable に出る
- gradient_norms と併用しても crash しない
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


def _make_learner(tmp_path, *, target_kl_cfg: dict | None = None,
                   gradient_norms_cfg: dict | None = None,
                   epochs: int = 1, batch_size: int = 4):
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
    model = _make_model()
    return Stage2aLearner(
        config=cfg, model=model, run_dir=tmp_path / "run",
        device=torch.device("cpu"),
    )


def _write_shard(shard_dir, n_d=8, n_c=6):
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


def _write_shard_with_old_log_prob(shard_dir, n_d=8, n_c=6, *,
                                     old_lp_d=-0.5, old_lp_c=-0.5):
    """old_log_prob を強制指定して approx KL の方向を制御するための shard"""
    writer = DecisionShardWriter(shard_dir, max_samples=100)
    rng = np.random.RandomState(42)
    for i in range(n_d):
        writer.add(_mk_discard(
            step_id=i,
            log_prob=old_lp_d,
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=(i == n_d - 1),
        ))
    for i in range(n_c):
        writer.add(_mk_call(
            step_id=n_d + i,
            log_prob=old_lp_c,
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=(i == n_c - 1),
        ))
    writer.close()


# ========== 1. default off (既存互換) ==========


class TestDefaultOff:
    """default off で既存挙動が変わらないこと"""

    def test_no_target_kl_key_means_disabled(self, tmp_path):
        """training.ppo_target_kl 未指定で disabled として動作"""
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(tmp_path, target_kl_cfg=None)
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        assert diag["target_kl_enabled"] is False
        assert diag["target_kl_stop_count"] == 0
        assert diag["target_kl_skipped_minibatches"] == 0

    def test_disabled_step_count_unchanged(self, tmp_path, monkeypatch):
        """default off で optimizer.step() 呼び出し回数が既存通り"""
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(tmp_path, target_kl_cfg=None)
        # optimizer.step を spy
        original_step = learner._optimizer.step
        call_count = {"n": 0}

        def _spy_step(*args, **kwargs):
            call_count["n"] += 1
            return original_step(*args, **kwargs)

        monkeypatch.setattr(learner._optimizer, "step", _spy_step)
        learner.train(shard_dir)
        # n_d=8, batch_size=4 → 2 mb (discard) + n_c=6, batch=4 → 2 mb (call)
        # = 4 step
        assert call_count["n"] == 4, (
            f"expected 4 optimizer steps, got {call_count['n']}")

    def test_disabled_explicit_false_step_count_unchanged(self, tmp_path,
                                                          monkeypatch):
        """ppo_target_kl.enabled=False でも同様"""
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": False, "target": 0.0001,
                           "stop_multiplier": 1.5})
        original_step = learner._optimizer.step
        call_count = {"n": 0}

        def _spy_step(*args, **kwargs):
            call_count["n"] += 1
            return original_step(*args, **kwargs)

        monkeypatch.setattr(learner._optimizer, "step", _spy_step)
        learner.train(shard_dir)
        assert call_count["n"] == 4

    def test_diag_present_default_off(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(tmp_path, target_kl_cfg=None)
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        for k in ("target_kl_enabled", "target_kl",
                  "target_kl_threshold",
                  "target_kl_skip_minibatch_on_exceed",
                  "target_kl_stop_count",
                  "target_kl_skipped_minibatches",
                  "target_kl_checked_minibatches",
                  "approx_kl_mean", "approx_kl_max"):
            assert k in diag, f"missing top-level key {k}"
        # branch 別にも入る (discard/call 両方 sample あり)
        for branch in ("discard", "call"):
            assert "target_kl_enabled" in diag[branch]
            assert "target_kl_checked_minibatches" in diag[branch]


# ========== 2. enabled forces early stop ==========


class TestEnabledEarlyStop:
    """approx KL > threshold で early stop が発火"""

    def test_low_threshold_skips_minibatch(self, tmp_path, monkeypatch):
        """target=0 / stop_multiplier=1 → threshold=0、ほぼ全 minibatch
        が skip → step 数が 0 になるべき (skip_minibatch_on_exceed=true)"""
        shard_dir = tmp_path / "shards"
        # old_log_prob を 0.0 に統一して、初期 log_softmax 結果との差分を
        # 大きく取らせる (action_log_prob は -log(34) などなので、
        # ratio=exp(action_lp - 0) ≈ 0 → approx_kl=((ratio-1) - log_ratio).mean()
        # が正に大きく出る)
        _write_shard_with_old_log_prob(shard_dir, n_d=8, n_c=6,
                                         old_lp_d=0.0, old_lp_c=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True,
                           "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": True})
        original_step = learner._optimizer.step
        call_count = {"n": 0}

        def _spy_step(*args, **kwargs):
            call_count["n"] += 1
            return original_step(*args, **kwargs)

        monkeypatch.setattr(learner._optimizer, "step", _spy_step)
        metrics = learner.train(shard_dir)
        # threshold=0 でほぼ確実に超過 → 各 branch の最初の minibatch で
        # 即 skip + break。step は呼ばれないはず (= 0)
        assert call_count["n"] == 0, (
            f"expected no step calls when threshold=0 with skip=true, "
            f"got {call_count['n']}")
        diag = metrics["ppo_diag"]
        assert diag["target_kl_enabled"] is True
        assert diag["target_kl_stop_count"] >= 1
        assert diag["target_kl_skipped_minibatches"] >= 1

    def test_high_threshold_no_skip(self, tmp_path, monkeypatch):
        """threshold が極端に大きいと early stop しない (= 既存通り step)"""
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True,
                           "target": 1e6,
                           "stop_multiplier": 1.5,
                           "skip_minibatch_on_exceed": True})
        original_step = learner._optimizer.step
        call_count = {"n": 0}

        def _spy_step(*args, **kwargs):
            call_count["n"] += 1
            return original_step(*args, **kwargs)

        monkeypatch.setattr(learner._optimizer, "step", _spy_step)
        metrics = learner.train(shard_dir)
        assert call_count["n"] == 4
        diag = metrics["ppo_diag"]
        assert diag["target_kl_stop_count"] == 0
        assert diag["target_kl_skipped_minibatches"] == 0

    def test_skip_false_does_step_then_break(self, tmp_path, monkeypatch):
        """skip_minibatch_on_exceed=False で step 1 回 (各 branch 最初の mb)
        してから break する"""
        shard_dir = tmp_path / "shards"
        _write_shard_with_old_log_prob(shard_dir, n_d=8, n_c=6,
                                         old_lp_d=0.0, old_lp_c=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True,
                           "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": False})
        original_step = learner._optimizer.step
        call_count = {"n": 0}

        def _spy_step(*args, **kwargs):
            call_count["n"] += 1
            return original_step(*args, **kwargs)

        monkeypatch.setattr(learner._optimizer, "step", _spy_step)
        metrics = learner.train(shard_dir)
        # 各 branch で 1 minibatch だけ step してから break → 計 2 step
        assert call_count["n"] == 2
        diag = metrics["ppo_diag"]
        assert diag["target_kl_stop_count"] >= 1
        # skip=False のため skipped は 0 のまま
        assert diag["target_kl_skipped_minibatches"] == 0


# ========== 3. branch coverage ==========


class TestBranchCoverage:
    """discard / call branch 両方で early stop"""

    def test_discard_only_branch(self, tmp_path, monkeypatch):
        shard_dir = tmp_path / "shards"
        # call 0 件 + discard 多め
        _write_shard_with_old_log_prob(shard_dir, n_d=8, n_c=0,
                                         old_lp_d=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": True})
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        assert diag["target_kl_stop_count"] >= 1
        # discard branch にだけ stats
        assert "discard" in diag
        # call branch は無いか checked=0
        if "call" in diag:
            assert diag["call"]["target_kl_checked_minibatches"] == 0

    def test_call_only_branch(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard_with_old_log_prob(shard_dir, n_d=0, n_c=6,
                                         old_lp_c=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": True})
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        assert diag["target_kl_stop_count"] >= 1
        assert "call" in diag
        if "discard" in diag:
            assert diag["discard"]["target_kl_checked_minibatches"] == 0

    def test_both_branches_can_stop(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard_with_old_log_prob(shard_dir, n_d=8, n_c=6,
                                         old_lp_d=0.0, old_lp_c=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": True})
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        # 1 epoch なので両 branch でそれぞれ 1 回ずつ stop が起きるはず
        assert diag["discard"]["target_kl_stop_count"] >= 1
        assert diag["call"]["target_kl_stop_count"] >= 1
        # top-level は両 branch の合計
        assert diag["target_kl_stop_count"] >= 2


# ========== 4. schema / serializability ==========


class TestSchemaSerializable:
    def test_json_serializable_disabled(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(tmp_path, target_kl_cfg=None)
        metrics = learner.train(shard_dir)
        # crash しないこと
        json.dumps(metrics["ppo_diag"])

    def test_json_serializable_enabled(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard_with_old_log_prob(shard_dir, n_d=8, n_c=6,
                                         old_lp_d=0.0, old_lp_c=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": True})
        metrics = learner.train(shard_dir)
        # crash しないこと
        json.dumps(metrics["ppo_diag"])
        diag = metrics["ppo_diag"]
        # 必須 key
        for k in ("target_kl_enabled", "target_kl",
                  "target_kl_threshold",
                  "target_kl_skip_minibatch_on_exceed",
                  "target_kl_stop_count",
                  "target_kl_skipped_minibatches",
                  "target_kl_checked_minibatches",
                  "approx_kl_mean", "approx_kl_max"):
            assert k in diag

    def test_threshold_value(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.04,
                           "stop_multiplier": 2.0,
                           "skip_minibatch_on_exceed": True})
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        assert diag["target_kl"] == pytest.approx(0.04)
        assert diag["target_kl_threshold"] == pytest.approx(0.08)
        assert diag["target_kl_skip_minibatch_on_exceed"] is True


# ========== 5. smoke + gradient_norms 併用 ==========


class TestSmokeWithGradientNorms:
    def test_smoke_target_kl_enabled(self, tmp_path):
        """通常 threshold で smoke 完走"""
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.03,
                           "stop_multiplier": 1.5,
                           "skip_minibatch_on_exceed": True})
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        assert metrics["num_updates"] > 0

    def test_smoke_target_kl_with_gradient_norms(self, tmp_path):
        """target_kl + gradient_norms の併用 smoke"""
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.03,
                           "stop_multiplier": 1.5,
                           "skip_minibatch_on_exceed": True},
            gradient_norms_cfg={"enabled": True,
                                 "max_batches_per_epoch": 2})
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        diag = metrics["ppo_diag"]
        assert "gradient_norms" in diag
        assert "target_kl_enabled" in diag

    def test_smoke_high_kl_with_gradient_norms(self, tmp_path):
        """high KL → skip → gradient_norms 計測も skip 前 forward まで OK"""
        shard_dir = tmp_path / "shards"
        _write_shard_with_old_log_prob(shard_dir, n_d=8, n_c=6,
                                         old_lp_d=0.0, old_lp_c=0.0)
        learner = _make_learner(
            tmp_path,
            target_kl_cfg={"enabled": True, "target": 0.0,
                           "stop_multiplier": 1.0,
                           "skip_minibatch_on_exceed": True},
            gradient_norms_cfg={"enabled": True,
                                 "max_batches_per_epoch": 4})
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        # crash せず stats が出ること
        assert metrics["ppo_diag"]["target_kl_stop_count"] >= 1


# ========== 6. approx_kl values ==========


class TestApproxKlValues:
    def test_approx_kl_recorded(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir)
        learner = _make_learner(tmp_path, target_kl_cfg=None)
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        assert diag["target_kl_checked_minibatches"] > 0
        # approx_kl_mean / max は finite (大きく崩れていない初期 model)
        assert diag["approx_kl_mean"] is not None
        assert diag["approx_kl_max"] is not None
        assert np.isfinite(diag["approx_kl_mean"])
        assert np.isfinite(diag["approx_kl_max"])

    def test_max_ge_mean(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shard(shard_dir, n_d=12, n_c=8)
        learner = _make_learner(tmp_path, target_kl_cfg=None,
                                 batch_size=4)
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        assert diag["approx_kl_max"] >= diag["approx_kl_mean"] - 1e-9
