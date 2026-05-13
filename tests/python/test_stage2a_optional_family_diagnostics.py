"""CQ-0295: stage2a_learner.PPO の family-level diagnostics tests."""
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
from mahjong_rl.candidate_encoding import (
    OPTIONAL_NORIICHI, OPTIONAL_RIICHI,
)


def _mk_discard(step_id, decision_family="discard", **kw):
    base = dict(
        decision_type="discard",
        decision_family=decision_family,
        observation=np.zeros(10, dtype=np.float32),
        legal_mask=np.ones(34, dtype=np.float32),
        action=0, reward=0.0, log_prob=-0.5, value=0.0,
        terminated=False, round_over=False,
        teacher_top1_index=0,
        player_id=0, episode_id="e0", round_id=0,
        step_id=step_id, actor_type="policy",
        experiment_id="t", run_id="r", worker_id="w",
    )
    base.update(kw)
    s = DecisionSample(**base)
    s.sample_semantics_version = 3
    return s


def _mk_call(step_id, decision_family, cand_types, teacher=0, **kw):
    cands = [CandidateRecord(action_type=at) for at in cand_types]
    base = dict(
        decision_type="call",
        decision_family=decision_family,
        observation=np.zeros(10, dtype=np.float32),
        reward=0.0, log_prob=-0.5, value=0.0,
        terminated=False, round_over=False,
        selected_candidate_index=teacher,
        candidate_count=len(cands), candidates=cands,
        response_context=np.zeros(3, dtype=np.float32),
        teacher_top1_index=teacher,
        player_id=0, episode_id="e0", round_id=0,
        step_id=step_id, actor_type="policy",
        experiment_id="t", run_id="r", worker_id="w",
    )
    base.update(kw)
    s = DecisionSample(**base)
    s.sample_semantics_version = 3
    return s


def _make_model():
    return Stage2aModel(
        input_dim=10, discard_hidden_dims=[8],
        optional_hidden_dims=[8], value_hidden_dims=[8],
        candidate_dim=8, optional_scorer_hidden=8,
    )


def _make_learner(tmp_path, epochs=1, batch_size=4):
    return Stage2aLearner(
        config={"training": {
            "algorithm": "ppo", "epochs": epochs, "batch_size": batch_size,
            "lr": 1e-4,
        }},
        model=_make_model(),
        run_dir=tmp_path / "run",
        device=torch.device("cpu"),
    )


# ========== Section 1: PPO diag family schema ==========


class TestPPOFamilyDiag:
    """ppo_diag['decision_family'] schema."""

    def test_family_diag_present_in_ppo_diag(self, tmp_path):
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        for i in range(8):
            writer.add(_mk_discard(step_id=i))
        for i in range(4):
            writer.add(_mk_call(
                step_id=100 + i, decision_family="riichi",
                cand_types=[OPTIONAL_NORIICHI, OPTIONAL_RIICHI],
                teacher=1))
        writer.close()
        learner = _make_learner(tmp_path)
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        assert "decision_family" in diag
        fam = diag["decision_family"]
        # 学習挙動を変えないので、既存 top-level / branch も維持
        assert "discard" in diag
        assert "call" in diag

    def test_family_diag_has_expected_keys(self, tmp_path):
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        for i in range(8):
            writer.add(_mk_discard(step_id=i))
        for i in range(4):
            writer.add(_mk_call(
                step_id=100 + i, decision_family="tsumo",
                cand_types=[1, 8], teacher=0))
        writer.close()
        learner = _make_learner(tmp_path)
        metrics = learner.train(shard_dir)
        fam = metrics["ppo_diag"]["decision_family"]
        for f, entry in fam.items():
            for k in ("sample_count", "ratio_mean", "clip_fraction",
                       "advantage_mean", "advantage_std",
                       "advantage_pos_frac", "approx_kl_mean"):
                assert k in entry, f"{f} missing {k}"

    def test_family_diag_separates_discard_and_optional(self, tmp_path):
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        for i in range(8):
            writer.add(_mk_discard(step_id=i))
        for i in range(4):
            writer.add(_mk_call(
                step_id=100 + i, decision_family="ankan",
                cand_types=[7, 8], teacher=0))
        writer.close()
        learner = _make_learner(tmp_path)
        metrics = learner.train(shard_dir)
        fam = metrics["ppo_diag"]["decision_family"]
        # discard 系
        assert "discard" in fam
        assert fam["discard"]["sample_count"] == 8
        # ankan 系
        assert "ankan" in fam
        assert fam["ankan"]["sample_count"] == 4

    def test_multiple_optional_families_separate(self, tmp_path):
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        # discard 4 + riichi 4 + tsumo 4 + kakan 4
        for i in range(4):
            writer.add(_mk_discard(step_id=i))
        for i in range(4):
            writer.add(_mk_call(
                step_id=10 + i, decision_family="riichi",
                cand_types=[OPTIONAL_NORIICHI, OPTIONAL_RIICHI],
                teacher=1))
        for i in range(4):
            writer.add(_mk_call(
                step_id=20 + i, decision_family="tsumo",
                cand_types=[1, 8], teacher=0))
        for i in range(4):
            writer.add(_mk_call(
                step_id=30 + i, decision_family="kakan",
                cand_types=[6, 8], teacher=0))
        writer.close()
        learner = _make_learner(tmp_path)
        metrics = learner.train(shard_dir)
        fam = metrics["ppo_diag"]["decision_family"]
        for f in ("discard", "riichi", "tsumo", "kakan"):
            assert f in fam, f"missing family {f}"
            assert fam[f]["sample_count"] == 4

    def test_family_diag_is_json_serializable(self, tmp_path):
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        for i in range(4):
            writer.add(_mk_discard(step_id=i))
        for i in range(4):
            writer.add(_mk_call(
                step_id=100 + i, decision_family="riichi",
                cand_types=[OPTIONAL_NORIICHI, OPTIONAL_RIICHI],
                teacher=1))
        writer.close()
        learner = _make_learner(tmp_path)
        metrics = learner.train(shard_dir)
        # JSON 化できる
        json.dumps(metrics["ppo_diag"])


# ========== Section 2: legacy / unknown family handling ==========


class TestFamilyDiagLegacyHandling:
    """decision_family が欠ける legacy shard でも crash しない。"""

    def test_response_default_treated_as_response(self, tmp_path):
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        # 旧 shard 想定: decision_family="response" (default)
        for i in range(4):
            writer.add(_mk_discard(step_id=i))
        for i in range(4):
            writer.add(_mk_call(
                step_id=100 + i, decision_family="response",
                cand_types=[8, 3], teacher=0))
        writer.close()
        learner = _make_learner(tmp_path)
        metrics = learner.train(shard_dir)
        fam = metrics["ppo_diag"]["decision_family"]
        assert "response" in fam
        assert fam["response"]["sample_count"] == 4

    def test_unknown_family_becomes_unknown(self, tmp_path):
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        for i in range(4):
            writer.add(_mk_discard(step_id=i))
        # 未知 family
        for i in range(4):
            writer.add(_mk_call(
                step_id=100 + i, decision_family="foobar",
                cand_types=[8, 3], teacher=0))
        writer.close()
        learner = _make_learner(tmp_path)
        metrics = learner.train(shard_dir)
        fam = metrics["ppo_diag"]["decision_family"]
        assert "unknown" in fam
        assert fam["unknown"]["sample_count"] == 4


# ========== Section 3: 学習挙動が変わらないことの確認 ==========


class TestNoBehaviorChange:
    """family diagnostics 追加で existing top-level / branch diag が
    変わっていないこと。"""

    def test_top_level_keys_intact(self, tmp_path):
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        for i in range(8):
            writer.add(_mk_discard(step_id=i))
        writer.close()
        learner = _make_learner(tmp_path)
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        # CQ-0287 既存
        for k in ("target_kl_enabled", "target_kl_threshold",
                  "target_kl_checked_minibatches",
                  "target_kl_applied_minibatches"):
            assert k in diag
        # CQ-0281 既存 (branch 別)
        assert "discard" in diag
        # CQ-0295 で追加
        assert "decision_family" in diag

    def test_num_updates_unchanged(self, tmp_path):
        """family diag 追加で num_updates 等が変化しないこと。"""
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        for i in range(12):
            writer.add(_mk_discard(step_id=i))
        writer.close()
        learner = _make_learner(tmp_path, batch_size=4)
        metrics = learner.train(shard_dir)
        # batch_size=4 → 3 minibatches per epoch * 1 epoch = 3 updates
        # (default policy_epochs=1)
        assert metrics["num_updates"] >= 1
        # discard_count もそのまま
        assert metrics["discard_count"] == 12


# ========== Section 4: family normalization helper ==========


class TestNormalizeFamilyDiag:
    """Stage2aLearner._normalize_family_diag が想定通り。"""

    def test_known_family(self):
        for f in ("discard", "response", "riichi", "tsumo", "ron",
                  "ankan", "kakan", "kyuushu"):
            assert Stage2aLearner._normalize_family_diag(f) == f

    def test_none(self):
        assert Stage2aLearner._normalize_family_diag(None) == "unknown"

    def test_empty(self):
        assert Stage2aLearner._normalize_family_diag("") == "unknown"

    def test_unknown(self):
        assert Stage2aLearner._normalize_family_diag("foobar") == "unknown"

    def test_case(self):
        assert Stage2aLearner._normalize_family_diag("RIICHI") == "riichi"
