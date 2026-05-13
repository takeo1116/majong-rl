"""CQ-0295: optional decision family offline audit unit tests.

`python/mahjong_rl/optional_family_audit.py` の挙動を確認する。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.smoke

from mahjong_rl.optional_family_audit import (
    SUPPORTED_FAMILIES,
    BINARY_POSITIVE_INDEX,
    audit_samples,
    audit_shard_dir,
    normalize_family,
    format_summary,
)
from mahjong_rl.call_shard import (
    DecisionSample,
    DecisionShardWriter,
    CandidateRecord,
)
from mahjong_rl.candidate_encoding import (
    NUM_ACTION_TYPE_INDICES,
    OPTIONAL_NORIICHI,
    OPTIONAL_RIICHI,
)


# ========== helpers ==========


def _mk_discard(step_id=0, action=0, family="discard", teacher=0,
                best=()):
    s = DecisionSample(
        decision_type="discard",
        decision_family=family,
        observation=np.zeros(10, dtype=np.float32),
        legal_mask=np.ones(34, dtype=np.float32),
        action=action,
        reward=0.0, log_prob=-0.5, value=0.0,
        terminated=False, round_over=False,
        teacher_top1_index=teacher,
        teacher_best_indices=list(best),
        teacher_source="rule_based",
        player_id=0, episode_id="e0", round_id=0,
        step_id=step_id, actor_type="policy",
        experiment_id="t", run_id="r", worker_id="w",
    )
    s.sample_semantics_version = 3
    return s


def _mk_optional(family, teacher_idx, cand_types,
                  step_id=0):
    cands = [
        CandidateRecord(action_type=at, tile_type=0,
                         target_rel_seat=-1)
        for at in cand_types
    ]
    s = DecisionSample(
        decision_type="call",
        decision_family=family,
        observation=np.zeros(10, dtype=np.float32),
        reward=0.0, log_prob=-0.5, value=0.0,
        terminated=False, round_over=False,
        selected_candidate_index=teacher_idx,
        candidate_count=len(cands),
        candidates=cands,
        response_context=np.zeros(3, dtype=np.float32),
        teacher_top1_index=teacher_idx,
        teacher_source="auto",
        player_id=0, episode_id="e0", round_id=0,
        step_id=step_id, actor_type="policy",
        experiment_id="t", run_id="r", worker_id="w",
    )
    s.sample_semantics_version = 3
    return s


# ========== Section 1: normalize_family ==========


class TestNormalizeFamily:
    def test_known_families_pass_through(self):
        for fam in SUPPORTED_FAMILIES:
            assert normalize_family(fam) == fam

    def test_none_returns_unknown(self):
        assert normalize_family(None) == "unknown"

    def test_empty_returns_unknown(self):
        assert normalize_family("") == "unknown"

    def test_unknown_value_returns_unknown(self):
        assert normalize_family("foobar") == "unknown"

    def test_case_insensitive(self):
        assert normalize_family("RIICHI") == "riichi"


# ========== Section 2: audit_samples without model ==========


class TestAuditSamplesNoModel:
    """model なしでも sample-level 集計が動く。"""

    def test_empty_samples(self):
        result = audit_samples([])
        assert result["total_samples"] == 0
        for fam in SUPPORTED_FAMILIES:
            assert result["families"][fam]["sample_count"] == 0

    def test_discard_family_count(self):
        samples = [_mk_discard(step_id=i, action=i % 4) for i in range(10)]
        result = audit_samples(samples)
        assert result["total_samples"] == 10
        assert result["families"]["discard"]["sample_count"] == 10

    def test_riichi_binary_positive_rate(self):
        # 5 Riichi (idx=1), 3 NoRiichi (idx=0)
        samples = []
        for _ in range(5):
            samples.append(_mk_optional(
                "riichi", 1,
                [OPTIONAL_NORIICHI, OPTIONAL_RIICHI]))
        for _ in range(3):
            samples.append(_mk_optional(
                "riichi", 0,
                [OPTIONAL_NORIICHI, OPTIONAL_RIICHI]))
        result = audit_samples(samples)
        info = result["families"]["riichi"]
        assert info["sample_count"] == 8
        assert info["binary_positive"]["positive_index"] == 1
        assert info["binary_positive"]["positive_count"] == 5
        assert info["binary_positive"]["positive_rate"] == pytest.approx(5 / 8)

    def test_tsumo_binary_positive_rate(self):
        # 2 TsumoWin (idx=0), 5 Skip (idx=1) — positive=0
        # engine ActionType: 1=TsumoWin, 8=Skip
        samples = []
        for _ in range(2):
            samples.append(_mk_optional("tsumo", 0, [1, 8]))
        for _ in range(5):
            samples.append(_mk_optional("tsumo", 1, [1, 8]))
        result = audit_samples(samples)
        info = result["families"]["tsumo"]
        assert info["binary_positive"]["positive_index"] == 0
        assert info["binary_positive"]["positive_count"] == 2
        assert info["binary_positive"]["positive_rate"] == pytest.approx(2 / 7)

    def test_kan_kyuushu_families_present(self):
        # ankan, kakan, kyuushu それぞれ
        from mahjong_rl._mahjong_core import ActionType  # noqa: F401
        # engine action types: 6=Kakan, 7=Ankan, 9=Kyuushu, 8=Skip
        samples = [
            _mk_optional("ankan", 0, [7, 8]),
            _mk_optional("ankan", 1, [7, 8]),
            _mk_optional("kakan", 0, [6, 8]),
            _mk_optional("kyuushu", 1, [9, 8]),
        ]
        result = audit_samples(samples)
        assert result["families"]["ankan"]["sample_count"] == 2
        assert result["families"]["kakan"]["sample_count"] == 1
        assert result["families"]["kyuushu"]["sample_count"] == 1
        # ankan positive_index=0
        ank = result["families"]["ankan"]
        assert ank["binary_positive"]["positive_index"] == 0
        assert ank["binary_positive"]["positive_count"] == 1

    def test_teacher_top1_distribution(self):
        samples = [
            _mk_discard(action=0, teacher=0),
            _mk_discard(action=0, teacher=0),
            _mk_discard(action=5, teacher=5),
        ]
        result = audit_samples(samples)
        td = result["families"]["discard"]["teacher_top1_distribution"]
        assert {e["index"] for e in td} == {0, 5}
        # 0 が 2 回、5 が 1 回 (順は count 降順)
        assert td[0]["index"] == 0
        assert td[0]["count"] == 2

    def test_best_set_size_stats(self):
        # best-set のある sample
        samples = [
            _mk_discard(action=0, teacher=0, best=[0, 1, 2]),
            _mk_discard(action=0, teacher=3, best=[0, 1]),
            _mk_discard(action=0, teacher=-1),
        ]
        result = audit_samples(samples)
        bs = result["families"]["discard"].get("teacher_best_set_size")
        assert bs is not None
        assert bs["count"] == 2
        assert bs["max"] == 3
        # top1 ∈ best-set 率: sample 0 (0 in [0,1,2]) ok, sample 1 (3 in [0,1]) ng
        inb = result["families"]["discard"]["teacher_top1_in_best_set"]
        assert inb["count"] == 1
        assert inb["total"] == 2


# ========== Section 3: decision_family missing fallback ==========


class TestMissingDecisionFamily:
    """decision_family が無い (legacy "response" default) sample を crash
    せず扱う。"""

    def test_call_with_response_default(self):
        # legacy 形式: discard_type=call + decision_family="response"
        s = _mk_optional("response", 0, [8, 1])  # Skip vs Chi
        result = audit_samples([s])
        assert result["families"]["response"]["sample_count"] == 1

    def test_unknown_family_string(self):
        s = _mk_optional("foobar", 0, [1, 8])
        # __post_init__ なし dataclass なので異常文字列もそのまま入る
        result = audit_samples([s])
        # "foobar" は SUPPORTED_FAMILIES に無いので unknown へ
        assert result["families"]["unknown"]["sample_count"] == 1

    def test_discard_decision_type_forces_discard_family(self):
        # decision_family を間違って "response" にしても decision_type=
        # "discard" なら "discard" として集計する
        s = _mk_discard(family="response")
        result = audit_samples([s])
        assert result["families"]["discard"]["sample_count"] == 1
        assert result["families"]["response"]["sample_count"] == 0


# ========== Section 4: shard round-trip ==========


class TestAuditShardDir:
    def test_audit_shard_dir_no_model(self, tmp_path):
        # 簡単 shard を書き出して読み込む
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        for i in range(3):
            writer.add(_mk_discard(step_id=i, action=i))
        writer.add(_mk_optional("riichi", 1,
                                  [OPTIONAL_NORIICHI, OPTIONAL_RIICHI]))
        writer.close()

        result = audit_shard_dir(tmp_path)
        assert result["total_samples"] == 4
        assert result["families"]["discard"]["sample_count"] == 3
        assert result["families"]["riichi"]["sample_count"] == 1

    def test_audit_shard_dir_max_samples(self, tmp_path):
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        for i in range(10):
            writer.add(_mk_discard(step_id=i, action=i % 4))
        writer.close()
        result = audit_shard_dir(tmp_path, max_samples=3)
        assert result["total_samples"] == 3


# ========== Section 5: with model (policy forward) ==========


def _make_model(input_dim=10):
    from mahjong_rl.models.stage2a_model import Stage2aModel
    return Stage2aModel(
        input_dim=input_dim, discard_hidden_dims=[8],
        optional_hidden_dims=[8], value_hidden_dims=[8],
        candidate_dim=8, optional_scorer_hidden=8,
    )


class TestAuditSamplesWithModel:
    """model 付きで policy forward が動く。"""

    def test_discard_policy_stats_present(self):
        model = _make_model()
        samples = [_mk_discard(action=0, teacher=0) for _ in range(4)]
        result = audit_samples(samples, model=model)
        pol = result["families"]["discard"]["policy"]
        assert pol["policy_evaluated"] == 4
        for k in ("entropy", "max_prob", "teacher_action_prob"):
            assert k in pol
            assert "mean" in pol[k]
        # agreement は 0..4
        assert 0 <= pol["teacher_top1_agreement_count"] <= 4

    def test_optional_policy_stats_present(self):
        model = _make_model()
        samples = [
            _mk_optional("riichi", 1,
                          [OPTIONAL_NORIICHI, OPTIONAL_RIICHI])
            for _ in range(3)
        ]
        result = audit_samples(samples, model=model)
        pol = result["families"]["riichi"]["policy"]
        assert pol["policy_evaluated"] == 3
        assert "entropy" in pol
        assert "max_prob" in pol
        # max_prob.p99 が出る (cycle ごとの安定性のため)
        assert "p99" in pol["max_prob"]

    def test_audit_is_json_serializable(self):
        model = _make_model()
        samples = [
            _mk_discard(action=0, teacher=0),
            _mk_optional("riichi", 1,
                          [OPTIONAL_NORIICHI, OPTIONAL_RIICHI]),
            _mk_optional("tsumo", 0, [1, 8]),
        ]
        result = audit_samples(samples, model=model)
        # JSON serializable
        s = json.dumps(result)
        loaded = json.loads(s)
        assert loaded["total_samples"] == 3


# ========== Section 6: format_summary ==========


class TestFormatSummary:
    def test_summary_text_contains_families(self):
        samples = [
            _mk_discard(action=0, teacher=0),
            _mk_optional("riichi", 1,
                          [OPTIONAL_NORIICHI, OPTIONAL_RIICHI]),
        ]
        result = audit_samples(samples)
        text = format_summary(result, label="t")
        assert "family: discard" in text
        assert "family: riichi" in text
        assert "total_samples: 2" in text

    def test_empty_summary(self):
        result = audit_samples([])
        text = format_summary(result)
        assert "total_samples: 0" in text


# ========== Section 7: binary positive index correctness ==========


class TestBinaryPositiveIndex:
    def test_known_families(self):
        assert BINARY_POSITIVE_INDEX["riichi"] == 1
        assert BINARY_POSITIVE_INDEX["tsumo"] == 0
        assert BINARY_POSITIVE_INDEX["ron"] == 0
        assert BINARY_POSITIVE_INDEX["ankan"] == 0
        assert BINARY_POSITIVE_INDEX["kakan"] == 0
        assert BINARY_POSITIVE_INDEX["kyuushu"] == 0

    def test_response_is_not_binary(self):
        assert "response" not in BINARY_POSITIVE_INDEX

    def test_discard_is_not_binary(self):
        assert "discard" not in BINARY_POSITIVE_INDEX
