"""CQ-0260: encoder / evaluator の open-hand shanten hint 整合テスト"""
import pytest
import numpy as np

pytestmark = pytest.mark.smoke

from mahjong_rl.encoders.flat_encoder import FlatFeatureEncoder
from mahjong_rl.baseline.shanten import compute_shanten
from mahjong_rl._mahjong_core import (
    GameEngine, EnvironmentState, RunMode, Phase,
    make_full_observation, make_partial_observation,
    compute_shanten as compute_shanten_cpp,
    analyze_discards as analyze_discards_cpp,
)


# ========== helpers ==========

def _counts(*args):
    c = [0] * 34
    for tile, cnt in args:
        c[tile] = cnt
    return c


# ========== current_shanten ==========

class TestCurrentShantenOpenHand:
    """open hand で current_shanten が closed-hand 値にならない"""

    def test_open_tenpai(self):
        """1副露 tenpai: closed=2 → open=0"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (27, 1),
        ), dtype=np.float32)
        closed_sh = FlatFeatureEncoder._compute_current_shanten(counts, meld_count=0)
        open_sh = FlatFeatureEncoder._compute_current_shanten(counts, meld_count=1)
        assert closed_sh[0] > open_sh[0]
        assert open_sh[0] == pytest.approx(0.0 / 8.0)  # tenpai

    def test_open_iishanten(self):
        """1副露 iishanten"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1),
            (24, 1), (25, 1),
            (27, 2), (31, 1),
        ), dtype=np.float32)
        open_sh = FlatFeatureEncoder._compute_current_shanten(counts, meld_count=1)
        # 1 meld + 1 mentsu (123m) + 2 partial (45p, 78s) + jantai (東東) = 1-shanten
        assert open_sh[0] == pytest.approx(1.0 / 8.0)

    def test_closed_regression(self):
        """closed hand は meld_count=0 で変わらない"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (27, 3), (28, 1),
        ), dtype=np.float32)
        sh = FlatFeatureEncoder._compute_current_shanten(counts, meld_count=0)
        assert sh[0] == pytest.approx(0.0 / 8.0)  # tenpai

    def test_pon_tenpai(self):
        """ポン後 tenpai"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1),
            (27, 2),
        ), dtype=np.float32)
        sh = FlatFeatureEncoder._compute_current_shanten(counts, meld_count=1)
        assert sh[0] == pytest.approx(0.0 / 8.0)

    def test_2_meld_tenpai(self):
        """2副露 tenpai"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (27, 1),
        ), dtype=np.float32)
        sh = FlatFeatureEncoder._compute_current_shanten(counts, meld_count=2)
        assert sh[0] == pytest.approx(0.0 / 8.0)

    def test_3_meld_agari(self):
        """3副露 agari"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (27, 2),
        ), dtype=np.float32)
        sh = FlatFeatureEncoder._compute_current_shanten(counts, meld_count=3)
        # agari = -1, -1/8 is negative but clamp check
        assert sh[0] < 0  # -1/8


# ========== shanten_hint / ukeire ==========

class TestHintAndUkeireOpenHand:
    """open hand の shanten_hint / ukeire が baseline teacher と矛盾しない"""

    def test_open_hint_meld_count_1(self):
        """1副露 tenpai: 不要牌の shanten_sign=-1"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (28, 1),  # 南: 不要牌
        ), dtype=np.float32)
        mask = np.array([1.0 if c > 0 else 0.0 for c in counts], dtype=np.float32)
        hint, ukeire = FlatFeatureEncoder._compute_hint_and_ukeire(
            counts, mask, meld_count=1)
        # 南(28) を切ってもテンパイ維持 → sign=0
        # mentsu 牌を切ると悪化 → sign=-1
        assert hint[0] == -1.0  # 1m を切ると悪化
        assert hint[28] == 0.0  # 南を切っても維持

    def test_closed_hint_regression(self):
        """closed hand の hint が変わらない"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (27, 3), (28, 1),
        ), dtype=np.float32)
        mask = np.array([1.0 if c > 0 else 0.0 for c in counts], dtype=np.float32)
        hint0, ukeire0 = FlatFeatureEncoder._compute_hint_and_ukeire(
            counts.copy(), mask, meld_count=0)
        # 南(28) を切ると tenpai 維持
        assert hint0[28] == 0.0

    def test_hint_sign_matches_baseline_best(self):
        """open hand の shanten_sign=0 牌に baseline best_tile が含まれる"""
        from mahjong_rl.baseline.rule_based import RuleBasedBaseline

        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (28, 1),
        )
        counts_f = np.array(counts, dtype=np.float32)
        mask = np.array([1.0 if c > 0 else 0.0 for c in counts], dtype=np.float32)

        hint, _ = FlatFeatureEncoder._compute_hint_and_ukeire(
            counts_f.copy(), mask, meld_count=1)

        bl = RuleBasedBaseline()
        hand_ids = []
        for t in range(34):
            for i in range(counts[t]):
                hand_ids.append(t * 4 + i)
        bl_best = bl.select_discard(hand_ids, mask, meld_count=1)

        # baseline best_tile は shanten_sign >= 0 (維持以上)
        assert hint[bl_best] >= 0.0

    def test_chi_open_hint(self):
        """チー後の hint"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1),
            (27, 2),
        ), dtype=np.float32)
        mask = np.array([1.0 if c > 0 else 0.0 for c in counts], dtype=np.float32)
        hint, _ = FlatFeatureEncoder._compute_hint_and_ukeire(
            counts, mask, meld_count=1)
        # tenpai (waiting for 6s or 9s for 789s mentsu)
        # cutting 東 breaks jantai → sign should be -1
        assert hint[27] == -1.0

    def test_yakuhai_pon_hint(self):
        """役牌ポン後の hint"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (28, 1),
        ), dtype=np.float32)
        mask = np.array([1.0 if c > 0 else 0.0 for c in counts], dtype=np.float32)
        hint_open, _ = FlatFeatureEncoder._compute_hint_and_ukeire(
            counts, mask, meld_count=1)
        hint_closed, _ = FlatFeatureEncoder._compute_hint_and_ukeire(
            counts.copy(), mask, meld_count=0)
        # open: 南(28) を切ってもテンパイ維持
        # closed: 10 tiles as 4mentsu+jantai impossible → different
        assert hint_open[28] == 0.0  # open: 維持


# ========== opponent shanten ==========

class TestOpponentShantenOpenHand:
    """opponent shanten が open hand で正しい"""

    def test_opponent_shanten_uses_melds(self):
        """opponent の meld_count が反映される"""
        # 直接テスト: compute_shanten に meld_count を渡す
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (27, 1),
        )
        closed_sh = compute_shanten(counts, meld_count=0)
        open_sh = compute_shanten(counts, meld_count=1)
        assert open_sh < closed_sh  # open で tenpai


# ========== integration ==========

class TestEvalSmoke:
    """Stage2 eval smoke"""

    def test_eval_smoke_no_crash(self):
        """Stage2a eval が crash しない"""
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
        from mahjong_rl.models.stage2a_model import Stage2aModel
        import torch

        encoder = FlatFeatureEncoder(observation_mode="full")
        input_dim = encoder.metadata().output_shape[0]
        model = Stage2aModel(input_dim=input_dim,
                              discard_hidden_dims=[32],
                              optional_hidden_dims=[32])
        evaluator = Stage2aEvaluator(
            model=model, encoder=encoder,
            device=torch.device("cpu"))
        result = evaluator.evaluate(num_matches=2, seed_start=42)
        assert result["num_matches"] == 2
        assert result["total_rounds"] > 0

    def test_eval_with_hints_no_crash(self):
        """hint 有効 config で eval crash しない"""
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
        from mahjong_rl.models.stage2a_model import Stage2aModel
        import torch

        encoder = FlatFeatureEncoder(
            observation_mode="full",
            shanten_hint=True,
            discard_ukeire_hint=True,
            current_shanten_input=True,
            shape_hint=True,
        )
        input_dim = encoder.metadata().output_shape[0]
        model = Stage2aModel(input_dim=input_dim,
                              discard_hidden_dims=[32],
                              optional_hidden_dims=[32])
        evaluator = Stage2aEvaluator(
            model=model, encoder=encoder,
            device=torch.device("cpu"))
        result = evaluator.evaluate(num_matches=2, seed_start=42)
        assert result["num_matches"] == 2
        assert result["total_rounds"] > 0


# ========== CQ-0261: fallback + danger_mask ==========

class TestFallbackShantenHint:
    """Python fallback の _compute_shanten_hint に meld_count が効く"""

    def test_open_tenpai_hint_fallback(self):
        """1副露 tenpai: 不要牌は sign=0, mentsu 牌は sign=-1"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (28, 1),
        ), dtype=np.float32)
        hint = FlatFeatureEncoder._compute_shanten_hint(counts.copy(), meld_count=1)
        assert hint[28] == 0.0   # 南: 切ってもテンパイ維持
        assert hint[0] == -1.0   # 1m: mentsu を崩す → 悪化

    def test_closed_regression_fallback(self):
        """meld_count=0 で従来と同じ"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (27, 3), (28, 1),
        ), dtype=np.float32)
        hint = FlatFeatureEncoder._compute_shanten_hint(counts.copy(), meld_count=0)
        assert hint[28] == 0.0  # 南: テンパイ維持

    def test_open_vs_closed_current_shanten_different(self):
        """open と closed で current_shanten が異なる"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (28, 1),
        ), dtype=np.float32)
        sh_closed = FlatFeatureEncoder._compute_current_shanten(counts.copy(), meld_count=0)
        sh_open = FlatFeatureEncoder._compute_current_shanten(counts.copy(), meld_count=1)
        # closed: 10 tiles = 3 mentsu → shanten=2
        # open: 1 meld + 3 mentsu + isolated → tenpai=0
        assert sh_closed[0] > sh_open[0]


class TestFallbackDiscardUkeire:
    """Python fallback の _compute_discard_ukeire に meld_count が効く"""

    def test_open_tenpai_ukeire_fallback(self):
        """1副露 tenpai: ukeire が valid float"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (28, 1),
        ), dtype=np.float32)
        mask = np.array([1.0 if c > 0 else 0.0 for c in counts], dtype=np.float32)
        ukeire = FlatFeatureEncoder._compute_discard_ukeire(
            counts.copy(), legal_mask=mask, meld_count=1)
        assert np.all(ukeire >= 0.0)
        assert np.all(ukeire <= 1.0)

    def test_closed_regression_ukeire_fallback(self):
        """meld_count=0 で回帰しない"""
        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (27, 3), (28, 1),
        ), dtype=np.float32)
        mask = np.array([1.0 if c > 0 else 0.0 for c in counts], dtype=np.float32)
        ukeire = FlatFeatureEncoder._compute_discard_ukeire(
            counts.copy(), legal_mask=mask, meld_count=0)
        assert np.all(ukeire >= 0.0)
        assert np.all(ukeire <= 1.0)


class TestFallbackHintAndUkeire:
    """_compute_hint_and_ukeire が fallback でも meld_count を伝播する"""

    def test_fallback_with_monkeypatch(self, monkeypatch):
        """C++ analyze を無効化して fallback 経路をテスト"""
        import mahjong_rl.encoders.flat_encoder as fe_mod
        monkeypatch.setattr(fe_mod, "_HAS_CPP_ANALYZE", False)

        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (28, 1),
        ), dtype=np.float32)
        mask = np.array([1.0 if c > 0 else 0.0 for c in counts], dtype=np.float32)
        hint, ukeire = FlatFeatureEncoder._compute_hint_and_ukeire(
            counts.copy(), mask, meld_count=1)
        assert hint[28] == 0.0
        assert hint[0] == -1.0

    def test_fallback_closed_regression(self, monkeypatch):
        """fallback 経路で closed hand 回帰しない"""
        import mahjong_rl.encoders.flat_encoder as fe_mod
        monkeypatch.setattr(fe_mod, "_HAS_CPP_ANALYZE", False)

        counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (27, 3), (28, 1),
        ), dtype=np.float32)
        mask = np.array([1.0 if c > 0 else 0.0 for c in counts], dtype=np.float32)
        hint, ukeire = FlatFeatureEncoder._compute_hint_and_ukeire(
            counts.copy(), mask, meld_count=0)
        assert hint[28] == 0.0


class TestDangerMaskOpenHand:
    """opponent open-hand tenpai の danger_mask"""

    def test_open_tenpai_danger(self):
        """3副露 tenpai opponent: 東(27) が danger=1"""
        opp_counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (27, 1),
        ), dtype=np.float32)
        mask = FlatFeatureEncoder._compute_danger_mask(
            opp_counts.copy(), known_shanten=0, meld_count=3)
        assert mask[27] == 1.0
        assert mask[0] == 0.0

    def test_closed_tenpai_danger_regression(self):
        """closed hand tenpai の danger_mask は変わらない"""
        opp_counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (27, 3), (28, 1),
        ), dtype=np.float32)
        mask = FlatFeatureEncoder._compute_danger_mask(
            opp_counts.copy(), known_shanten=0, meld_count=0)
        assert mask[28] == 1.0

    def test_open_not_tenpai_returns_zeros(self):
        """not tenpai → all zeros"""
        opp_counts = np.array(_counts(
            (0, 1), (1, 1),
            (27, 1),
        ), dtype=np.float32)
        mask = FlatFeatureEncoder._compute_danger_mask(
            opp_counts.copy(), known_shanten=3, meld_count=1)
        assert mask.sum() == 0.0

    def test_open_vs_closed_danger_differs(self):
        """同じ counts でも meld_count で danger が変わる"""
        opp_counts = np.array(_counts(
            (0, 1), (1, 1), (2, 1),
            (27, 1),
        ), dtype=np.float32)
        open_sh = compute_shanten(_counts((0, 1), (1, 1), (2, 1), (27, 1)), meld_count=3)
        closed_sh = compute_shanten(_counts((0, 1), (1, 1), (2, 1), (27, 1)), meld_count=0)
        mask_open = FlatFeatureEncoder._compute_danger_mask(
            opp_counts.copy(), known_shanten=open_sh, meld_count=3)
        mask_closed = FlatFeatureEncoder._compute_danger_mask(
            opp_counts.copy(), known_shanten=closed_sh, meld_count=0)
        assert mask_open.sum() > 0   # open: tenpai → has danger
        assert mask_closed.sum() == 0  # closed: not tenpai
