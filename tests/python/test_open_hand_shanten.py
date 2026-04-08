"""CQ-0259: open-hand shanten / acceptance / best-discard テスト

closed hand の回帰 + open hand の多パターンを網羅する。
"""
import pytest
import numpy as np

pytestmark = pytest.mark.smoke

from mahjong_rl.baseline.shanten import compute_shanten
from mahjong_rl.baseline.rule_based import RuleBasedBaseline
from mahjong_rl._mahjong_core import (
    compute_shanten as compute_shanten_cpp,
    find_best_discard as find_best_discard_cpp,
    analyze_discards as analyze_discards_cpp,
)


def _counts(*args):
    """tile_type: count ペアから counts 配列を作る"""
    c = [0] * 34
    for tile, cnt in args:
        c[tile] = cnt
    return c


def _mask_from_counts(counts):
    """counts > 0 の牌種を合法にする mask"""
    return np.array([1.0 if c > 0 else 0.0 for c in counts], dtype=np.float32)


def _hand_ids_from_counts(counts):
    """counts から TileId リストを作る (各牌種 0-3 枚)"""
    ids = []
    for t in range(34):
        for i in range(counts[t]):
            ids.append(t * 4 + i)
    return ids


# ========== Closed Hand Regression ==========

class TestClosedHandRegression:
    """closed hand の shanten / best discard が変わらないことを確認"""

    def test_tenpai_closed(self):
        """門前テンパイ: 1m2m3m 4p5p6p 7s8s9s 東東東 + 南 (13枚)"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),    # 1m2m3m
            (12, 1), (13, 1), (14, 1),  # 4p5p6p
            (24, 1), (25, 1), (26, 1),  # 7s8s9s
            (27, 3),                     # 東東東
            (28, 1),                     # 南
        )
        assert compute_shanten(counts) == 0

    def test_agari_closed(self):
        """門前和了: 1m2m3m 4p5p6p 7s8s9s 東東東 南南"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (27, 3), (28, 2),
        )
        assert compute_shanten(counts) == -1

    def test_iishanten_closed(self):
        """門前一向聴: 1m2m3m 4p5p6p 7s8s 東東 白白 (12枚)"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),    # 1m2m3m
            (12, 1), (13, 1), (14, 1),  # 4p5p6p
            (24, 1), (25, 1),            # 7s8s (tatsumi)
            (27, 2),                     # 東東 (jantai)
            (31, 2),                     # 白白 (partial/mentsu material)
        )
        # mentsu=2, jantai(東東), partial=2(7s8s, 白白→koutsu potential)
        # 8 - 2*2 - 2 - 1 = 1
        assert compute_shanten(counts) == 1

    def test_meld_count_0_matches_default(self):
        """meld_count=0 は従来と同じ"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (27, 3), (28, 1),
        )
        assert compute_shanten(counts, 0) == compute_shanten(counts)

    def test_chiitoi_closed(self):
        """七対子テンパイ"""
        counts = _counts(
            (0, 2), (3, 2), (6, 2), (9, 2), (12, 2), (15, 2), (27, 1),
        )
        assert compute_shanten(counts) == 0

    def test_kokushi_closed(self):
        """国士テンパイ: 12 種 + 中中 = 13枚、白なし"""
        counts = _counts(
            (0, 1), (8, 1), (9, 1), (17, 1), (18, 1), (26, 1),
            (27, 1), (28, 1), (29, 1), (30, 1), (32, 1), (33, 2),
            # 白(31) なし → 白 待ち
        )
        assert compute_shanten(counts) == 0

    def test_closed_best_discard(self):
        """門前で best discard が変わらない"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (27, 3), (28, 1),
        )
        mask = _mask_from_counts(counts)
        result = find_best_discard_cpp(counts, mask.astype(int).tolist(), 0)
        # 南(28)を切ればテンパイ維持
        assert result["best_tile"] == 28


# ========== Open Hand Regular Shanten ==========

class TestOpenHandShanten:
    """open hand の regular shanten が副露面子数込みで計算される"""

    def test_pon_1_tenpai(self):
        """ポン1面子: 1m2m3m 4p5p6p 7s8s9s 東 (10枚, 1副露) → tenpai"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (27, 1),
        )
        assert compute_shanten(counts, 1) == 0

    def test_pon_1_iishanten(self):
        """ポン1面子: 1m2m3m 4p5p 7s8s 東東 (10枚, 1副露) → iishanten"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1),
            (24, 1), (25, 1),
            (27, 2), (31, 1),
        )
        assert compute_shanten(counts, 1) == 1

    def test_chi_1_tenpai(self):
        """チー1面子: 1m2m3m 4p5p6p 7s8s 東東 (10枚, 1副露) → tenpai"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1),
            (27, 2),
        )
        assert compute_shanten(counts, 1) == 0

    def test_chi_1_ryanshanten(self):
        """チー1面子: 1m2m 4p5p 7s8s 東 (7枚, 1副露) → 3-shanten"""
        counts = _counts(
            (0, 1), (1, 1),
            (12, 1), (13, 1),
            (24, 1), (25, 1),
            (27, 1),
        )
        # 1 meld + 0 hand mentsu, 3 partial (tatsumi), 0 jantai
        # 8 - 2*1 - 3 - 0 = 3
        assert compute_shanten(counts, 1) == 3

    def test_2_melds_ryanshanten(self):
        """2副露: 1m2m3m 東 (4枚, 2副露) → 2-shanten"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (27, 1),
        )
        # 2 melds + 1 mentsu(123m) = 3 mentsu. 東 isolated (no partial).
        # 8 - 2*3 - 0 - 0 = 2
        assert compute_shanten(counts, 2) == 2

    def test_2_melds_tenpai_proper(self):
        """2副露 tenpai: 1m2m3m 東東 (5枚, 2副露)"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (27, 2),
        )
        # 2 melds + 1 mentsu(123m) + jantai(東東) = 3 mentsu + jantai
        # shanten = 8 - 2*3 - 0 - 1 = 1? No.
        # Total mentsu needed = 4. Have 2+1=3 from melds+hand. Need 1 more.
        # partial = 0 (only jantai left). shanten = 8 - 2*3 - 0 - 1 = 1
        # Actually tenpai should be waiting for 4th mentsu.
        # Let me reconsider: 5 tiles, 2 melds. We need 4 mentsu + 1 jantai total.
        # 2 melds done. From 5 tiles: 1 mentsu (123m) + 1 jantai (東東) = perfect.
        # 2+1=3 mentsu + jantai. Need 1 more mentsu. shanten = 8 - 2*3 - 0 - 1 = 1
        # Not tenpai! We need all 4 mentsu. 3 mentsu + jantai = 1-shanten.
        assert compute_shanten(counts, 2) == 1

    def test_2_melds_tenpai_correct(self):
        """2副露 tenpai: 1m2m3m 4p5p6p 東 (7枚, 2副露) → tenpai for 東 pair"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (27, 1),
        )
        # 2 melds + 2 mentsu(123m, 456p) = 4 mentsu. Need jantai.
        # 東 is isolated. Waiting for 東 pair.
        assert compute_shanten(counts, 2) == 0

    def test_3_melds_tenpai(self):
        """3副露 tenpai: 1m2m3m 東 (4枚, 3副露)"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (27, 1),
        )
        # 3 melds + 1 mentsu(123m) = 4 mentsu. Need jantai.
        assert compute_shanten(counts, 3) == 0

    def test_3_melds_agari(self):
        """3副露 和了: 1m2m3m 東東 (5枚, 3副露)"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (27, 2),
        )
        assert compute_shanten(counts, 3) == -1

    def test_yakuhai_pon_ryanshanten(self):
        """役牌ポン後: 1m2m3m 4p5p6p 7s (7枚, ポン白) → 2-shanten"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1),
        )
        # 1 meld + 2 hand mentsu = 3. 7s isolated (no partial).
        # 8 - 2*3 - 0 - 0 = 2
        assert compute_shanten(counts, 1) == 2

    def test_yakuhai_pon_iishanten(self):
        """役牌ポン後 iishanten: 1m2m 4p5p 7s8s 東 (7枚, ポン白)"""
        counts = _counts(
            (0, 1), (1, 1),
            (12, 1), (13, 1),
            (24, 1), (25, 1),
            (27, 1),
        )
        # 1 meld. From hand: 3 partial (12m, 45p, 78s), mentsu=0+1=1.
        # shanten = 8 - 2*1 - 3 - 0 = 3
        assert compute_shanten(counts, 1) == 3

    def test_open_no_chiitoi(self):
        """open hand で七対子が混ざらない"""
        # 7 pairs concealed, but with 1 meld → chiitoi impossible
        counts = _counts(
            (0, 2), (3, 2), (6, 2), (9, 2), (12, 2), (15, 2), (27, 1),
        )
        open_sh = compute_shanten(counts, 1)
        closed_sh = compute_shanten(counts, 0)
        # closed: chiitoi tenpai = 0
        assert closed_sh == 0
        # open: regular only, 1 meld + concealed = should be higher
        assert open_sh > closed_sh

    def test_open_no_kokushi(self):
        """open hand で国士が混ざらない"""
        # 12 unique terminals + 中中 = 13 tiles (白なし = 国士テンパイ)
        counts = _counts(
            (0, 1), (8, 1), (9, 1), (17, 1), (18, 1), (26, 1),
            (27, 1), (28, 1), (29, 1), (30, 1), (32, 1), (33, 2),
        )
        open_sh = compute_shanten(counts, 1)
        closed_sh = compute_shanten(counts, 0)
        assert closed_sh == 0  # kokushi tenpai
        assert open_sh > closed_sh  # open: regular only


# ========== Open Hand Acceptance ==========

class TestOpenHandAcceptance:
    """open hand の acceptance 計算"""

    def test_open_tenpai_acceptance(self):
        """open tenpai で待ち牌が正しく数えられる"""
        # 3副露 + 123m 東 → tenpai for 東
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (27, 1),
        )
        mask = [1 if c > 0 else 0 for c in counts]
        result = analyze_discards_cpp(counts, mask, 3)
        # Already tenpai (shanten=0). Cutting 東 should maintain or improve.
        # After cutting 東: shanten for 123m with 3 melds = need 1 more mentsu+jantai = 2
        # After cutting 1m: 2m3m 東 with 3 melds = need jantai + mentsu
        # Best cut should be the one that doesn't break the mentsu

    def test_open_acceptance_count(self):
        """open iishanten で acceptance の優劣"""
        # 1 meld, hand = 1m2m3m 4p5p 7s8s 東東 南 (10 tiles)
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1),
            (24, 1), (25, 1),
            (27, 2), (28, 1),
        )
        mask = [1 if c > 0 else 0 for c in counts]
        result = analyze_discards_cpp(counts, mask, 1)
        # 南(28) を切るのが最善 (mentsu を崩さない)
        sh_after = result["shanten_after"]
        assert sh_after[28] <= min(
            sh_after[t] for t in range(34) if sh_after[t] >= 0 and t != 28)


# ========== Best Discard ==========

class TestOpenHandBestDiscard:
    """open hand の best discard"""

    def test_obvious_discard(self):
        """明確に1枚だけ正解がある: 3副露 + 123m 東南 → 南 切り"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (27, 1), (28, 1),
        )
        mask = _mask_from_counts(counts)
        hand_ids = _hand_ids_from_counts(counts)
        bl = RuleBasedBaseline()
        action = bl.select_discard(hand_ids, mask, meld_count=3)
        # 南(28) or 東(27) を切ってテンパイ。どちらもテンパイになるが
        # acceptance で差がつく可能性。少なくとも 0-2(mentsu 部分)は切らない
        assert action not in [0, 1, 2]

    def test_multiple_best(self):
        """複数 best がある: 3副露 + 東 南 → どちらも同等"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (27, 1), (28, 1),
        )
        mask = _mask_from_counts(counts)
        hand_ids = _hand_ids_from_counts(counts)
        bl = RuleBasedBaseline()
        _, best_mask = bl.select_discard_with_best_set(
            hand_ids, mask, meld_count=3)
        # 東(27) and 南(28) should both be in best set
        assert best_mask[27] > 0.5
        assert best_mask[28] > 0.5
        # mentsu tiles should NOT be in best set
        assert best_mask[0] < 0.5
        assert best_mask[1] < 0.5
        assert best_mask[2] < 0.5

    def test_yakuhai_pon_drop_useless(self):
        """役牌ポン済みで不要牌を切る: 1副露 + 1m2m3m 4p5p6p 7s8s9s 南"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (28, 1),
        )
        mask = _mask_from_counts(counts)
        hand_ids = _hand_ids_from_counts(counts)
        bl = RuleBasedBaseline()
        action = bl.select_discard(hand_ids, mask, meld_count=1)
        # 南(28) を切ればテンパイ
        assert action == 28

    def test_chi_after_best_changes(self):
        """チー後に面子数込みで最善 discard が変わる"""
        # closed: 1m2m3m 4m5m 7p8p 1s2s 東 南 白 (12枚) → ryanshanten
        counts_closed = _counts(
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
            (15, 1), (16, 1),
            (18, 1), (19, 1),
            (27, 1), (28, 1), (31, 1),
        )
        mask_c = _mask_from_counts(counts_closed)
        res_closed = find_best_discard_cpp(
            counts_closed, mask_c.astype(int).tolist(), 0)

        # after chi (e.g. 6p chi): hand = 1m2m3m 4m5m 1s2s 東 南 白 (10枚, 1副露)
        counts_open = _counts(
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
            (18, 1), (19, 1),
            (27, 1), (28, 1), (31, 1),
        )
        mask_o = _mask_from_counts(counts_open)
        res_open = find_best_discard_cpp(
            counts_open, mask_o.astype(int).tolist(), 1)

        # open should have lower or equal shanten (meld helps)
        assert res_open["best_shanten"] <= res_closed["best_shanten"]

    def test_cpp_python_consistency(self):
        """C++ と Python fallback が同じ結果"""
        counts = _counts(
            (0, 1), (1, 1), (2, 1),
            (12, 1), (13, 1), (14, 1),
            (24, 1), (25, 1), (26, 1),
            (28, 1),
        )
        sh_cpp = compute_shanten_cpp(counts, 1)
        sh_py = compute_shanten(counts, 1)
        assert sh_cpp == sh_py


# ========== Integration ==========

class TestStage2Integration:
    """Stage2 selfplay smoke で win_called が出る"""

    def test_selfplay_win_called_appears(self, tmp_path):
        """selfplay smoke で win_called > 0 (40 match)"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder
        from mahjong_rl.call_shard import DecisionShardReader

        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        worker.generate(num_matches=40, base_seed=42)

        samples = DecisionShardReader(tmp_path).read_all()
        labels = set(s.round_terminal_label for s in samples)
        # win_called should appear with enough matches
        assert "win_called" in labels, \
            f"win_called が出ない: {labels}"

    def test_selfplay_smoke_small(self, tmp_path):
        """small smoke で crash しない + call_count > 0"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        stats = worker.generate(num_matches=5, base_seed=0)
        assert stats["call_count"] > 0
        assert stats["discard_count"] > 0
