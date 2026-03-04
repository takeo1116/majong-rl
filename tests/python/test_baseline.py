"""CQ-0033: Rule-based Baseline テスト"""
import pytest
import numpy as np

from mahjong_rl import NUM_TILE_TYPES
from mahjong_rl.baseline import compute_shanten, RuleBasedBaseline
from mahjong_rl.env import Stage1Env


# --- シャンテン数テスト ---

@pytest.mark.smoke
class TestShantenAgari:
    """和了形 (シャンテン数 = -1)"""

    def test_regular_agari(self):
        """1m2m3m 4m5m6m 7m8m9m 1p2p3p 東東 → 和了"""
        counts = [0] * 34
        for t in range(9):  # 萬子 1-9 各1枚
            counts[t] = 1
        counts[9] = 1   # 1p
        counts[10] = 1  # 2p
        counts[11] = 1  # 3p
        counts[27] = 2  # 東×2
        assert compute_shanten(counts) == -1

    def test_chiitoitsu_agari(self):
        """7対子の和了形"""
        counts = [0] * 34
        for t in [0, 1, 2, 3, 4, 5, 6]:
            counts[t] = 2
        assert compute_shanten(counts) == -1

    def test_kokushi_agari(self):
        """国士無双の和了形"""
        counts = [0] * 34
        terminals = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        for t in terminals:
            counts[t] = 1
        counts[0] = 2  # 1mが対子
        assert compute_shanten(counts) == -1


@pytest.mark.smoke
class TestShantenTenpai:
    """テンパイ (シャンテン数 = 0)"""

    def test_regular_tenpai(self):
        """1m2m3m 4m5m6m 7m8m9m 1p2p 東東 → 1p3pか東待ち"""
        counts = [0] * 34
        for t in range(9):
            counts[t] = 1
        counts[9] = 1   # 1p
        counts[10] = 1  # 2p
        counts[27] = 2  # 東×2
        # 13枚でテンパイ
        assert compute_shanten(counts) == 0

    def test_chiitoitsu_tenpai(self):
        """七対子テンパイ: 6対子 + 1枚"""
        counts = [0] * 34
        for t in [0, 1, 2, 3, 4, 5]:
            counts[t] = 2
        counts[6] = 1
        assert compute_shanten(counts) == 0

    def test_kokushi_tenpai(self):
        """国士テンパイ: 13種のうち12種 + 1対子"""
        counts = [0] * 34
        terminals = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        for t in terminals:
            counts[t] = 1
        counts[33] = 0  # 中を抜く
        counts[0] = 2    # 1mを対子
        # 12種 + 対子 = テンパイ
        assert compute_shanten(counts) == 0


@pytest.mark.smoke
class TestShantenIshanten:
    """一向聴 (シャンテン数 = 1)"""

    def test_regular_iishanten(self):
        """1m2m3m 4m5m6m 7m8m 1p3p 東東東 → 一向聴
        (1p3p は嵌張ターツ → 9m or 2p 待ちにはならず、2つの不完全面子)
        """
        counts = [0] * 34
        for t in range(6):  # 1m-6m 各1枚
            counts[t] = 1
        counts[6] = 1   # 7m
        counts[7] = 1   # 8m
        counts[9] = 1   # 1p
        counts[11] = 1  # 3p (嵌張)
        counts[27] = 3  # 東×3
        # 3面子 (1m2m3m, 4m5m6m, 東東東) + 2ターツ (7m8m, 1p3p) = 一向聴
        assert compute_shanten(counts) == 1


@pytest.mark.smoke
class TestShantenEdgeCases:
    """シャンテン数エッジケース"""

    def test_empty_hand(self):
        """空の手牌"""
        counts = [0] * 34
        sh = compute_shanten(counts)
        assert sh == 6  # 七対子が最善 (6-0=6)

    def test_accepts_numpy(self):
        """numpy 配列を受け付ける"""
        counts = np.zeros(34, dtype=int)
        counts[0] = 2
        counts[1] = 2
        counts[2] = 2
        counts[3] = 2
        counts[4] = 2
        counts[5] = 2
        counts[6] = 2
        sh = compute_shanten(counts)
        assert sh == -1  # 七対子和了


# --- RuleBasedBaseline テスト ---

@pytest.mark.smoke
class TestRuleBasedBaseline:
    """RuleBasedBaseline テスト"""

    def test_select_returns_legal(self):
        """合法手を返す"""
        baseline = RuleBasedBaseline()
        # 1m×4 2m×4 3m×4 4m×2 = 14枚
        hand = list(range(4)) + list(range(4, 8)) + list(range(8, 12)) + [12, 13]
        mask = np.zeros(34, dtype=np.float32)
        mask[0] = 1.0  # 1m
        mask[1] = 1.0  # 2m
        mask[2] = 1.0  # 3m
        mask[3] = 1.0  # 4m
        result = baseline.select_discard(hand, mask)
        assert mask[result] > 0.5

    def test_select_minimizes_shanten(self):
        """シャンテン数を最小化する打牌を選ぶ"""
        baseline = RuleBasedBaseline()
        # テンパイ手牌に孤立牌が1枚混ざっている
        # 1m2m3m 4m5m6m 7m8m9m 東東 + 白 + 中 = 14枚
        counts_expected = [0] * 34
        for t in range(9):
            counts_expected[t] = 1
        counts_expected[27] = 2
        counts_expected[31] = 1  # 白
        counts_expected[33] = 1  # 中

        hand = []
        for t in range(9):
            hand.append(t * 4)  # 各種1枚目
        hand.append(27 * 4)      # 東
        hand.append(27 * 4 + 1)  # 東
        hand.append(31 * 4)      # 白
        hand.append(33 * 4)      # 中

        mask = np.zeros(34, dtype=np.float32)
        for tid in hand:
            mask[tid // 4] = 1.0

        result = baseline.select_discard(hand, mask)
        # 白か中を切ればテンパイ → それ以外を切ると一向聴以上
        assert result in (31, 33)


@pytest.mark.slow
class TestBaselineOnStage1:
    """Stage1Env 上でのベースライン動作テスト"""

    def test_baseline_completes_match(self):
        """ベースラインが半荘を完走できる"""
        env = Stage1Env(observation_mode="full")
        baseline = RuleBasedBaseline()

        env.reset(seed=42)
        steps = 0
        while steps < 10000:
            mask = env.get_legal_mask()
            legal_types = np.where(mask > 0.5)[0]
            if len(legal_types) == 0:
                break

            hand = env.env_state.round_state.players[
                env.current_player
            ].hand
            tile_type = baseline.select_discard(list(hand), mask)

            obs, rewards, terminated, truncated, info = env.step(tile_type)
            steps += 1
            if terminated:
                break

        assert info["is_match_over"], f"ベースラインが半荘を完走していない (steps={steps})"

    def test_baseline_multiple_seeds(self):
        """複数 seed でベースラインが完走"""
        env = Stage1Env(observation_mode="full")
        baseline = RuleBasedBaseline()

        for seed in [0, 7, 42]:
            env.reset(seed=seed)
            steps = 0
            while steps < 10000:
                mask = env.get_legal_mask()
                legal_types = np.where(mask > 0.5)[0]
                if len(legal_types) == 0:
                    break

                hand = env.env_state.round_state.players[
                    env.current_player
                ].hand
                tile_type = baseline.select_discard(list(hand), mask)

                _, _, terminated, _, info = env.step(tile_type)
                steps += 1
                if terminated:
                    break

            assert info["is_match_over"], f"seed={seed} でベースライン半荘未完走"
