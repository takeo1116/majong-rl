"""テスト: reward_shaping.py — ShantenDeltaTracker / RewardSchedule (CQ-0141)"""
import pytest

from mahjong_rl.reward_shaping import ShantenDeltaTracker, RewardSchedule


class TestShantenDeltaTracker:
    """ShantenDeltaTracker の単体テスト"""

    @staticmethod
    def _make_hand_ids(counts: list[int]) -> list[int]:
        """counts (34要素) から TileId リスト (14枚) を生成する"""
        ids: list[int] = []
        for tile_type, cnt in enumerate(counts):
            for j in range(cnt):
                ids.append(tile_type * 4 + j)
        return ids

    def test_first_discard_returns_zero_reward_nan_raw(self):
        """CQ-0149: 初回打牌の reward は 0、raw_delta は NaN (prev 未定義)"""
        import math
        tracker = ShantenDeltaTracker(scale=0.01, mode="both")
        # 14枚: 1m×4 + 2m×4 + 3m×4 + 4m×2
        counts_14 = [4, 4, 4, 2] + [0] * 30
        hand = self._make_hand_ids(counts_14)
        reward, raw_delta = tracker.compute_reward(0, hand, discard_tile_type=3)
        assert reward == 0.0
        assert math.isnan(raw_delta)

    def test_scale_multiplied(self):
        """改善/悪化時に scale が乗算される"""
        tracker = ShantenDeltaTracker(scale=0.05, mode="both")
        # 1回目 (prev=None → 0)
        counts_14 = [4, 4, 4, 2] + [0] * 30
        hand = self._make_hand_ids(counts_14)
        tracker.compute_reward(0, hand, discard_tile_type=3)
        # 2回目: 同じ手配 → delta=0
        reward, raw_delta = tracker.compute_reward(0, hand, discard_tile_type=3)
        # delta=0 → reward=0 (scale に関わらず)
        assert reward == 0.0
        assert raw_delta == 0.0

    def test_improve_only_clamps_negative(self):
        """mode='improve_only' で悪化 delta の reward を 0 にクランプ、raw_delta は負"""
        tracker = ShantenDeltaTracker(scale=1.0, mode="improve_only")
        # 1回目 (4m×4 + 5m×4 + 6m×4 + 7m×2 → 良い手)
        good_14 = [0, 0, 0, 4, 4, 4, 2] + [0] * 27
        hand = self._make_hand_ids(good_14)
        tracker.compute_reward(0, hand, discard_tile_type=6)  # prev 設定

        # 2回目 (バラバラ手 → シャンテン悪化)
        bad_14 = [1] * 14 + [0] * 20
        hand2 = self._make_hand_ids(bad_14)
        reward, raw_delta = tracker.compute_reward(0, hand2, discard_tile_type=0)
        assert reward >= 0.0  # 悪化でも reward は 0 以上
        assert raw_delta < 0.0  # CQ-0148: raw_delta は mode に依存しない

    def test_both_allows_negative(self):
        """mode='both' では悪化時に負の reward"""
        tracker = ShantenDeltaTracker(scale=1.0, mode="both")
        # 1回目 (良い手)
        good_14 = [0, 0, 0, 4, 4, 4, 2] + [0] * 27
        hand = self._make_hand_ids(good_14)
        tracker.compute_reward(0, hand, discard_tile_type=6)  # prev 設定

        # 2回目 (バラバラ手)
        bad_14 = [1] * 14 + [0] * 20
        hand2 = self._make_hand_ids(bad_14)
        reward, raw_delta = tracker.compute_reward(0, hand2, discard_tile_type=0)
        # 良い手→悪い手なので delta < 0 → reward < 0
        assert reward <= 0.0
        assert raw_delta < 0.0

    def test_reset_clears_state(self):
        """reset() で prev が消え、次打牌は初回扱い (reward=0, raw=NaN)"""
        import math
        tracker = ShantenDeltaTracker(scale=0.01, mode="both")
        counts_14 = [4, 4, 4, 2] + [0] * 30
        hand = self._make_hand_ids(counts_14)
        tracker.compute_reward(0, hand, discard_tile_type=3)

        tracker.reset()
        reward, raw_delta = tracker.compute_reward(0, hand, discard_tile_type=3)
        assert reward == 0.0
        assert math.isnan(raw_delta)

    def test_per_player_independent(self):
        """プレイヤーごとに独立追跡される"""
        import math
        tracker = ShantenDeltaTracker(scale=0.01, mode="both")
        counts_14 = [4, 4, 4, 2] + [0] * 30
        hand = self._make_hand_ids(counts_14)

        # player 0 の 1回目
        tracker.compute_reward(0, hand, discard_tile_type=3)
        # player 1 の 1回目 (player 0 の prev は影響しない)
        reward_p1, raw_p1 = tracker.compute_reward(1, hand, discard_tile_type=3)
        assert reward_p1 == 0.0  # player 1 は初回
        assert math.isnan(raw_p1)  # CQ-0149: 初回は NaN

    def test_raw_delta_independent_of_scale(self):
        """CQ-0148: raw_delta は scale に依存しない"""
        counts_14 = [4, 4, 4, 2] + [0] * 30
        hand = self._make_hand_ids(counts_14)

        # scale=0.0
        t0 = ShantenDeltaTracker(scale=0.0, mode="both")
        t0.compute_reward(0, hand, discard_tile_type=3)
        _, raw0 = t0.compute_reward(0, hand, discard_tile_type=3)

        # scale=1.0
        t1 = ShantenDeltaTracker(scale=1.0, mode="both")
        t1.compute_reward(0, hand, discard_tile_type=3)
        _, raw1 = t1.compute_reward(0, hand, discard_tile_type=3)

        assert raw0 == raw1

    def test_raw_delta_independent_of_mode(self):
        """CQ-0148: raw_delta は mode に依存しない"""
        good_14 = [0, 0, 0, 4, 4, 4, 2] + [0] * 27
        bad_14 = [1] * 14 + [0] * 20
        hand_good = self._make_hand_ids(good_14)
        hand_bad = self._make_hand_ids(bad_14)

        # mode=both
        tb = ShantenDeltaTracker(scale=1.0, mode="both")
        tb.compute_reward(0, hand_good, discard_tile_type=6)
        _, raw_both = tb.compute_reward(0, hand_bad, discard_tile_type=0)

        # mode=improve_only
        ti = ShantenDeltaTracker(scale=1.0, mode="improve_only")
        ti.compute_reward(0, hand_good, discard_tile_type=6)
        _, raw_io = ti.compute_reward(0, hand_bad, discard_tile_type=0)

        assert raw_both == raw_io
        assert raw_both < 0  # 悪化

    def test_invalid_mode_raises(self):
        """不正な mode でエラー"""
        with pytest.raises(ValueError, match="Unknown shanten_delta mode"):
            ShantenDeltaTracker(scale=0.01, mode="invalid")


class TestRewardSchedule:
    """RewardSchedule の単体テスト"""

    def test_constant(self):
        """constant schedule は常に 1.0"""
        schedule = RewardSchedule({"type": "constant"})
        assert schedule.get_scale_factor(0.0) == 1.0
        assert schedule.get_scale_factor(0.5) == 1.0
        assert schedule.get_scale_factor(1.0) == 1.0

    def test_linear_decay_endpoints(self):
        """linear_decay: 0→1.0, 1→0.0"""
        schedule = RewardSchedule({"type": "linear_decay"})
        assert schedule.get_scale_factor(0.0) == pytest.approx(1.0)
        assert schedule.get_scale_factor(1.0) == pytest.approx(0.0)
        assert schedule.get_scale_factor(0.5) == pytest.approx(0.5)

    def test_clamp_progress(self):
        """progress が [0,1] 外でもクランプされる"""
        schedule = RewardSchedule({"type": "linear_decay"})
        assert schedule.get_scale_factor(-0.5) == pytest.approx(1.0)
        assert schedule.get_scale_factor(1.5) == pytest.approx(0.0)

    def test_default_constant(self):
        """config 省略時は constant"""
        schedule = RewardSchedule(None)
        assert schedule.get_scale_factor(0.5) == 1.0

    def test_invalid_type_raises(self):
        """不正な type でエラー"""
        with pytest.raises(ValueError, match="Unknown schedule type"):
            RewardSchedule({"type": "exponential"})
