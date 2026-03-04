"""CQ-0029: Stage 1 環境ラッパーテスト"""
import pytest
import numpy as np

pytestmark = pytest.mark.smoke
from mahjong_rl.env import Stage1Env
from mahjong_rl import (
    ActionType, Phase, RunMode, EventType, MeldType,
    PartialObservation, FullObservation, NUM_TILE_TYPES,
)


@pytest.fixture
def env_full():
    return Stage1Env(observation_mode="full")


@pytest.fixture
def env_partial():
    return Stage1Env(observation_mode="partial")


class TestReset:
    """reset テスト"""

    def test_reset_returns_observation(self, env_full):
        obs, info = env_full.reset(seed=42)
        assert isinstance(obs, FullObservation)
        assert "current_player" in info
        assert "scores" in info

    def test_reset_partial_mode(self, env_partial):
        obs, info = env_partial.reset(seed=42)
        assert isinstance(obs, PartialObservation)

    def test_reset_at_discard_point(self, env_full):
        """reset 後は打牌決定点にいる"""
        env_full.reset(seed=42)
        phase = env_full.env_state.round_state.phase
        assert phase == Phase.SelfActionPhase


class TestStep:
    """step テスト"""

    def test_step_with_legal_action(self, env_full):
        env_full.reset(seed=42)
        mask = env_full.get_legal_mask()
        legal_types = np.where(mask > 0.5)[0]
        assert len(legal_types) > 0

        obs, rewards, terminated, truncated, info = env_full.step(int(legal_types[0]))
        assert isinstance(obs, FullObservation)
        assert rewards.shape == (4,)
        assert isinstance(terminated, bool)
        assert truncated is False

    def test_step_illegal_action_raises(self, env_full):
        env_full.reset(seed=42)
        mask = env_full.get_legal_mask()
        illegal_types = np.where(mask < 0.5)[0]
        if len(illegal_types) > 0:
            with pytest.raises(ValueError):
                env_full.step(int(illegal_types[0]))


class TestLegalMask:
    """legal mask テスト"""

    def test_mask_shape(self, env_full):
        env_full.reset(seed=42)
        mask = env_full.get_legal_mask()
        assert mask.shape == (NUM_TILE_TYPES,)
        assert mask.dtype == np.float32
        assert mask.sum() > 0

    def test_mask_matches_hand(self, env_full):
        """legal mask は手牌に含まれる牌種のサブセット"""
        env_full.reset(seed=42)
        mask = env_full.get_legal_mask()
        hand = env_full.env_state.round_state.players[
            env_full.current_player
        ].hand
        hand_types = set()
        for tid in hand:
            hand_types.add(tid // 4)
        for t in range(NUM_TILE_TYPES):
            if mask[t] > 0.5:
                assert t in hand_types, f"マスクに含まれる牌種 {t} が手牌にない"


class TestStage1Rules:
    """Stage 1 固定ルール確認"""

    def test_no_calls_in_full_match(self, env_full):
        """半荘完走中に副露（チー/ポン/大明槓）が発生しない"""
        env_full.reset(seed=42)
        steps = 0
        while steps < 10000:
            mask = env_full.get_legal_mask()
            legal_types = np.where(mask > 0.5)[0]
            if len(legal_types) == 0:
                break
            obs, rewards, terminated, truncated, info = env_full.step(int(legal_types[0]))
            steps += 1
            if terminated:
                break

        # 全プレイヤーの副露を確認
        rs = env_full.env_state.round_state
        for p_idx in range(4):
            for meld in rs.players[p_idx].melds:
                assert meld.type not in (MeldType.Chi, MeldType.Pon, MeldType.Daiminkan), \
                    f"Player {p_idx} に副露 {meld.type} がある"

    def test_full_match_completes(self, env_full):
        """半荘が正常に完走する"""
        env_full.reset(seed=42)
        steps = 0
        while steps < 10000:
            mask = env_full.get_legal_mask()
            legal_types = np.where(mask > 0.5)[0]
            if len(legal_types) == 0:
                break
            _, _, terminated, _, info = env_full.step(int(legal_types[0]))
            steps += 1
            if terminated:
                break
        assert info["is_match_over"], f"半荘が完走していない (steps={steps})"

    def test_multiple_seeds_complete(self, env_full):
        """複数 seed で半荘完走"""
        for seed in [0, 7, 42, 256]:
            env_full.reset(seed=seed)
            steps = 0
            while steps < 10000:
                mask = env_full.get_legal_mask()
                legal_types = np.where(mask > 0.5)[0]
                if len(legal_types) == 0:
                    break
                _, _, terminated, _, info = env_full.step(int(legal_types[0]))
                steps += 1
                if terminated:
                    break
            assert info["is_match_over"], f"seed={seed} で半荘未完走"


class TestObservationMode:
    """観測モード切替テスト"""

    def test_full_mode(self, env_full):
        obs, _ = env_full.reset(seed=42)
        assert isinstance(obs, FullObservation)
        assert env_full.observation_mode == "full"

    def test_partial_mode(self, env_partial):
        obs, _ = env_partial.reset(seed=42)
        assert isinstance(obs, PartialObservation)
        assert env_partial.observation_mode == "partial"

    def test_step_preserves_mode(self, env_partial):
        env_partial.reset(seed=42)
        mask = env_partial.get_legal_mask()
        legal_types = np.where(mask > 0.5)[0]
        obs, _, _, _, _ = env_partial.step(int(legal_types[0]))
        assert isinstance(obs, PartialObservation)


class TestDualObservation:
    """make_dual_observation テスト (CQ-0040)"""

    def test_dual_observation_returns_both_types(self, env_full):
        """同一局面から Full と Partial の両方を取得できる"""
        env_full.reset(seed=42)
        full_obs, partial_obs = env_full.make_dual_observation()
        assert isinstance(full_obs, FullObservation)
        assert isinstance(partial_obs, PartialObservation)

    def test_dual_observation_same_game_state(self, env_full):
        """Full と Partial は同一局面の観測である"""
        env_full.reset(seed=42)
        full_obs, partial_obs = env_full.make_dual_observation()
        # 同一局の round_number であること
        assert full_obs.round_number == partial_obs.round_number

    def test_dual_observation_after_step(self, env_full):
        """step 後も make_dual_observation が動作する"""
        env_full.reset(seed=42)
        mask = env_full.get_legal_mask()
        legal_types = np.where(mask > 0.5)[0]
        env_full.step(int(legal_types[0]))

        full_obs, partial_obs = env_full.make_dual_observation()
        assert isinstance(full_obs, FullObservation)
        assert isinstance(partial_obs, PartialObservation)
