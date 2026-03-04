"""CQ-0030: FeatureEncoder テスト"""
import pytest
import numpy as np

pytestmark = pytest.mark.smoke
from mahjong_rl import (
    GameEngine, EnvironmentState, RunMode,
    PartialObservation, FullObservation,
    NUM_TILE_TYPES,
)
from mahjong_rl._mahjong_core import make_partial_observation, make_full_observation
from mahjong_rl.encoders import (
    FeatureEncoder, EncoderMetadata,
    FlatFeatureEncoder, ChannelTensorEncoder,
)


@pytest.fixture
def full_obs():
    """FullObservation を生成するフィクスチャ"""
    engine = GameEngine()
    env = EnvironmentState()
    env.run_mode = RunMode.Fast
    engine.reset_match(env, 42)
    return make_full_observation(env)


@pytest.fixture
def partial_obs():
    """PartialObservation を生成するフィクスチャ"""
    engine = GameEngine()
    env = EnvironmentState()
    env.run_mode = RunMode.Fast
    engine.reset_match(env, 42)
    player = env.round_state.current_player
    return make_partial_observation(env, player)


class TestFlatEncoderPartial:
    """FlatFeatureEncoder Partial モードテスト"""

    def test_output_shape(self, partial_obs):
        enc = FlatFeatureEncoder(observation_mode="partial")
        result = enc.encode(partial_obs)
        assert result.shape == enc.metadata().output_shape

    def test_dtype(self, partial_obs):
        enc = FlatFeatureEncoder(observation_mode="partial")
        result = enc.encode(partial_obs)
        assert result.dtype == np.float32

    def test_partial_dim(self, partial_obs):
        enc = FlatFeatureEncoder(observation_mode="partial")
        result = enc.encode(partial_obs)
        assert result.shape == (353,)

    def test_hand_counts_nonnegative(self, partial_obs):
        enc = FlatFeatureEncoder(observation_mode="partial")
        result = enc.encode(partial_obs)
        # 最初の34次元は手牌カウント → 非負
        hand_part = result[:NUM_TILE_TYPES]
        assert np.all(hand_part >= 0)
        # 手牌は13枚(他家)または14枚(ツモ直後の自家)
        assert hand_part.sum() in (13.0, 14.0)


class TestFlatEncoderFull:
    """FlatFeatureEncoder Full モードテスト"""

    def test_output_shape(self, full_obs):
        enc = FlatFeatureEncoder(observation_mode="full")
        result = enc.encode(full_obs)
        assert result.shape == enc.metadata().output_shape

    def test_dtype(self, full_obs):
        enc = FlatFeatureEncoder(observation_mode="full")
        result = enc.encode(full_obs)
        assert result.dtype == np.float32

    def test_full_dim(self, full_obs):
        enc = FlatFeatureEncoder(observation_mode="full")
        result = enc.encode(full_obs)
        assert result.shape == (455,)

    def test_all_hands_present(self, full_obs):
        """Full モードでは全4家の手牌が含まれる"""
        enc = FlatFeatureEncoder(observation_mode="full")
        result = enc.encode(full_obs)
        # Full: 4家手牌(136) + 4家河(136) + 4家副露(136) + ドラ(34) + スカラー(5) + スコア(4) + 立直(4)
        # 最初の136次元 = 4家手牌
        hands_part = result[:4 * NUM_TILE_TYPES]
        for p in range(4):
            hand_counts = hands_part[p * NUM_TILE_TYPES:(p + 1) * NUM_TILE_TYPES]
            assert np.all(hand_counts >= 0)
            # 各プレイヤーは13枚(他家)または14枚(ツモ後の親)
            assert hand_counts.sum() in (13.0, 14.0)


class TestChannelEncoderPartial:
    """ChannelTensorEncoder Partial モードテスト"""

    def test_output_shape(self, partial_obs):
        enc = ChannelTensorEncoder(observation_mode="partial")
        result = enc.encode(partial_obs)
        assert result.shape == enc.metadata().output_shape

    def test_dtype(self, partial_obs):
        enc = ChannelTensorEncoder(observation_mode="partial")
        result = enc.encode(partial_obs)
        assert result.dtype == np.float32

    def test_partial_channels(self, partial_obs):
        enc = ChannelTensorEncoder(observation_mode="partial")
        result = enc.encode(partial_obs)
        assert result.shape == (16, 4, 9)

    def test_hand_binary_planes(self, partial_obs):
        """手牌の binary plane が正しい"""
        enc = ChannelTensorEncoder(observation_mode="partial")
        result = enc.encode(partial_obs)
        # ch 0-3 は手牌 binary planes (0 or 1)
        hand_planes = result[:4]
        assert np.all((hand_planes == 0.0) | (hand_planes == 1.0))
        # ch0 >= ch1 >= ch2 >= ch3 (1枚以上なら ch0=1, 2枚以上なら ch1=1, ...)
        for r in range(4):
            for c in range(9):
                for k in range(3):
                    if hand_planes[k + 1, r, c] == 1.0:
                        assert hand_planes[k, r, c] == 1.0


class TestChannelEncoderFull:
    """ChannelTensorEncoder Full モードテスト"""

    def test_output_shape(self, full_obs):
        enc = ChannelTensorEncoder(observation_mode="full")
        result = enc.encode(full_obs)
        assert result.shape == enc.metadata().output_shape

    def test_dtype(self, full_obs):
        enc = ChannelTensorEncoder(observation_mode="full")
        result = enc.encode(full_obs)
        assert result.dtype == np.float32

    def test_full_channels(self, full_obs):
        enc = ChannelTensorEncoder(observation_mode="full")
        result = enc.encode(full_obs)
        assert result.shape == (32, 4, 9)

    def test_all_player_hand_planes(self, full_obs):
        """Full の ch 16-31 に全4家手牌が含まれる"""
        enc = ChannelTensorEncoder(observation_mode="full")
        result = enc.encode(full_obs)
        # ch 16-31: 4家 × 4 binary planes
        all_hand_planes = result[16:32]
        assert np.all((all_hand_planes == 0.0) | (all_hand_planes == 1.0))
        # 各プレイヤーの手牌は少なくとも1枚ある
        for p in range(4):
            player_planes = all_hand_planes[p * 4:(p + 1) * 4]
            assert player_planes[0].sum() > 0  # 少なくとも ch0 に1枚以上


class TestMetadata:
    """EncoderMetadata テスト"""

    def test_flat_partial_metadata(self):
        enc = FlatFeatureEncoder(observation_mode="partial")
        meta = enc.metadata()
        assert meta.output_shape == (353,)
        assert meta.dtype == np.float32
        assert meta.observation_mode == "partial"
        assert meta.name == "FlatFeatureEncoder"

    def test_flat_full_metadata(self):
        enc = FlatFeatureEncoder(observation_mode="full")
        meta = enc.metadata()
        assert meta.output_shape == (455,)

    def test_channel_partial_metadata(self):
        enc = ChannelTensorEncoder(observation_mode="partial")
        meta = enc.metadata()
        assert meta.output_shape == (16, 4, 9)
        assert meta.dtype == np.float32
        assert meta.observation_mode == "partial"
        assert meta.name == "ChannelTensorEncoder"

    def test_channel_full_metadata(self):
        enc = ChannelTensorEncoder(observation_mode="full")
        meta = enc.metadata()
        assert meta.output_shape == (32, 4, 9)

    def test_output_dim_flat(self):
        enc = FlatFeatureEncoder(observation_mode="partial")
        assert enc.output_dim == 353

    def test_output_dim_channel(self):
        enc = ChannelTensorEncoder(observation_mode="partial")
        assert enc.output_dim == 16 * 4 * 9


class TestEncoderInterchangeability:
    """エンコーダ差し替え可能性テスト"""

    def test_both_are_feature_encoder(self):
        flat = FlatFeatureEncoder()
        channel = ChannelTensorEncoder()
        assert isinstance(flat, FeatureEncoder)
        assert isinstance(channel, FeatureEncoder)

    def test_both_encode_partial(self, partial_obs):
        flat = FlatFeatureEncoder(observation_mode="partial")
        channel = ChannelTensorEncoder(observation_mode="partial")
        flat_result = flat.encode(partial_obs)
        channel_result = channel.encode(partial_obs)
        assert flat_result.dtype == channel_result.dtype == np.float32

    def test_both_encode_full(self, full_obs):
        flat = FlatFeatureEncoder(observation_mode="full")
        channel = ChannelTensorEncoder(observation_mode="full")
        flat_result = flat.encode(full_obs)
        channel_result = channel.encode(full_obs)
        assert flat_result.dtype == channel_result.dtype == np.float32


class TestBothMode:
    """observation_mode='both' テスト"""

    def test_flat_both_accepts_partial(self, partial_obs):
        enc = FlatFeatureEncoder(observation_mode="both")
        result = enc.encode(partial_obs)
        assert result.shape == (353,)

    def test_flat_both_accepts_full(self, full_obs):
        enc = FlatFeatureEncoder(observation_mode="both")
        result = enc.encode(full_obs)
        assert result.shape == (455,)

    def test_channel_both_accepts_partial(self, partial_obs):
        enc = ChannelTensorEncoder(observation_mode="both")
        result = enc.encode(partial_obs)
        assert result.shape == (16, 4, 9)

    def test_channel_both_accepts_full(self, full_obs):
        enc = ChannelTensorEncoder(observation_mode="both")
        result = enc.encode(full_obs)
        assert result.shape == (32, 4, 9)
