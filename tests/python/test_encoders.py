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

    def test_legal_mask_ignored(self, partial_obs):
        """legal_mask を渡しても出力が変わらない (CQ-0173)"""
        enc = ChannelTensorEncoder(observation_mode="partial")
        result_no_mask = enc.encode(partial_obs)
        mask = np.ones(NUM_TILE_TYPES, dtype=np.float32)
        mask[0] = 0.0
        result_with_mask = enc.encode(partial_obs, legal_mask=mask)
        np.testing.assert_array_equal(result_no_mask, result_with_mask)

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


class TestShantenHint:
    """シャンテン補助特徴 on/off テスト (CQ-0120)"""

    def test_off_preserves_partial_dim(self):
        """off 時 partial=353 を維持"""
        enc = FlatFeatureEncoder(observation_mode="partial", shanten_hint=False)
        assert enc.metadata().output_shape == (353,)

    def test_off_preserves_full_dim(self):
        """off 時 full=455 を維持"""
        enc = FlatFeatureEncoder(observation_mode="full", shanten_hint=False)
        assert enc.metadata().output_shape == (455,)

    def test_on_adds_34_partial(self):
        """on 時 partial=387"""
        enc = FlatFeatureEncoder(observation_mode="partial", shanten_hint=True)
        assert enc.metadata().output_shape == (387,)

    def test_on_adds_34_full(self):
        """on 時 full=489"""
        enc = FlatFeatureEncoder(observation_mode="full", shanten_hint=True)
        assert enc.metadata().output_shape == (489,)

    def test_metadata_matches_output_partial(self, partial_obs):
        """on 時 metadata.output_shape と encode() 結果 shape が一致 (partial)"""
        enc = FlatFeatureEncoder(observation_mode="partial", shanten_hint=True)
        result = enc.encode(partial_obs)
        assert result.shape == enc.metadata().output_shape
        assert result.dtype == np.float32

    def test_metadata_matches_output_full(self, full_obs):
        """on 時 metadata.output_shape と encode() 結果 shape が一致 (full)"""
        enc = FlatFeatureEncoder(observation_mode="full", shanten_hint=True)
        result = enc.encode(full_obs)
        assert result.shape == enc.metadata().output_shape
        assert result.dtype == np.float32

    def test_shanten_hint_values(self, partial_obs):
        """shanten_hint の値が {-1, 0} の範囲にある (CQ-0123: +1 は discard 評価で不発生)"""
        enc_on = FlatFeatureEncoder(observation_mode="partial", shanten_hint=True)
        enc_off = FlatFeatureEncoder(observation_mode="partial", shanten_hint=False)
        result_on = enc_on.encode(partial_obs)
        result_off = enc_off.encode(partial_obs)

        # off 部分は一致する
        assert np.array_equal(result_on[:353], result_off)

        # 末尾34次元が shanten hint
        hint = result_on[353:]
        assert hint.shape == (34,)
        # 値は -1, 0 のいずれか（+1 は shanten 単調性により不発生）
        for v in hint:
            assert v in (-1.0, 0.0), f"想定外の hint 値: {v}"
        # 手牌に含まれる牌種のいずれかは非ゼロ（13-14枚あるので）
        assert np.any(hint != 0.0), "hint が全て0: 手牌があるはず"


class TestShantenHintSemantics:
    """シャンテン補助特徴の意味検証テスト (CQ-0122, CQ-0124)

    delta_shanten_sign の定義:
      base = shanten(手牌), after = shanten(手牌から t を除去)
      delta = base - after

    運用値域 (discard 評価):
      0.0 = 維持（最適打牌候補）または手牌に存在しない牌種
     -1.0 = 悪化（シャンテン数が増加する打牌）

    +1 (改善) は shanten(n枚) <= shanten(n-1枚) の単調性により、
    現行の discard 評価では数学的に発生しない。
    実装上 delta > 0 分岐は将来拡張互換のガード節として残っている。
    """

    def test_tenpai_hand_worsening(self):
        """テンパイ手: 面子構成牌を切ると悪化(-1)"""
        from mahjong_rl.baseline.shanten import compute_shanten

        # 1m2m3m 4p5p6p 7s8s9s 東東南北 (13枚, shanten=1)
        # 面子牌(1m等)を切ると shanten=2 → 悪化(-1)
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[0] = 1; counts[1] = 1; counts[2] = 1  # 1m2m3m
        counts[12] = 1; counts[13] = 1; counts[14] = 1  # 4p5p6p
        counts[24] = 1; counts[25] = 1; counts[26] = 1  # 7s8s9s
        counts[27] = 2  # 東東 (雀頭)
        counts[28] = 1  # 南 (浮き)
        counts[30] = 1  # 北 (浮き)
        assert compute_shanten(counts) == 1

        hint = FlatFeatureEncoder._compute_shanten_hint(counts.copy())

        # 面子構成牌 (1m=0) を切ると悪化
        assert hint[0] == -1.0, "1m を切ると悪化するはず"
        # 雀頭 (東=27) を切っても悪化
        assert hint[27] == -1.0, "東を切ると悪化するはず"

    def test_tenpai_hand_maintenance(self):
        """テンパイ手: 浮き牌を切ると維持(0) = 最適打牌"""
        from mahjong_rl.baseline.shanten import compute_shanten

        # 同上の手牌
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[0] = 1; counts[1] = 1; counts[2] = 1
        counts[12] = 1; counts[13] = 1; counts[14] = 1
        counts[24] = 1; counts[25] = 1; counts[26] = 1
        counts[27] = 2; counts[28] = 1; counts[30] = 1
        assert compute_shanten(counts) == 1

        hint = FlatFeatureEncoder._compute_shanten_hint(counts.copy())

        # 浮き牌 (南=28, 北=30) を切っても shanten 維持
        assert hint[28] == 0.0, "南を切ってもシャンテン維持のはず"
        assert hint[30] == 0.0, "北を切ってもシャンテン維持のはず"

    def test_absent_tile_is_zero(self):
        """手牌にない牌種は 0.0"""
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[0] = 1; counts[1] = 1; counts[2] = 1
        counts[12] = 1; counts[13] = 1; counts[14] = 1
        counts[24] = 1; counts[25] = 1; counts[26] = 1
        counts[27] = 2; counts[28] = 1; counts[30] = 1

        hint = FlatFeatureEncoder._compute_shanten_hint(counts.copy())

        # 手牌にない牌種 (4m=3, 白=31 等) は 0.0
        assert hint[3] == 0.0, "4m は手牌にないので 0.0"
        assert hint[31] == 0.0, "白は手牌にないので 0.0"
        assert hint[33] == 0.0, "中は手牌にないので 0.0"

    def test_improvement_never_occurs(self):
        """仕様検証: discard 評価で +1 は非発生（shanten 単調性による不在保証）"""
        from mahjong_rl.baseline.shanten import compute_shanten

        # 複数の手牌パターンで +1 が出ないことを確認
        rng = np.random.RandomState(42)
        for _ in range(20):
            counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
            for _ in range(13):
                t = rng.randint(0, NUM_TILE_TYPES)
                while counts[t] >= 4:
                    t = rng.randint(0, NUM_TILE_TYPES)
                counts[t] += 1
            hint = FlatFeatureEncoder._compute_shanten_hint(counts.copy())
            assert np.all(hint <= 0.0), \
                f"+1 (改善) が発生: hint={hint[hint > 0]}"

    def test_mixed_hand_has_both_zero_and_minus(self):
        """典型的な手牌で 0(維持) と -1(悪化) が混在する"""
        # 1m2m3m 4p5p6p 7s8s9s 東東南北 (shanten=1)
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[0] = 1; counts[1] = 1; counts[2] = 1
        counts[12] = 1; counts[13] = 1; counts[14] = 1
        counts[24] = 1; counts[25] = 1; counts[26] = 1
        counts[27] = 2; counts[28] = 1; counts[30] = 1

        hint = FlatFeatureEncoder._compute_shanten_hint(counts.copy())

        has_zero = np.any(hint == 0.0)
        has_minus = np.any(hint == -1.0)
        assert has_zero, "0.0 (維持/候補外) が存在するはず"
        assert has_minus, "-1.0 (悪化) が存在するはず"


class TestDiscardUkeireHint:
    """打牌候補受け入れ枚数テスト (CQ-0168)"""

    def test_off_preserves_partial_dim(self):
        """off 時 partial=353 を維持"""
        enc = FlatFeatureEncoder(observation_mode="partial", discard_ukeire_hint=False)
        assert enc.metadata().output_shape == (353,)

    def test_off_preserves_full_dim(self):
        """off 時 full=455 を維持"""
        enc = FlatFeatureEncoder(observation_mode="full", discard_ukeire_hint=False)
        assert enc.metadata().output_shape == (455,)

    def test_on_adds_34_partial(self):
        """on 時 partial=353+34=387"""
        enc = FlatFeatureEncoder(observation_mode="partial", discard_ukeire_hint=True)
        assert enc.metadata().output_shape == (387,)

    def test_on_adds_34_full(self):
        """on 時 full=455+34=489"""
        enc = FlatFeatureEncoder(observation_mode="full", discard_ukeire_hint=True)
        assert enc.metadata().output_shape == (489,)

    def test_metadata_matches_output_partial(self, partial_obs):
        """metadata と encode 結果の shape が一致"""
        enc = FlatFeatureEncoder(observation_mode="partial", discard_ukeire_hint=True)
        result = enc.encode(partial_obs)
        assert result.shape == enc.metadata().output_shape
        assert result.dtype == np.float32

    def test_metadata_matches_output_full(self, full_obs):
        """metadata と encode 結果の shape が一致"""
        enc = FlatFeatureEncoder(observation_mode="full", discard_ukeire_hint=True)
        result = enc.encode(full_obs)
        assert result.shape == enc.metadata().output_shape
        assert result.dtype == np.float32

    def test_values_in_range(self, partial_obs):
        """値域が [0, 1]"""
        enc = FlatFeatureEncoder(observation_mode="partial", discard_ukeire_hint=True)
        result = enc.encode(partial_obs)
        ukeire = result[353:]
        assert ukeire.shape == (34,)
        assert np.all(ukeire >= 0.0)
        assert np.all(ukeire <= 1.0)

    def test_absent_tile_is_zero(self):
        """手牌にない牌種は 0.0"""
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[0] = 1; counts[1] = 1; counts[2] = 1  # 1m2m3m
        counts[12] = 1; counts[13] = 1; counts[14] = 1  # 4p5p6p
        counts[24] = 1; counts[25] = 1; counts[26] = 1  # 7s8s9s
        counts[27] = 2; counts[28] = 1; counts[30] = 1
        ukeire = FlatFeatureEncoder._compute_discard_ukeire(counts.copy())
        # 手牌にない牌種は 0.0
        assert ukeire[3] == 0.0, "4m は手牌にないので 0.0"
        assert ukeire[31] == 0.0, "白は手牌にないので 0.0"

    def test_max_is_one(self):
        """最大値が 1.0 (正規化の確認)"""
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[0] = 1; counts[1] = 1; counts[2] = 1
        counts[12] = 1; counts[13] = 1; counts[14] = 1
        counts[24] = 1; counts[25] = 1; counts[26] = 1
        counts[27] = 2; counts[28] = 1; counts[30] = 1
        ukeire = FlatFeatureEncoder._compute_discard_ukeire(counts.copy())
        # 手牌に含まれる牌種のうち少なくとも1つは受け入れが最大 → 1.0
        assert ukeire.max() == pytest.approx(1.0), "正規化後の最大値は 1.0"

    def test_combined_with_shanten_hint(self):
        """shanten_hint と併用可"""
        enc = FlatFeatureEncoder(
            observation_mode="partial",
            shanten_hint=True,
            discard_ukeire_hint=True,
        )
        # 353 + 34 (shanten_hint) + 34 (discard_ukeire) = 421
        assert enc.metadata().output_shape == (421,)


class TestCurrentShantenInput:
    """policy 用 current_shanten テスト (CQ-0169)"""

    def test_off_preserves_partial_dim(self):
        """off 時 partial=353 を維持"""
        enc = FlatFeatureEncoder(observation_mode="partial", current_shanten_input=False)
        assert enc.metadata().output_shape == (353,)

    def test_on_adds_1_partial(self):
        """on 時 partial=354"""
        enc = FlatFeatureEncoder(observation_mode="partial", current_shanten_input=True)
        assert enc.metadata().output_shape == (354,)

    def test_on_adds_1_full(self):
        """on 時 full=456"""
        enc = FlatFeatureEncoder(observation_mode="full", current_shanten_input=True)
        assert enc.metadata().output_shape == (456,)

    def test_metadata_matches_output_partial(self, partial_obs):
        """metadata と encode 結果の shape が一致"""
        enc = FlatFeatureEncoder(observation_mode="partial", current_shanten_input=True)
        result = enc.encode(partial_obs)
        assert result.shape == enc.metadata().output_shape
        assert result.dtype == np.float32

    def test_metadata_matches_output_full(self, full_obs):
        """metadata と encode 結果の shape が一致"""
        enc = FlatFeatureEncoder(observation_mode="full", current_shanten_input=True)
        result = enc.encode(full_obs)
        assert result.shape == enc.metadata().output_shape
        assert result.dtype == np.float32

    def test_value_in_range(self, partial_obs):
        """値域 [0, 1]"""
        enc = FlatFeatureEncoder(observation_mode="partial", current_shanten_input=True)
        result = enc.encode(partial_obs)
        shanten_val = result[353]
        assert 0.0 <= shanten_val <= 1.0

    def test_known_value(self):
        """既知シャンテン数の検証"""
        from mahjong_rl.baseline.shanten import compute_shanten

        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[0] = 1; counts[1] = 1; counts[2] = 1
        counts[12] = 1; counts[13] = 1; counts[14] = 1
        counts[24] = 1; counts[25] = 1; counts[26] = 1
        counts[27] = 2; counts[28] = 1; counts[30] = 1
        sh = compute_shanten(counts)
        result = FlatFeatureEncoder._compute_current_shanten(counts)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(sh / 8.0)


class TestShapeHint:
    """手牌形状ヒントテスト (CQ-0170)"""

    def test_off_preserves_partial_dim(self):
        """off 時 partial=353 を維持"""
        enc = FlatFeatureEncoder(observation_mode="partial", shape_hint=False)
        assert enc.metadata().output_shape == (353,)

    def test_on_adds_66_partial(self):
        """on 時 partial=353+66=419"""
        enc = FlatFeatureEncoder(observation_mode="partial", shape_hint=True)
        assert enc.metadata().output_shape == (419,)

    def test_on_adds_66_full(self):
        """on 時 full=455+66=521"""
        enc = FlatFeatureEncoder(observation_mode="full", shape_hint=True)
        assert enc.metadata().output_shape == (521,)

    def test_metadata_matches_output_partial(self, partial_obs):
        """metadata と encode 結果の shape が一致"""
        enc = FlatFeatureEncoder(observation_mode="partial", shape_hint=True)
        result = enc.encode(partial_obs)
        assert result.shape == enc.metadata().output_shape
        assert result.dtype == np.float32

    def test_metadata_matches_output_full(self, full_obs):
        """metadata と encode 結果の shape が一致"""
        enc = FlatFeatureEncoder(observation_mode="full", shape_hint=True)
        result = enc.encode(full_obs)
        assert result.shape == enc.metadata().output_shape
        assert result.dtype == np.float32

    def test_values_binary(self, partial_obs):
        """値が {0, 1} のみ"""
        enc = FlatFeatureEncoder(observation_mode="partial", shape_hint=True)
        result = enc.encode(partial_obs)
        shape_part = result[353:]
        assert shape_part.shape == (66,)
        for v in shape_part:
            assert v in (0.0, 1.0), f"想定外の値: {v}"

    def test_chi_detection(self):
        """順子検出: 1m2m3m → chi[0]=1"""
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[0] = 1; counts[1] = 1; counts[2] = 1  # 1m2m3m
        counts[27] = 2  # 東東
        # 追加でパディング
        for _ in range(8):
            pass  # 13枚にする
        counts[28] = 1; counts[29] = 1; counts[30] = 1; counts[31] = 1
        counts[32] = 1; counts[33] = 1; counts[12] = 1; counts[13] = 1
        result = FlatFeatureEncoder._compute_shape_hint(counts)
        chi_part = result[:21]
        # 1m2m3m の中心=2m(index 1), suit=0 → chi[0*7 + (1-1)] = chi[0]
        assert chi_part[0] == 1.0, "1m2m3m の順子が検出されるべき"

    def test_inside_wait_detection(self):
        """嵌張検出: 1m(ない)3m → inside_wait"""
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[0] = 1  # 1m
        counts[2] = 1  # 3m (2m=index 1 なし)
        counts[27] = 2; counts[28] = 1; counts[29] = 1; counts[30] = 1
        counts[31] = 1; counts[32] = 1; counts[33] = 1
        counts[12] = 1; counts[13] = 1; counts[14] = 1
        result = FlatFeatureEncoder._compute_shape_hint(counts)
        inside_part = result[21 + 24:]  # chi(21) + outside_wait(24) の後
        # 1m_3m 嵌張: 中心=2m(index 1), suit=0 → inside_wait[0*7 + (1-1)] = inside_wait[0]
        assert inside_part[0] == 1.0, "1m-3m の嵌張が検出されるべき"

    def test_outside_wait_detection(self):
        """塔子検出: 1m2m → outside_wait"""
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[0] = 1; counts[1] = 1  # 1m2m
        counts[27] = 2; counts[28] = 1; counts[29] = 1; counts[30] = 1
        counts[31] = 1; counts[32] = 1; counts[33] = 1
        counts[12] = 1; counts[13] = 1; counts[14] = 1
        result = FlatFeatureEncoder._compute_shape_hint(counts)
        outside_part = result[21:21 + 24]
        # 1m2m: pair_start=0, suit=0 → outside_wait[0*8 + 0] = outside_wait[0]
        assert outside_part[0] == 1.0, "1m2m の塔子が検出されるべき"

    def test_no_zihai_detection(self):
        """字牌は形状検出対象外"""
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[27] = 1; counts[28] = 1; counts[29] = 1  # 東南西 (連番ではない)
        counts[0] = 2; counts[1] = 1; counts[2] = 1
        counts[12] = 1; counts[13] = 1; counts[14] = 1
        counts[24] = 1; counts[25] = 1; counts[26] = 1
        result = FlatFeatureEncoder._compute_shape_hint(counts)
        # 字牌のスートインデックスは 27-33 → 3スート(0-26)外なので影響なし
        # 全体は数牌の形状のみ反映
        assert result.shape == (66,)

    def test_combined_all_features(self):
        """全オプション併用時の次元"""
        enc = FlatFeatureEncoder(
            observation_mode="partial",
            shanten_hint=True,
            discard_ukeire_hint=True,
            current_shanten_input=True,
            shape_hint=True,
        )
        # 353 + 34 + 34 + 1 + 66 = 488
        assert enc.metadata().output_shape == (488,)

    def test_combined_all_features_full(self):
        """全オプション併用時の次元 (full)"""
        enc = FlatFeatureEncoder(
            observation_mode="full",
            shanten_hint=True,
            discard_ukeire_hint=True,
            current_shanten_input=True,
            shape_hint=True,
        )
        # 455 + 34 + 34 + 1 + 66 = 590
        assert enc.metadata().output_shape == (590,)


class TestDiscardUkeireLegalMask:
    """discard_ukeire_norm の legal mask 整合テスト (CQ-0172)"""

    def test_legal_mask_zeros_illegal_tile(self):
        """legal_mask で非合法の牌種が 0.0 になる"""
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[0] = 1; counts[1] = 1; counts[2] = 1  # 1m2m3m
        counts[12] = 1; counts[13] = 1; counts[14] = 1  # 4p5p6p
        counts[24] = 1; counts[25] = 1; counts[26] = 1  # 7s8s9s
        counts[27] = 2; counts[28] = 1; counts[30] = 1

        # 1m (index 0) を非合法にする
        mask = np.ones(NUM_TILE_TYPES, dtype=np.float32)
        mask[0] = 0.0

        ukeire = FlatFeatureEncoder._compute_discard_ukeire(counts.copy(), legal_mask=mask)
        assert ukeire[0] == 0.0, "非合法牌種は 0.0"
        # 合法牌種はまだ値を持つ
        assert ukeire[28] >= 0.0  # 南は合法

    def test_no_mask_backward_compat(self):
        """legal_mask=None で従来と同じ挙動"""
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[0] = 1; counts[1] = 1; counts[2] = 1
        counts[12] = 1; counts[13] = 1; counts[14] = 1
        counts[24] = 1; counts[25] = 1; counts[26] = 1
        counts[27] = 2; counts[28] = 1; counts[30] = 1

        with_none = FlatFeatureEncoder._compute_discard_ukeire(counts.copy(), legal_mask=None)
        without = FlatFeatureEncoder._compute_discard_ukeire(counts.copy())
        np.testing.assert_array_equal(with_none, without)

    def test_riichi_like_scenario(self):
        """立直後相当: 手牌にあるが1種のみ合法"""
        counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        counts[0] = 1; counts[1] = 1; counts[2] = 1
        counts[12] = 1; counts[13] = 1; counts[14] = 1
        counts[24] = 1; counts[25] = 1; counts[26] = 1
        counts[27] = 2; counts[28] = 1; counts[30] = 1

        # 立直後: ツモ切りのみ合法 → 1種のみ mask=1
        mask = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        mask[30] = 1.0  # 北のみ合法

        ukeire = FlatFeatureEncoder._compute_discard_ukeire(counts.copy(), legal_mask=mask)
        # 北以外は全て 0.0
        for t in range(NUM_TILE_TYPES):
            if t != 30:
                assert ukeire[t] == 0.0, f"牌種 {t} は非合法なので 0.0"
        # 北の受け入れがあれば 1.0 (唯一の合法手なので max=自身)
        # 値は 0.0 か 1.0 のいずれか
        assert ukeire[30] in (0.0, 1.0)

    def test_encode_with_legal_mask(self, partial_obs):
        """encode() に legal_mask を渡して shape が正しい"""
        enc = FlatFeatureEncoder(observation_mode="partial", discard_ukeire_hint=True)
        mask = np.ones(NUM_TILE_TYPES, dtype=np.float32)
        result = enc.encode(partial_obs, legal_mask=mask)
        assert result.shape == enc.metadata().output_shape

    def test_encode_without_mask_still_works(self, partial_obs):
        """legal_mask なしでも encode() が動作する"""
        enc = FlatFeatureEncoder(observation_mode="partial", discard_ukeire_hint=True)
        result = enc.encode(partial_obs)
        assert result.shape == enc.metadata().output_shape


class TestCppBindingKeyContract:
    """C++ pybind11 戻り値の dict キー契約テスト"""

    def test_analyze_discards_keys(self):
        """analyze_discards が期待するキーを全て返す"""
        from mahjong_rl._mahjong_core import analyze_discards
        counts = [0] * 34
        counts[0] = 1; counts[1] = 1; counts[2] = 1
        counts[27] = 2; counts[28] = 1
        mask = [1] * 34
        result = analyze_discards(counts, mask)
        expected_keys = {"shanten_after", "acceptance", "ukeire_norm", "shanten_sign"}
        assert set(result.keys()) == expected_keys
        for key in expected_keys:
            assert len(result[key]) == 34, f"{key} の長さが 34 でない"

    def test_find_best_discard_keys(self):
        """find_best_discard が期待するキーを全て返す"""
        from mahjong_rl._mahjong_core import find_best_discard
        counts = [0] * 34
        counts[0] = 1; counts[1] = 1; counts[2] = 1
        counts[27] = 2; counts[28] = 1
        mask = [1] * 34
        result = find_best_discard(counts, mask)
        expected_keys = {"best_shanten", "best_acceptance", "best_tile", "best_mask"}
        assert set(result.keys()) == expected_keys
        assert isinstance(result["best_shanten"], int)
        assert isinstance(result["best_acceptance"], int)
        assert isinstance(result["best_tile"], int)
        assert len(result["best_mask"]) == 34


class TestTurnContext:
    """turn/time 文脈特徴テスト (CQ-0175)"""

    def test_off_preserves_partial_dim(self):
        enc = FlatFeatureEncoder(observation_mode="partial", turn_context=False)
        assert enc.metadata().output_shape == (353,)

    def test_on_adds_4_partial(self):
        enc = FlatFeatureEncoder(observation_mode="partial", turn_context=True)
        assert enc.metadata().output_shape == (357,)

    def test_on_adds_4_full(self):
        enc = FlatFeatureEncoder(observation_mode="full", turn_context=True)
        assert enc.metadata().output_shape == (459,)

    def test_metadata_matches_output_partial(self, partial_obs):
        enc = FlatFeatureEncoder(observation_mode="partial", turn_context=True)
        result = enc.encode(partial_obs)
        assert result.shape == enc.metadata().output_shape

    def test_metadata_matches_output_full(self, full_obs):
        enc = FlatFeatureEncoder(observation_mode="full", turn_context=True)
        result = enc.encode(full_obs)
        assert result.shape == enc.metadata().output_shape

    def test_turn_progress_range(self, partial_obs):
        enc = FlatFeatureEncoder(observation_mode="partial", turn_context=True)
        result = enc.encode(partial_obs)
        tc = result[353:]
        assert tc.shape == (4,)
        progress = tc[0]
        assert 0.0 <= progress <= 1.0

    def test_bucket_one_hot(self):
        """各巡目で exactly 1 つの bucket が立つ"""
        for turn in range(18):
            ctx = FlatFeatureEncoder._compute_turn_context(turn)
            assert ctx.shape == (4,)
            bucket = ctx[1:]
            assert bucket.sum() == 1.0, f"turn={turn}: bucket={bucket}"

    def test_early_mid_late_boundaries(self):
        """early(0-5), mid(6-11), late(12-17) の境界"""
        # early
        for turn in [0, 3, 5]:
            ctx = FlatFeatureEncoder._compute_turn_context(turn)
            assert ctx[1] == 1.0, f"turn={turn} should be early"
        # mid
        for turn in [6, 9, 11]:
            ctx = FlatFeatureEncoder._compute_turn_context(turn)
            assert ctx[2] == 1.0, f"turn={turn} should be mid"
        # late
        for turn in [12, 15, 17]:
            ctx = FlatFeatureEncoder._compute_turn_context(turn)
            assert ctx[3] == 1.0, f"turn={turn} should be late"

    def test_combined_all_features(self):
        enc = FlatFeatureEncoder(
            observation_mode="partial",
            shanten_hint=True,
            discard_ukeire_hint=True,
            current_shanten_input=True,
            shape_hint=True,
            turn_context=True,
        )
        # 353 + 34 + 34 + 1 + 66 + 4 = 492
        assert enc.metadata().output_shape == (492,)
