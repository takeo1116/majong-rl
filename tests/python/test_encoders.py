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
        assert result.shape == (361,)

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
        assert result.shape == (467,)  # CQ-0264: +4 menzen

    def test_all_hands_present(self, full_obs):
        """Full モードでは全4家の手牌が含まれる"""
        enc = FlatFeatureEncoder(observation_mode="full")
        result = enc.encode(full_obs)
        # Full: 4家手牌(136) + 4家河(136) + 4家副露(136) + ドラ(34) + スカラー(5) + スコア(4) + 立直(4) + menzen(4)
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
        assert meta.output_shape == (361,)
        assert meta.dtype == np.float32
        assert meta.observation_mode == "partial"
        assert meta.name == "FlatFeatureEncoder"

    def test_flat_full_metadata(self):
        enc = FlatFeatureEncoder(observation_mode="full")
        meta = enc.metadata()
        assert meta.output_shape == (467,)

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
        assert enc.output_dim == 361

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
        assert result.shape == (361,)

    def test_flat_both_accepts_full(self, full_obs):
        enc = FlatFeatureEncoder(observation_mode="both")
        result = enc.encode(full_obs)
        assert result.shape == (467,)

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
        """off 時 partial=361 を維持"""
        enc = FlatFeatureEncoder(observation_mode="partial", shanten_hint=False)
        assert enc.metadata().output_shape == (361,)

    def test_off_preserves_full_dim(self):
        """off 時 full=467 を維持"""
        enc = FlatFeatureEncoder(observation_mode="full", shanten_hint=False)
        assert enc.metadata().output_shape == (467,)

    def test_on_adds_34_partial(self):
        """on 時 partial=387"""
        enc = FlatFeatureEncoder(observation_mode="partial", shanten_hint=True)
        assert enc.metadata().output_shape == (395,)

    def test_on_adds_34_full(self):
        """on 時 full=489"""
        enc = FlatFeatureEncoder(observation_mode="full", shanten_hint=True)
        assert enc.metadata().output_shape == (501,)

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

        # base 部分 (353 dim) は一致する
        assert np.array_equal(result_on[:353], result_off[:353])

        # shanten hint range を metadata から取得
        ranges = enc_on.metadata().feature_ranges
        sh_start, sh_end = ranges["shanten_hint"]
        hint = result_on[sh_start:sh_end]
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
        """off 時 partial=361 を維持"""
        enc = FlatFeatureEncoder(observation_mode="partial", discard_ukeire_hint=False)
        assert enc.metadata().output_shape == (361,)

    def test_off_preserves_full_dim(self):
        """off 時 full=467 を維持"""
        enc = FlatFeatureEncoder(observation_mode="full", discard_ukeire_hint=False)
        assert enc.metadata().output_shape == (467,)

    def test_on_adds_34_partial(self):
        """on 時 partial=361+34=387"""
        enc = FlatFeatureEncoder(observation_mode="partial", discard_ukeire_hint=True)
        assert enc.metadata().output_shape == (395,)

    def test_on_adds_34_full(self):
        """on 時 full=467+34=493"""
        enc = FlatFeatureEncoder(observation_mode="full", discard_ukeire_hint=True)
        assert enc.metadata().output_shape == (501,)

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
        ranges = enc.metadata().feature_ranges
        s, e = ranges["discard_ukeire_hint"]
        ukeire = result[s:e]
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
        # 361 + 34 (shanten_hint) + 34 (discard_ukeire) = 421
        assert enc.metadata().output_shape == (429,)


class TestCurrentShantenInput:
    """policy 用 current_shanten テスト (CQ-0169)"""

    def test_off_preserves_partial_dim(self):
        """off 時 partial=361 を維持"""
        enc = FlatFeatureEncoder(observation_mode="partial", current_shanten_input=False)
        assert enc.metadata().output_shape == (361,)

    def test_on_adds_1_partial(self):
        """on 時 partial=362"""
        enc = FlatFeatureEncoder(observation_mode="partial", current_shanten_input=True)
        assert enc.metadata().output_shape == (362,)

    def test_on_adds_1_full(self):
        """on 時 full=456"""
        enc = FlatFeatureEncoder(observation_mode="full", current_shanten_input=True)
        assert enc.metadata().output_shape == (468,)

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
        """off 時 partial=361 を維持"""
        enc = FlatFeatureEncoder(observation_mode="partial", shape_hint=False)
        assert enc.metadata().output_shape == (361,)

    def test_on_adds_66_partial(self):
        """on 時 partial=361+66=419"""
        enc = FlatFeatureEncoder(observation_mode="partial", shape_hint=True)
        assert enc.metadata().output_shape == (427,)

    def test_on_adds_66_full(self):
        """on 時 full=467+66=525"""
        enc = FlatFeatureEncoder(observation_mode="full", shape_hint=True)
        assert enc.metadata().output_shape == (533,)

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
        ranges = enc.metadata().feature_ranges
        s, e = ranges["shape_hint"]
        shape_part = result[s:e]
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
        # 361 + 34 + 34 + 1 + 66 = 488
        assert enc.metadata().output_shape == (496,)

    def test_combined_all_features_full(self):
        """全オプション併用時の次元 (full)"""
        enc = FlatFeatureEncoder(
            observation_mode="full",
            shanten_hint=True,
            discard_ukeire_hint=True,
            current_shanten_input=True,
            shape_hint=True,
        )
        # 467 + 34 + 34 + 1 + 66 = 594
        assert enc.metadata().output_shape == (602,)


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


class TestFullObsCurrentPlayer:
    """CQ-0208: full 観測の補助特徴が current_player 基準であることを検証"""

    @staticmethod
    def _get_two_player_obs():
        """異なる current_player の FullObservation を 2 つ取得する"""
        from mahjong_rl import GameEngine, EnvironmentState, RunMode
        from mahjong_rl._mahjong_core import make_full_observation
        engine = GameEngine()
        env = EnvironmentState()
        env.run_mode = RunMode.Fast
        engine.reset_match(env, 123)

        obs0 = make_full_observation(env)
        p0 = obs0.current_player

        # step して別の current_player の局面を探す
        for _ in range(200):
            actions = engine.get_legal_actions(env)
            if not actions:
                break
            engine.step(env, actions[0])
            obs_next = make_full_observation(env)
            if obs_next.current_player != p0:
                return obs0, obs_next
        pytest.skip("同一 seed で current_player の異なる局面が得られなかった")

    def test_auxiliary_features_follow_current_player(self):
        """shanten_hint が current_player の手牌に基づくことを確認"""
        obs0, obs1 = self._get_two_player_obs()
        enc = FlatFeatureEncoder(
            observation_mode="full",
            shanten_hint=True,
            discard_ukeire_hint=True,
            current_shanten_input=True,
            shape_hint=True,
        )
        meta = enc.metadata()
        fr = meta.feature_ranges

        result0 = enc.encode(obs0)
        result1 = enc.encode(obs1)

        # 各補助特徴ブロックを取り出す
        for key in ["shanten_hint", "discard_ukeire_hint", "current_shanten", "shape_hint"]:
            s, e = fr[key]
            block0 = result0[s:e]
            block1 = result1[s:e]
            # current_player が異なるので、手牌も異なる → 補助特徴も異なるはず
            # (完全一致は偶然あり得るが、4 ブロック全部一致は極めて低確率)
            if not np.array_equal(block0, block1):
                return  # 少なくとも1ブロックが異なれば OK
        pytest.fail("全補助特徴ブロックが一致: current_player が反映されていない可能性")

    def test_shanten_hint_matches_current_player_hand(self):
        """shanten_hint が hands[current_player] から計算された値と一致する"""
        obs0, _ = self._get_two_player_obs()
        enc = FlatFeatureEncoder(
            observation_mode="full", shanten_hint=True)
        meta = enc.metadata()
        result = enc.encode(obs0)
        s, e = meta.feature_ranges["shanten_hint"]
        actual_hint = result[s:e]

        # current_player の手牌カウントを手動計算
        cp = obs0.current_player
        hand_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for tid in obs0.hands[cp]:
            hand_counts[tid // 4] += 1.0
        expected_hint = FlatFeatureEncoder._compute_shanten_hint(hand_counts.copy())
        np.testing.assert_array_equal(actual_hint, expected_hint)

    def test_current_shanten_matches_current_player_hand(self):
        """current_shanten が hands[current_player] から計算された値と一致する"""
        obs0, _ = self._get_two_player_obs()
        enc = FlatFeatureEncoder(
            observation_mode="full", current_shanten_input=True)
        meta = enc.metadata()
        result = enc.encode(obs0)
        s, e = meta.feature_ranges["current_shanten"]
        actual = result[s:e]

        cp = obs0.current_player
        hand_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for tid in obs0.hands[cp]:
            hand_counts[tid // 4] += 1.0
        expected = FlatFeatureEncoder._compute_current_shanten(hand_counts)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_shape_hint_matches_current_player_hand(self):
        """shape_hint が hands[current_player] から計算された値と一致する"""
        obs0, _ = self._get_two_player_obs()
        enc = FlatFeatureEncoder(
            observation_mode="full", shape_hint=True)
        meta = enc.metadata()
        result = enc.encode(obs0)
        s, e = meta.feature_ranges["shape_hint"]
        actual = result[s:e]

        cp = obs0.current_player
        hand_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for tid in obs0.hands[cp]:
            hand_counts[tid // 4] += 1.0
        expected = FlatFeatureEncoder._compute_shape_hint(hand_counts)
        np.testing.assert_array_equal(actual, expected)

    def test_discard_ukeire_matches_current_player_hand(self):
        """discard_ukeire_hint が hands[current_player] から計算された値と一致する"""
        obs0, _ = self._get_two_player_obs()
        enc = FlatFeatureEncoder(
            observation_mode="full", discard_ukeire_hint=True)
        meta = enc.metadata()

        # legal_mask を用意して encode に渡す
        mask = np.ones(NUM_TILE_TYPES, dtype=np.float32)
        result = enc.encode(obs0, legal_mask=mask)
        s, e = meta.feature_ranges["discard_ukeire_hint"]
        actual = result[s:e]

        # current_player の手牌から期待値を手計算
        cp = obs0.current_player
        hand_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for tid in obs0.hands[cp]:
            hand_counts[tid // 4] += 1.0
        expected = FlatFeatureEncoder._compute_discard_ukeire(
            hand_counts.copy(), legal_mask=mask)
        np.testing.assert_array_almost_equal(actual, expected, decimal=5)


class TestFeatureRanges:
    """feature_ranges テスト (CQ-0203)"""

    def test_no_options_has_always_on_ranges(self):
        """CQ-0267: self_tenpai_flag / remaining_draws_norm は常に入る"""
        enc = FlatFeatureEncoder(observation_mode="partial")
        meta = enc.metadata()
        assert "self_tenpai_flag" in meta.feature_ranges
        assert "remaining_draws_norm" in meta.feature_ranges

    def test_shanten_hint_range(self):
        enc = FlatFeatureEncoder(observation_mode="partial", shanten_hint=True)
        meta = enc.metadata()
        assert "shanten_hint" in meta.feature_ranges
        s, e = meta.feature_ranges["shanten_hint"]
        assert e - s == 34
        assert s == 353  # base partial dim

    def test_multiple_ranges(self):
        enc = FlatFeatureEncoder(
            observation_mode="partial",
            shanten_hint=True,
            discard_ukeire_hint=True,
            current_shanten_input=True,
        )
        meta = enc.metadata()
        assert "shanten_hint" in meta.feature_ranges
        assert "discard_ukeire_hint" in meta.feature_ranges
        assert "current_shanten" in meta.feature_ranges
        # 順序: shanten(353-387), ukeire(387-421), current(421-422)
        assert meta.feature_ranges["shanten_hint"] == (353, 387)
        assert meta.feature_ranges["discard_ukeire_hint"] == (387, 421)
        assert meta.feature_ranges["current_shanten"] == (421, 422)

    def test_full_mode_ranges(self):
        enc = FlatFeatureEncoder(
            observation_mode="full", shanten_hint=True)
        meta = enc.metadata()
        s, e = meta.feature_ranges["shanten_hint"]
        assert s == 459  # full base dim (CQ-0264: +4 menzen)


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
        assert enc.metadata().output_shape == (361,)

    def test_on_adds_4_partial(self):
        enc = FlatFeatureEncoder(observation_mode="partial", turn_context=True)
        assert enc.metadata().output_shape == (365,)

    def test_on_adds_4_full(self):
        enc = FlatFeatureEncoder(observation_mode="full", turn_context=True)
        assert enc.metadata().output_shape == (471,)

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
        ranges = enc.metadata().feature_ranges
        s, e = ranges["turn_context"]
        tc = result[s:e]
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
        # 361 + 34 + 34 + 1 + 66 + 4 = 492
        assert enc.metadata().output_shape == (500,)


class TestOpponentFeatures:
    """CQ-0213: opponent 防御特徴テスト"""

    def test_full_adds_opponent_features(self, full_obs):
        """full mode で opponent 特徴が正しい次元で追加される"""
        enc = FlatFeatureEncoder(
            observation_mode="full",
            opponent_current_shanten=True,
            opponent_tenpai_flag=True,
            danger_mask=True,
        )
        meta = enc.metadata()
        result = enc.encode(full_obs)
        assert result.shape == meta.output_shape
        # 459 + 3 + 3 + 34*3 = 567
        assert meta.output_shape == (575,)

    def test_full_feature_ranges(self, full_obs):
        """feature_ranges に新 feature が登録される"""
        enc = FlatFeatureEncoder(
            observation_mode="full",
            opponent_current_shanten=True,
            opponent_tenpai_flag=True,
            danger_mask=True,
        )
        fr = enc.metadata().feature_ranges
        assert "opponent_current_shanten" in fr
        assert "opponent_tenpai_flag" in fr
        assert "danger_mask_kamicha" in fr
        assert "danger_mask_toimen" in fr
        assert "danger_mask_shimo" in fr

    def test_partial_auto_off(self):
        """partial mode で opponent 特徴が自動的に無効化される"""
        enc = FlatFeatureEncoder(
            observation_mode="partial",
            opponent_current_shanten=True,
            opponent_tenpai_flag=True,
            danger_mask=True,
        )
        meta = enc.metadata()
        # partial base = 353, opponent features は含まれない
        assert meta.output_shape == (361,)
        fr = meta.feature_ranges
        assert "opponent_current_shanten" not in fr
        assert "danger_mask_kamicha" not in fr

    def test_opponent_shanten_values(self, full_obs):
        """opponent_current_shanten が [0,1] 範囲"""
        enc = FlatFeatureEncoder(
            observation_mode="full",
            opponent_current_shanten=True,
        )
        meta = enc.metadata()
        result = enc.encode(full_obs)
        s, e = meta.feature_ranges["opponent_current_shanten"]
        opp_sh = result[s:e]
        assert opp_sh.shape == (3,)
        assert np.all(opp_sh >= 0.0)
        assert np.all(opp_sh <= 1.0)

    def test_tenpai_flag_binary(self, full_obs):
        """opponent_tenpai_flag が 0/1"""
        enc = FlatFeatureEncoder(
            observation_mode="full",
            opponent_tenpai_flag=True,
        )
        meta = enc.metadata()
        result = enc.encode(full_obs)
        s, e = meta.feature_ranges["opponent_tenpai_flag"]
        tp = result[s:e]
        for v in tp:
            assert v in (0.0, 1.0)

    def test_danger_mask_binary(self, full_obs):
        """danger_mask が 0/1"""
        enc = FlatFeatureEncoder(
            observation_mode="full",
            danger_mask=True,
        )
        meta = enc.metadata()
        result = enc.encode(full_obs)
        for name in ("danger_mask_kamicha", "danger_mask_toimen", "danger_mask_shimo"):
            s, e = meta.feature_ranges[name]
            dm = result[s:e]
            assert dm.shape == (34,)
            for v in dm:
                assert v in (0.0, 1.0)

    def test_off_preserves_dim(self):
        """opponent features off で既存次元維持"""
        enc = FlatFeatureEncoder(observation_mode="full")
        assert enc.metadata().output_shape == (467,)

    def test_danger_mask_seat_order(self, full_obs):
        """CQ-0214: danger_mask の seat 順が shimo(+1), toimen(+2), kamicha(+3)"""
        enc = FlatFeatureEncoder(observation_mode="full", danger_mask=True)
        fr = enc.metadata().feature_ranges
        # feature_ranges のキー順を検証
        keys = [k for k in fr if k.startswith("danger_mask_")]
        assert keys == ["danger_mask_shimo", "danger_mask_toimen", "danger_mask_kamicha"]

    def test_danger_mask_source_in_direct_hints(self, full_obs):
        """CQ-0214: danger_mask_* を policy_direct_hints.sources に指定可能"""
        enc = FlatFeatureEncoder(
            observation_mode="full",
            danger_mask=True,
            shanten_hint=True,
        )
        fr = enc.metadata().feature_ranges
        # danger_mask_* が feature_ranges に存在すれば direct hint source として使える
        for name in ("danger_mask_shimo", "danger_mask_toimen", "danger_mask_kamicha"):
            assert name in fr
            s, e = fr[name]
            assert e - s == 34


class TestPartialMeldCountSelf:
    """CQ-0267 仕上げ: partial path の meld_count が self 基準"""

    def _make_partial_obs_observer(self, seed: int, target_observer: int):
        """target_observer が current_player の PartialObservation を返す"""
        from mahjong_rl._mahjong_core import Phase
        engine = GameEngine()
        env = EnvironmentState()
        engine.reset_match(env, seed)
        for _ in range(5000):
            p = env.round_state.phase
            if p in (Phase.EndRound, Phase.EndMatch):
                break
            cp = env.round_state.current_player
            if cp == target_observer:
                return make_partial_observation(env, cp)
            actions = engine.get_legal_actions(env)
            if not actions:
                break
            engine.step(env, actions[0])
        return None

    def test_observer_1_self_meld_count(self):
        """observer=1 の partial で meld_count が self 基準"""
        obs = self._make_partial_obs_observer(42, 1)
        if obs is None:
            pytest.skip("observer=1 が見つからなかった")
        assert obs.observer == 1
        # self の meld 数 (obs.melds は自家のみ)
        self_mc = len(obs.melds)
        # seat 0 の meld 数
        seat0_mc = len(obs.public_melds[0])
        # current_shanten は self 基準で計算されるべき
        from mahjong_rl.baseline.shanten import compute_shanten
        hc = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for tid in obs.hand:
            hc[tid // 4] += 1.0
        expected_sh = compute_shanten(hc, meld_count=self_mc)
        enc = FlatFeatureEncoder(
            observation_mode="partial", current_shanten_input=True)
        feat = enc.encode(obs)
        cs_range = enc.metadata().feature_ranges["current_shanten"]
        cs = feat[cs_range[0]:cs_range[1]]
        assert cs[0] == pytest.approx(expected_sh / 8.0)

    def test_observer_2_tenpai_flag_self(self):
        """observer=2 の partial で self_tenpai_flag が self 基準"""
        obs = self._make_partial_obs_observer(42, 2)
        if obs is None:
            pytest.skip()
        from mahjong_rl.baseline.shanten import compute_shanten
        hc = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for tid in obs.hand:
            hc[tid // 4] += 1.0
        self_mc = len(obs.melds)
        sh = compute_shanten(hc, meld_count=self_mc)
        expected_flag = 1.0 if sh == 0 else 0.0
        enc = FlatFeatureEncoder(observation_mode="partial")
        feat = enc.encode(obs)
        tf_range = enc.metadata().feature_ranges["self_tenpai_flag"]
        tf = feat[tf_range[0]:tf_range[1]]
        assert tf[0] == expected_flag

    def test_observer_3_hint_self_basis(self):
        """observer=3 の partial で shanten_hint が self 基準"""
        obs = self._make_partial_obs_observer(42, 3)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(
            observation_mode="partial", shanten_hint=True)
        feat = enc.encode(obs)
        # should not crash and produce valid hints
        sh_range = enc.metadata().feature_ranges["shanten_hint"]
        hint = feat[sh_range[0]:sh_range[1]]
        assert hint.shape == (34,)
        for v in hint:
            assert v in (-1.0, 0.0)


# ========== CQ-0270: tile presence flags ==========

class TestTilePresenceFlags:
    """tile presence flags テスト"""

    def test_honor_present(self):
        """手牌に字牌 → has_honor=1"""
        from mahjong_rl.encoders.flat_encoder import FlatFeatureEncoder
        # 東(27*4=108) だけの hand + empty melds
        class FakeMeld:
            tile_count = 0
            tiles = []
        flags = FlatFeatureEncoder._compute_tile_presence_flags([108], [])
        assert flags[0] == 1.0  # has_honor
        assert flags[1] == 0.0  # has_terminal (東 is honor, not terminal)
        assert flags[2] == 0.0  # has_simple

    def test_terminal_present(self):
        """手牌に 1m → has_terminal=1, has_man=1"""
        flags = FlatFeatureEncoder._compute_tile_presence_flags([0], [])
        assert flags[0] == 0.0  # has_honor
        assert flags[1] == 1.0  # has_terminal
        assert flags[3] == 1.0  # has_man

    def test_simple_present(self):
        """手牌に 5m(tile_id=16) → has_simple=1"""
        flags = FlatFeatureEncoder._compute_tile_presence_flags([16], [])
        assert flags[2] == 1.0  # has_simple

    def test_suit_detection(self):
        """各スーツの検出"""
        # pin: 1p = tile_type 9, tile_id 36
        flags = FlatFeatureEncoder._compute_tile_presence_flags([36], [])
        assert flags[3] == 0.0  # has_man
        assert flags[4] == 1.0  # has_pin
        assert flags[5] == 0.0  # has_sou
        # sou: 1s = tile_type 18, tile_id 72
        flags = FlatFeatureEncoder._compute_tile_presence_flags([72], [])
        assert flags[5] == 1.0  # has_sou

    def test_meld_only_counts(self):
        """meld にだけ存在する牌も検出される"""
        class FakeMeld:
            tile_count = 3
            tiles = [108, 109, 110]  # 東×3 (tile_type 27)
        flags = FlatFeatureEncoder._compute_tile_presence_flags(
            [0],  # hand: 1m only
            [FakeMeld()])
        assert flags[0] == 1.0  # has_honor (from meld)
        assert flags[1] == 1.0  # has_terminal (from hand: 1m)

    def test_all_absent(self):
        """空手牌は全 0"""
        flags = FlatFeatureEncoder._compute_tile_presence_flags([], [])
        np.testing.assert_array_equal(flags, [0, 0, 0, 0, 0, 0])

    def test_metadata_range(self):
        """feature_ranges に tile_presence_flags がある"""
        enc = FlatFeatureEncoder(observation_mode="full")
        meta = enc.metadata()
        assert "tile_presence_flags" in meta.feature_ranges
        s, e = meta.feature_ranges["tile_presence_flags"]
        assert e - s == 6

    def test_full_partial_dim(self):
        """dim 更新"""
        enc_f = FlatFeatureEncoder(observation_mode="full")
        enc_p = FlatFeatureEncoder(observation_mode="partial")
        assert enc_f.metadata().output_shape == (467,)
        assert enc_p.metadata().output_shape == (361,)

    def test_encode_full_no_crash(self, full_obs):
        """full encode が crash しない"""
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(full_obs)
        assert feat.shape == enc.metadata().output_shape
        # tile_presence_flags range
        s, e = enc.metadata().feature_ranges["tile_presence_flags"]
        tp = feat[s:e]
        assert tp.shape == (6,)
        for v in tp:
            assert v in (0.0, 1.0)

    def test_encode_partial_no_crash(self, partial_obs):
        """partial encode が crash しない"""
        enc = FlatFeatureEncoder(observation_mode="partial")
        feat = enc.encode(partial_obs)
        assert feat.shape == enc.metadata().output_shape
        s, e = enc.metadata().feature_ranges["tile_presence_flags"]
        tp = feat[s:e]
        assert tp.shape == (6,)
        for v in tp:
            assert v in (0.0, 1.0)
