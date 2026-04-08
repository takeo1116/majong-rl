"""CQ-0263: full observation actor-relative rotation テスト"""
import pytest
import numpy as np

pytestmark = pytest.mark.smoke

from mahjong_rl._mahjong_core import (
    GameEngine, EnvironmentState, RunMode, Phase,
    make_full_observation, make_partial_observation,
    NUM_TILE_TYPES, NUM_PLAYERS,
)
from mahjong_rl.encoders.flat_encoder import FlatFeatureEncoder


# ========== helpers ==========

def _make_obs_at_player(seed: int, target_player: int | None = None):
    """指定 player が current_player の FullObservation を返す"""
    e = GameEngine()
    env = EnvironmentState()
    e.reset_match(env, seed, RunMode.Fast)
    # target_player が None なら初期状態のまま
    if target_player is not None:
        for _ in range(5000):
            p = env.round_state.phase
            if p in (Phase.EndRound, Phase.EndMatch):
                break
            cp = env.round_state.current_player
            if cp == target_player:
                return make_full_observation(env)
            actions = e.get_legal_actions(env)
            if not actions:
                break
            e.step(env, actions[0])
    return make_full_observation(env)


def _hand_counts_from_obs(obs, player: int) -> np.ndarray:
    """obs から player の手牌カウントを作る"""
    c = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    for tid in obs.hands[player]:
        c[tid // 4] += 1.0
    return c


def _extract_hands_block(features: np.ndarray, idx: int) -> np.ndarray:
    """features から idx 番目の手牌ブロック (34 dim) を取り出す"""
    start = idx * NUM_TILE_TYPES
    return features[start:start + NUM_TILE_TYPES]


def _extract_discards_block(features: np.ndarray, idx: int) -> np.ndarray:
    """4家手牌(136) の後に 4家河(136) があるので offset=136"""
    base = 4 * NUM_TILE_TYPES  # hands block end
    start = base + idx * NUM_TILE_TYPES
    return features[start:start + NUM_TILE_TYPES]


def _extract_melds_block(features: np.ndarray, idx: int) -> np.ndarray:
    """hands(136) + discards(136) の後に melds(136)"""
    base = 8 * NUM_TILE_TYPES
    start = base + idx * NUM_TILE_TYPES
    return features[start:start + NUM_TILE_TYPES]


def _extract_scalars(features: np.ndarray) -> np.ndarray:
    """hands(136) + discards(136) + melds(136) + dora(34) の後 5 scalar"""
    base = 12 * NUM_TILE_TYPES + NUM_TILE_TYPES  # 12*34 + 34 = 442
    return features[base:base + 5]


def _extract_scores(features: np.ndarray) -> np.ndarray:
    """scalars(5) の後 4 scores"""
    base = 12 * NUM_TILE_TYPES + NUM_TILE_TYPES + 5  # 447
    return features[base:base + 4]


def _extract_riichi(features: np.ndarray) -> np.ndarray:
    """scores(4) の後 4 riichi"""
    base = 12 * NUM_TILE_TYPES + NUM_TILE_TYPES + 5 + 4  # 451
    return features[base:base + 4]


def _extract_menzen(features: np.ndarray) -> np.ndarray:
    """riichi(4) の後 4 menzen"""
    base = 12 * NUM_TILE_TYPES + NUM_TILE_TYPES + 5 + 4 + 4  # 455
    return features[base:base + 4]


# ========== full observation rotation tests ==========

class TestFullObsRotationPlayer0:
    """current_player=0 のとき: seat_order=[0,1,2,3] → 従来と一致"""

    def test_hands_block_0_is_self(self):
        obs = _make_obs_at_player(42, 0)
        if obs is None:
            pytest.skip("player 0 が見つからなかった")
        assert obs.current_player == 0
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        expected = _hand_counts_from_obs(obs, 0)
        np.testing.assert_array_equal(_extract_hands_block(feat, 0), expected)

    def test_hands_block_1_is_player1(self):
        obs = _make_obs_at_player(42, 0)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        expected = _hand_counts_from_obs(obs, 1)
        np.testing.assert_array_equal(_extract_hands_block(feat, 1), expected)


class TestFullObsRotationPlayer1:
    """current_player=1: seat_order=[1,2,3,0]"""

    def test_hands_block_0_is_player1(self):
        obs = _make_obs_at_player(42, 1)
        if obs is None:
            pytest.skip("player 1 が見つからなかった")
        assert obs.current_player == 1
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        expected = _hand_counts_from_obs(obs, 1)
        np.testing.assert_array_equal(_extract_hands_block(feat, 0), expected)

    def test_hands_block_1_is_player2(self):
        obs = _make_obs_at_player(42, 1)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        expected = _hand_counts_from_obs(obs, 2)
        np.testing.assert_array_equal(_extract_hands_block(feat, 1), expected)

    def test_hands_block_3_is_player0(self):
        obs = _make_obs_at_player(42, 1)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        expected = _hand_counts_from_obs(obs, 0)
        np.testing.assert_array_equal(_extract_hands_block(feat, 3), expected)


class TestFullObsRotationPlayer2:
    """current_player=2: seat_order=[2,3,0,1]"""

    def test_hands_block_0_is_player2(self):
        obs = _make_obs_at_player(42, 2)
        if obs is None:
            pytest.skip()
        assert obs.current_player == 2
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        expected = _hand_counts_from_obs(obs, 2)
        np.testing.assert_array_equal(_extract_hands_block(feat, 0), expected)


class TestFullObsRotationPlayer3:
    """current_player=3: seat_order=[3,0,1,2]"""

    def test_hands_block_0_is_player3(self):
        obs = _make_obs_at_player(42, 3)
        if obs is None:
            pytest.skip()
        assert obs.current_player == 3
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        expected = _hand_counts_from_obs(obs, 3)
        np.testing.assert_array_equal(_extract_hands_block(feat, 0), expected)


class TestDiscardsAndMeldsRotation:
    """discards / melds も同じ actor-relative 順"""

    def test_discards_rotation(self):
        """current_player=1 の discards block 0 が player1"""
        obs = _make_obs_at_player(42, 1)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        # discards block 0 should be player 1's discards
        d_block = _extract_discards_block(feat, 0)
        expected = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for di in obs.discards[1]:
            expected[di.tile // 4] += 1.0
        np.testing.assert_array_equal(d_block, expected)

    def test_melds_rotation(self):
        """current_player=1 の melds block 0 が player1"""
        obs = _make_obs_at_player(42, 1)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        m_block = _extract_melds_block(feat, 0)
        expected = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for meld in obs.melds[1]:
            for i in range(meld.tile_count):
                tiles = meld.tiles
                if i < len(tiles):
                    expected[tiles[i] // 4] += 1.0
        np.testing.assert_array_equal(m_block, expected)


class TestScoresRotation:
    """scores も actor-relative 順"""

    def test_score_0_is_self(self):
        obs = _make_obs_at_player(42, 1)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        scores = _extract_scores(feat)
        assert scores[0] == pytest.approx(obs.scores[1] / 100000.0)

    def test_score_1_is_shimo(self):
        obs = _make_obs_at_player(42, 1)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        scores = _extract_scores(feat)
        assert scores[1] == pytest.approx(obs.scores[2] / 100000.0)


# ========== dealer relative tests ==========

class TestDealerRelative:
    """dealer が actor-relative"""

    def test_self_is_dealer(self):
        """actor = dealer → relative_dealer = 0"""
        obs = _make_obs_at_player(42, None)
        if obs is None:
            pytest.skip()
        # Find a case where actor == dealer
        cp = obs.current_player
        dealer = obs.dealer
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        scalars = _extract_scalars(feat)
        expected_rel = (dealer - cp) % 4
        assert scalars[1] == pytest.approx(expected_rel / 3.0)

    def test_dealer_relative_player1(self):
        """actor=1, dealer=0 → relative_dealer=3 (kamicha)"""
        obs = _make_obs_at_player(42, 1)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        scalars = _extract_scalars(feat)
        expected_rel = (obs.dealer - 1) % 4
        assert scalars[1] == pytest.approx(expected_rel / 3.0)


# ========== consistency with hint features ==========

class TestHintConsistencyAfterRotation:
    """rotation 後も shanten_hint / current_shanten は actor 自身基準"""

    def test_current_shanten_is_self(self):
        """current_shanten は block 0 (self) の手牌から計算される"""
        obs = _make_obs_at_player(42, 1)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(
            observation_mode="full", current_shanten_input=True)
        feat = enc.encode(obs)
        # current_shanten は feature の末尾付近
        meta = enc.metadata()
        ranges = meta.feature_ranges
        cs_range = ranges["current_shanten"]
        cs = feat[cs_range[0]:cs_range[1]]
        # 手動で player 1 の shanten を計算
        from mahjong_rl.baseline.shanten import compute_shanten
        hc = _hand_counts_from_obs(obs, 1)
        mc = len(obs.melds[1])
        expected_sh = compute_shanten(hc, meld_count=mc) / 8.0
        assert cs[0] == pytest.approx(expected_sh)


class TestOpponentFeaturesConsistency:
    """opponent features が hands block と同じ相対順"""

    def test_opponent_shanten_order_matches_blocks(self):
        """opponent_current_shanten[0] = shimo = block[1] の player"""
        obs = _make_obs_at_player(42, 1)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(
            observation_mode="full", opponent_current_shanten=True)
        feat = enc.encode(obs)
        meta = enc.metadata()
        ranges = meta.feature_ranges
        opp_range = ranges["opponent_current_shanten"]
        opp_sh = feat[opp_range[0]:opp_range[1]]

        from mahjong_rl.baseline.shanten import compute_shanten
        # shimo of player 1 = player 2
        hc2 = _hand_counts_from_obs(obs, 2)
        mc2 = len(obs.melds[2])
        expected = compute_shanten(hc2, meld_count=mc2) / 8.0
        assert opp_sh[0] == pytest.approx(expected)


# ========== partial regression ==========

class TestPartialRegression:
    """partial encoder は回帰しない"""

    def test_partial_dim_unchanged(self):
        e = GameEngine()
        env = EnvironmentState()
        e.reset_match(env, 42, RunMode.Fast)
        player = env.round_state.current_player
        obs = make_partial_observation(env, player)
        enc = FlatFeatureEncoder(observation_mode="partial")
        feat = enc.encode(obs)
        assert feat.shape == (361,)

    def test_partial_hand_nonnegative(self):
        e = GameEngine()
        env = EnvironmentState()
        e.reset_match(env, 42, RunMode.Fast)
        player = env.round_state.current_player
        obs = make_partial_observation(env, player)
        enc = FlatFeatureEncoder(observation_mode="partial")
        feat = enc.encode(obs)
        hand = feat[:NUM_TILE_TYPES]
        assert np.all(hand >= 0)
        assert hand.sum() in (13.0, 14.0)


# ========== Stage2 smoke ==========

class TestStage2ModelSmoke:
    """Stage2 model が rotation 後の encoder で動く"""

    def test_smoke_disabled_semantic(self):
        from mahjong_rl.models.stage2a_model import Stage2aModel
        import torch
        enc = FlatFeatureEncoder(observation_mode="full")
        dim = enc.metadata().output_shape[0]
        model = Stage2aModel(input_dim=dim, discard_hidden_dims=[32],
                              optional_hidden_dims=[32])
        x = torch.randn(2, dim)
        m = torch.ones(2, 34)
        out = model.forward_discard(x, m)
        assert out.discard_logits.shape == (2, 34)

    def test_smoke_enabled_semantic(self):
        from mahjong_rl.models.stage2a_model import Stage2aModel
        import torch
        enc = FlatFeatureEncoder(observation_mode="full")
        dim = enc.metadata().output_shape[0]
        model = Stage2aModel(
            input_dim=dim, discard_hidden_dims=[32],
            optional_hidden_dims=[32], value_hidden_dims=[32],
            semantic_aux_config={"enabled": True, "policy_projection_dim": 8})
        x = torch.randn(2, dim)
        m = torch.ones(2, 34)
        out = model.forward_discard(x, m)
        assert out.discard_logits.shape == (2, 34)
        assert out.semantic is not None

    def test_evaluator_smoke(self):
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
        from mahjong_rl.models.stage2a_model import Stage2aModel
        import torch
        enc = FlatFeatureEncoder(observation_mode="full")
        dim = enc.metadata().output_shape[0]
        model = Stage2aModel(input_dim=dim, discard_hidden_dims=[32],
                              optional_hidden_dims=[32])
        evaluator = Stage2aEvaluator(
            model=model, encoder=enc, device=torch.device("cpu"))
        result = evaluator.evaluate(num_matches=1, seed_start=42)
        assert result["total_rounds"] > 0


# ========== CQ-0264: riichi / menzen ==========

class TestRiichiBlock:
    """riichi block が actor-relative 順で実値"""

    def test_riichi_all_false_at_start(self):
        """開始直後は全員 riichi=False"""
        obs = _make_obs_at_player(42, None)
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        riichi = _extract_riichi(feat)
        np.testing.assert_array_equal(riichi, [0.0, 0.0, 0.0, 0.0])

    def test_riichi_rotation_player1(self):
        """current_player=1 で riichi block が actor-relative 順"""
        obs = _make_obs_at_player(42, 1)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        riichi = _extract_riichi(feat)
        # actor-relative: [p1, p2, p3, p0]
        cp = 1
        seat_order = [(cp + off) % 4 for off in range(4)]
        expected = [1.0 if obs.riichi_declared[p] else 0.0 for p in seat_order]
        np.testing.assert_array_equal(riichi, expected)

    def test_riichi_rotation_player2(self):
        obs = _make_obs_at_player(42, 2)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        riichi = _extract_riichi(feat)
        cp = 2
        seat_order = [(cp + off) % 4 for off in range(4)]
        expected = [1.0 if obs.riichi_declared[p] else 0.0 for p in seat_order]
        np.testing.assert_array_equal(riichi, expected)

    def test_riichi_rotation_player3(self):
        obs = _make_obs_at_player(42, 3)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        riichi = _extract_riichi(feat)
        cp = 3
        seat_order = [(cp + off) % 4 for off in range(4)]
        expected = [1.0 if obs.riichi_declared[p] else 0.0 for p in seat_order]
        np.testing.assert_array_equal(riichi, expected)


class TestMenzenBlock:
    """menzen block が actor-relative 順"""

    def test_menzen_all_true_at_start(self):
        """開始直後は全員 menzen=True"""
        obs = _make_obs_at_player(42, None)
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        menzen = _extract_menzen(feat)
        np.testing.assert_array_equal(menzen, [1.0, 1.0, 1.0, 1.0])

    def test_menzen_rotation_player1(self):
        obs = _make_obs_at_player(42, 1)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        menzen = _extract_menzen(feat)
        cp = 1
        seat_order = [(cp + off) % 4 for off in range(4)]
        expected = [1.0 if obs.menzen_flags[p] else 0.0 for p in seat_order]
        np.testing.assert_array_equal(menzen, expected)

    def test_menzen_rotation_player2(self):
        obs = _make_obs_at_player(42, 2)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        menzen = _extract_menzen(feat)
        cp = 2
        seat_order = [(cp + off) % 4 for off in range(4)]
        expected = [1.0 if obs.menzen_flags[p] else 0.0 for p in seat_order]
        np.testing.assert_array_equal(menzen, expected)

    def test_menzen_rotation_player3(self):
        obs = _make_obs_at_player(42, 3)
        if obs is None:
            pytest.skip()
        enc = FlatFeatureEncoder(observation_mode="full")
        feat = enc.encode(obs)
        menzen = _extract_menzen(feat)
        cp = 3
        seat_order = [(cp + off) % 4 for off in range(4)]
        expected = [1.0 if obs.menzen_flags[p] else 0.0 for p in seat_order]
        np.testing.assert_array_equal(menzen, expected)


class TestMetadataUpdate:
    """metadata / output dim が更新されている"""

    def test_full_dim_includes_menzen(self):
        enc = FlatFeatureEncoder(observation_mode="full")
        meta = enc.metadata()
        assert meta.output_shape == (467,)

    def test_menzen_in_feature_ranges(self):
        enc = FlatFeatureEncoder(observation_mode="full")
        meta = enc.metadata()
        assert "menzen" in meta.feature_ranges
        start, end = meta.feature_ranges["menzen"]
        assert end - start == 4

    def test_riichi_in_feature_ranges(self):
        """riichi が feature_ranges にある"""
        enc = FlatFeatureEncoder(observation_mode="full")
        meta = enc.metadata()
        assert "riichi" in meta.feature_ranges
        start, end = meta.feature_ranges["riichi"]
        assert end - start == 4

    def test_riichi_and_menzen_both_present(self):
        """riichi と menzen の両方が metadata にある"""
        enc = FlatFeatureEncoder(observation_mode="full")
        meta = enc.metadata()
        assert "riichi" in meta.feature_ranges
        assert "menzen" in meta.feature_ranges
        # riichi は menzen の直前
        r_end = meta.feature_ranges["riichi"][1]
        m_start = meta.feature_ranges["menzen"][0]
        assert r_end == m_start

    def test_partial_no_riichi_menzen_ranges(self):
        """partial mode では riichi / menzen は ranges に出ない"""
        enc = FlatFeatureEncoder(observation_mode="partial")
        meta = enc.metadata()
        assert "riichi" not in meta.feature_ranges
        assert "menzen" not in meta.feature_ranges

    def test_partial_dim_unchanged(self):
        enc = FlatFeatureEncoder(observation_mode="partial")
        meta = enc.metadata()
        assert meta.output_shape == (361,)
