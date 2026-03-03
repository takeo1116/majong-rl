"""CQ-0028: pybind11 バインディングテスト"""
import pytest
import mahjong_rl
from mahjong_rl import (
    GameEngine, EnvironmentState, Action, ActionType, Phase, ErrorCode,
    RunMode, EventType, RoundEndReason, MeldType,
    PartialObservation, FullObservation,
    make_partial_observation, make_full_observation,
    make_type_counts, is_agari, is_tenpai, get_waits,
    Tile, NUM_TILES, NUM_TILE_TYPES, NUM_PLAYERS,
)


class TestImport:
    """import 確認"""

    def test_import_module(self):
        assert hasattr(mahjong_rl, "GameEngine")

    def test_constants(self):
        assert NUM_TILES == 136
        assert NUM_TILE_TYPES == 34
        assert NUM_PLAYERS == 4

    def test_enums(self):
        assert ActionType.Discard is not None
        assert Phase.SelfActionPhase is not None
        assert ErrorCode.Ok is not None
        assert RunMode.Fast is not None


class TestBasicFlow:
    """reset → get_legal_actions → step の基本フロー"""

    def test_reset_and_get_actions(self, engine, env):
        engine.reset_match(env, 42)
        actions = engine.get_legal_actions(env)
        assert len(actions) > 0
        assert env.round_state.phase == Phase.SelfActionPhase

    def test_step_returns_step_result(self, engine, initialized_env):
        actions = engine.get_legal_actions(initialized_env)
        result = engine.step(initialized_env, actions[0])
        assert result.error == ErrorCode.Ok
        assert isinstance(result.round_over, bool)
        assert isinstance(result.match_over, bool)
        assert len(result.rewards) == NUM_PLAYERS

    def test_reset_with_dealer(self, engine, env):
        engine.reset_match(env, 42, 2)
        assert env.match_state.first_dealer == 2

    def test_reset_with_mode(self, engine, env):
        engine.reset_match(env, 42, RunMode.Debug)
        assert env.run_mode == RunMode.Debug


class TestStructAccess:
    """構造体フィールドアクセス"""

    def test_action_fields(self, engine, initialized_env):
        actions = engine.get_legal_actions(initialized_env)
        a = actions[0]
        assert hasattr(a, "type")
        assert hasattr(a, "actor")
        assert hasattr(a, "tile")
        assert hasattr(a, "riichi")
        assert a.to_string() is not None

    def test_player_state_fields(self, initialized_env):
        players = initialized_env.round_state.players
        assert len(players) == NUM_PLAYERS
        p = players[0]
        assert len(p.hand) >= 13
        assert isinstance(p.is_riichi, bool)
        assert isinstance(p.score, int)

    def test_match_state_fields(self, initialized_env):
        ms = initialized_env.match_state
        assert len(ms.scores) == NUM_PLAYERS
        assert ms.is_match_over is False
        assert ms.round_number == 0

    def test_round_state_fields(self, initialized_env):
        rs = initialized_env.round_state
        assert rs.round_number == 0
        assert rs.dealer >= 0
        assert rs.current_player >= 0

    def test_event_fields(self, engine, initialized_env):
        actions = engine.get_legal_actions(initialized_env)
        result = engine.step(initialized_env, actions[0])
        for evt in result.events:
            assert hasattr(evt, "type")
            assert evt.to_string() is not None

    def test_tile_static_methods(self):
        tile = Tile.from_id(0)
        assert tile.id == 0
        assert tile.type == 0
        assert tile.to_string() is not None


class TestObservation:
    """Observation 生成テスト"""

    def test_partial_observation(self, initialized_env):
        obs = make_partial_observation(initialized_env, 0)
        assert isinstance(obs, PartialObservation)
        assert obs.observer == 0
        assert len(obs.hand) >= 13
        assert len(obs.scores) == NUM_PLAYERS
        assert len(obs.dora_indicators) >= 1

    def test_full_observation(self, initialized_env):
        obs = make_full_observation(initialized_env)
        assert isinstance(obs, FullObservation)
        assert len(obs.hands) == NUM_PLAYERS
        for hand in obs.hands:
            assert len(hand) >= 13
        assert len(obs.scores) == NUM_PLAYERS
        assert len(obs.wall) == NUM_TILES

    def test_full_partial_switching(self, initialized_env):
        """同一環境で Full / Partial 両方を取得できる"""
        obs_p = make_partial_observation(initialized_env, 0)
        obs_f = make_full_observation(initialized_env)
        # Partial は自家手牌のみ、Full は全員
        assert len(obs_p.hand) == len(obs_f.hands[0])
        # Partial は他家手牌を持たないが Full は持つ
        assert len(obs_f.hands[1]) >= 13

    def test_partial_observation_all_players(self, initialized_env):
        """全プレイヤーの部分観測を生成できる"""
        for p in range(NUM_PLAYERS):
            obs = make_partial_observation(initialized_env, p)
            assert obs.observer == p
            assert len(obs.hand) >= 13


class TestHandUtils:
    """hand_utils 関数のテスト"""

    def test_make_type_counts(self, initialized_env):
        hand = initialized_env.round_state.players[0].hand
        counts = make_type_counts(list(hand))
        assert len(counts) == NUM_TILE_TYPES
        assert sum(counts) == len(hand)

    def test_is_tenpai(self):
        # 1m2m3m 4m5m6m 7m8m9m 1p1p1p 2p — テンパイ (2p or 3p 待ち)
        hand_counts = [1, 1, 1, 1, 1, 1, 1, 1, 1,
                       3, 1, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0]
        assert is_tenpai(hand_counts)


class TestFullMatch:
    """半荘完走テスト"""

    def test_full_match_completes(self, engine, env):
        engine.reset_match(env, 42)
        steps = 0
        while not env.match_state.is_match_over and steps < 10000:
            actions = engine.get_legal_actions(env)
            if not actions:
                break
            result = engine.step(env, actions[0])
            if result.error != ErrorCode.Ok:
                break
            if result.round_over and not result.match_over:
                engine.advance_round(env)
            steps += 1
        assert env.match_state.is_match_over
        assert steps > 0

    def test_seed_reproducibility(self, engine):
        """同一 seed で同一結果"""
        def run_match(seed):
            env = EnvironmentState()
            engine.reset_match(env, seed)
            steps = 0
            while not env.match_state.is_match_over and steps < 10000:
                actions = engine.get_legal_actions(env)
                if not actions:
                    break
                result = engine.step(env, actions[0])
                if result.error != ErrorCode.Ok:
                    break
                if result.round_over and not result.match_over:
                    engine.advance_round(env)
                steps += 1
            return list(env.match_state.scores), steps

        scores1, steps1 = run_match(42)
        scores2, steps2 = run_match(42)
        assert scores1 == scores2
        assert steps1 == steps2
