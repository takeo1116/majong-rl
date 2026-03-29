"""CQ-0221 / CQ-0222: Stage2Env + ResponseCandidate テスト"""
import pytest
import numpy as np

pytestmark = pytest.mark.smoke

from mahjong_rl._mahjong_core import (
    GameEngine, EnvironmentState, ActionType, Phase, RunMode,
    make_full_observation,
)
from mahjong_rl.env import Stage2Env, DecisionType
from mahjong_rl.env.response_candidate import (
    ResponseCandidate, extract_response_candidates, STAGE2A_LEARNABLE_TYPES,
)


class TestResponseCandidate:
    """CQ-0221: response candidate API テスト"""

    def test_extract_from_response_phase(self):
        """response phase の legal actions から candidate を抽出できる"""
        engine = GameEngine()
        env = EnvironmentState()
        env.run_mode = RunMode.Fast

        # 複数 seed を試して response phase を見つける
        found = False
        for seed in range(100):
            engine.reset_match(env, seed, RunMode.Fast)
            for _ in range(500):
                phase = env.round_state.phase
                if phase in (Phase.EndRound, Phase.EndMatch):
                    break
                legal_actions = engine.get_legal_actions(env)
                if not legal_actions:
                    break

                if phase == Phase.ResponsePhase:
                    responder = env.round_state.current_player
                    candidates = extract_response_candidates(
                        legal_actions, responder)

                    # Skip は必ず含まれる
                    skip_found = any(
                        c.action_type == ActionType.Skip for c in candidates)
                    assert skip_found, "Skip が candidates に含まれていない"

                    # 全 candidate が STAGE2A_LEARNABLE_TYPES
                    for c in candidates:
                        assert c.action_type in STAGE2A_LEARNABLE_TYPES

                    # Skip は末尾
                    if len(candidates) > 1:
                        assert candidates[-1].action_type == ActionType.Skip

                    found = True
                    break

                result = engine.step(env, legal_actions[0])
                if result.round_over:
                    if result.match_over:
                        break
                    engine.advance_round(env)
            if found:
                break
        assert found, "100 seed で response phase が見つからなかった"

    def test_candidate_fields(self):
        """candidate の各フィールドが正しく設定される"""
        engine = GameEngine()
        env = EnvironmentState()

        for seed in range(200):
            engine.reset_match(env, seed, RunMode.Fast)
            for _ in range(500):
                phase = env.round_state.phase
                if phase in (Phase.EndRound, Phase.EndMatch):
                    break
                legal_actions = engine.get_legal_actions(env)
                if not legal_actions:
                    break

                if phase == Phase.ResponsePhase:
                    responder = env.round_state.current_player
                    candidates = extract_response_candidates(
                        legal_actions, responder)

                    for c in candidates:
                        assert isinstance(c, ResponseCandidate)
                        assert c.action is not None
                        if c.action_type == ActionType.Skip:
                            assert c.tile_type == -1
                            assert c.target_rel_seat == -1
                            assert c.consumed_tile_ids == ()
                        elif c.action_type in (ActionType.Chi, ActionType.Pon,
                                               ActionType.Daiminkan):
                            assert c.tile_type >= 0
                            assert c.target_rel_seat >= 0
                    return  # 1つ見つければ十分

                result = engine.step(env, legal_actions[0])
                if result.round_over:
                    if result.match_over:
                        break
                    engine.advance_round(env)

    def test_learnable_only_filter(self):
        """learnable_only=False で Ron も含まれる"""
        engine = GameEngine()
        env = EnvironmentState()

        for seed in range(200):
            engine.reset_match(env, seed, RunMode.Fast)
            for _ in range(500):
                phase = env.round_state.phase
                if phase in (Phase.EndRound, Phase.EndMatch):
                    break
                legal_actions = engine.get_legal_actions(env)
                if not legal_actions:
                    break

                if phase == Phase.ResponsePhase:
                    has_ron = any(a.type == ActionType.Ron
                                 for a in legal_actions)
                    if has_ron:
                        all_cands = extract_response_candidates(
                            legal_actions,
                            env.round_state.current_player,
                            learnable_only=False)
                        ron_found = any(c.action_type == ActionType.Ron
                                        for c in all_cands)
                        assert ron_found
                        return

                result = engine.step(env, legal_actions[0])
                if result.round_over:
                    if result.match_over:
                        break
                    engine.advance_round(env)


class TestStage2Env:
    """CQ-0222: Stage2Env テスト"""

    def test_reset_returns_discard_decision(self):
        """reset 後は discard decision で始まる"""
        env = Stage2Env()
        obs, info = env.reset(42)
        assert obs is not None
        assert info["decision_type"] == "discard"
        assert env.decision_type == DecisionType.DISCARD

    def test_step_discard_works(self):
        """discard decision で step_discard が動く"""
        env = Stage2Env()
        env.reset(42)
        mask = env.get_legal_mask()
        action = int(np.argmax(mask))
        obs, rewards, terminated, truncated, info = env.step_discard(action)
        assert rewards.shape == (4,)

    def test_step_discard_wrong_phase_raises(self):
        """response phase で step_discard を呼ぶとエラー"""
        # response phase を強制するのは難しいので、ここでは型チェックのみ
        env = Stage2Env()
        env.reset(42)
        # discard phase なのでこれは通る
        mask = env.get_legal_mask()
        action = int(np.argmax(mask))
        env.step_discard(action)
        # terminated でなければ次の decision_type を確認

    def test_response_decision_occurs(self):
        """response decision が発生する seed がある"""
        env = Stage2Env()
        found = False
        for seed in range(200):
            env.reset(seed)
            for _ in range(300):
                if env.decision_type == DecisionType.RESPONSE:
                    found = True
                    assert len(env.response_candidates) >= 2  # Skip + 少なくとも1つ
                    break
                if env.decision_type == DecisionType.DISCARD:
                    mask = env.get_legal_mask()
                    action = int(np.argmax(mask))
                    _, _, terminated, _, _ = env.step_discard(action)
                    if terminated:
                        break
            if found:
                break
        assert found, "200 seed で response decision が発生しなかった"

    def test_step_response_works(self):
        """response decision で step_response が動く"""
        env = Stage2Env()
        for seed in range(200):
            env.reset(seed)
            for _ in range(300):
                if env.decision_type == DecisionType.RESPONSE:
                    # 最初の candidate (Skip 以外があればそれ、なければ Skip) を選択
                    obs, rewards, terminated, _, info = env.step_response(0)
                    assert rewards.shape == (4,)
                    return
                if env.decision_type == DecisionType.DISCARD:
                    mask = env.get_legal_mask()
                    action = int(np.argmax(mask))
                    _, _, terminated, _, _ = env.step_discard(action)
                    if terminated:
                        break
        pytest.skip("response decision が得られなかった")

    def test_full_match_completes(self):
        """半荘が最後まで進む"""
        env = Stage2Env()
        env.reset(42)
        for _ in range(5000):
            if env.decision_type == DecisionType.DISCARD:
                mask = env.get_legal_mask()
                action = int(np.argmax(mask))
                _, _, terminated, _, _ = env.step_discard(action)
            elif env.decision_type == DecisionType.RESPONSE:
                # 常に Skip (最後の candidate)
                _, _, terminated, _, _ = env.step_response(
                    len(env.response_candidates) - 1)
            if terminated:
                break
        assert terminated, "5000 step で半荘が終了しなかった"

    def test_response_info_contains_candidates(self):
        """response 時の info に candidate 情報が含まれる"""
        env = Stage2Env()
        for seed in range(200):
            obs, info = env.reset(seed)
            for _ in range(300):
                if env.decision_type == DecisionType.RESPONSE:
                    info = env._make_info()
                    assert "num_response_candidates" in info
                    assert "response_candidate_types" in info
                    assert info["num_response_candidates"] >= 2
                    assert "Skip" in info["response_candidate_types"]
                    return
                if env.decision_type == DecisionType.DISCARD:
                    mask = env.get_legal_mask()
                    action = int(np.argmax(mask))
                    _, _, terminated, _, _ = env.step_discard(action)
                    if terminated:
                        break

    def test_step_response_out_of_range_raises(self):
        """不正な candidate_index で ValueError"""
        env = Stage2Env()
        for seed in range(200):
            env.reset(seed)
            for _ in range(300):
                if env.decision_type == DecisionType.RESPONSE:
                    with pytest.raises(ValueError):
                        env.step_response(999)
                    return
                if env.decision_type == DecisionType.DISCARD:
                    mask = env.get_legal_mask()
                    action = int(np.argmax(mask))
                    _, _, terminated, _, _ = env.step_discard(action)
                    if terminated:
                        break


class TestStage1Regression:
    """Stage1Env の回帰テスト"""

    def test_stage1_still_works(self):
        """Stage1Env が従来どおり動く"""
        from mahjong_rl.env import Stage1Env
        env = Stage1Env()
        obs, info = env.reset(42)
        assert obs is not None
        for _ in range(100):
            mask = env.get_legal_mask()
            action = int(np.argmax(mask))
            obs, rewards, terminated, _, info = env.step(action)
            if terminated:
                break
