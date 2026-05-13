"""CQ-0297: optional / response branch encoder mask consistency."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def _drive_until_riichi_optional(env, max_seeds=500, max_steps=300):
    from mahjong_rl.env import DecisionType

    for seed in range(max_seeds):
        env.reset(seed)
        for _ in range(max_steps):
            if env.decision_type == DecisionType.DISCARD:
                mask, snap = env.get_legal_discard_snapshot()
                if mask.sum() == 0:
                    break
                riichi_tile_types = sorted({a.tile // 4 for a in snap if a.riichi})
                tt = riichi_tile_types[0] if riichi_tile_types else int(np.argmax(mask))
                _, _, term, _, _ = env.step_discard_with_snapshot(tt, snap)
                if env.decision_type == DecisionType.RIICHI_OPTIONAL:
                    return seed
                if term:
                    break
            elif env.decision_type == DecisionType.RESPONSE:
                n = len(env.response_candidates)
                _, _, term, _, _ = env.step_response(n - 1)
                if term:
                    break
            elif env.decision_type == DecisionType.RIICHI_OPTIONAL:
                return seed
            else:
                break
    return None


def _encoder_with_riichi_mask():
    from mahjong_rl.encoders import FlatFeatureEncoder

    return FlatFeatureEncoder(
        observation_mode="full",
        shanten_hint=True,
        discard_ukeire_hint=True,
        current_shanten_input=True,
        shape_hint=True,
        turn_context=True,
        tile_presence_flags=True,
        riichi_discard_mask=True,
    )


class TestOptionalBranchEncoderMasks:
    def test_worker_riichi_optional_encodes_current_riichi_mask(self, tmp_path):
        from mahjong_rl.env import Stage2Env, DecisionType
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker

        env = Stage2Env(observation_mode="full", optional_riichi_enabled=True)
        seed = _drive_until_riichi_optional(env)
        if seed is None:
            pytest.skip("RIICHI_OPTIONAL に到達しなかった")
        assert env.decision_type == DecisionType.RIICHI_OPTIONAL

        enc = _encoder_with_riichi_mask()
        worker = Stage2SelfPlayWorker(
            config={"training": {"optional_riichi": {"enabled": True}}},
            output_dir=tmp_path,
            observation_mode="full",
            encoder=enc,
            model=None,
        )
        features = worker._encode_env_obs(env)
        rng = enc.metadata().feature_ranges["riichi_discard_mask"]
        expected = env.get_riichi_discard_mask()
        assert expected.sum() > 0
        assert np.allclose(features[rng[0]:rng[1]], expected)

        # Old buggy path encoded optional branches with no riichi_discard_mask.
        obs = env._make_observation()
        old_features = worker._encode_obs(obs)
        assert old_features[rng[0]:rng[1]].sum() == 0.0

    def test_evaluator_policy_call_encodes_current_riichi_mask(self):
        from mahjong_rl.env import Stage2Env, DecisionType
        from mahjong_rl.models.stage2a_model import Stage2aModel
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator

        env = Stage2Env(observation_mode="full", optional_riichi_enabled=True)
        seed = _drive_until_riichi_optional(env)
        if seed is None:
            pytest.skip("RIICHI_OPTIONAL に到達しなかった")
        assert env.decision_type == DecisionType.RIICHI_OPTIONAL

        enc = _encoder_with_riichi_mask()
        model = Stage2aModel(
            input_dim=enc.metadata().output_shape[0],
            discard_hidden_dims=[16],
            optional_hidden_dims=[16],
            value_hidden_dims=[16],
            candidate_dim=8,
            optional_scorer_hidden=8,
            semantic_aux_config={"enabled": True},
        )
        evaluator = Stage2aEvaluator(
            model=model,
            encoder=enc,
            observation_mode="full",
            optional_riichi_enabled=True,
        )
        features = evaluator._encode_env_obs(env)
        rng = enc.metadata().feature_ranges["riichi_discard_mask"]
        expected = env.get_riichi_discard_mask()
        assert expected.sum() > 0
        assert np.allclose(features[rng[0]:rng[1]], expected)

        # Smoke: the actual optional policy path can consume these features.
        idx = evaluator._policy_call(env, env.response_candidates, env.current_player)
        assert 0 <= idx < len(env.response_candidates)


# CQ-0297 follow-up: ResponsePhase legacy behavior preservation


def _drive_until_response(env, max_seeds=500, max_steps=300):
    """Drive env until DecisionType.RESPONSE (ResponsePhase)."""
    import random
    from mahjong_rl.env import DecisionType
    rng = random.Random(0)
    for seed in range(max_seeds):
        env.reset(seed)
        for _ in range(max_steps):
            if env.decision_type == DecisionType.RESPONSE:
                return seed
            if env.decision_type == DecisionType.DISCARD:
                mask = env.get_legal_mask()
                legal = [i for i in range(34) if mask[i] > 0.5]
                if not legal:
                    break
                _, _, term, _, _ = env.step_discard(rng.choice(legal))
                if term:
                    break
            elif env.decision_type in (
                DecisionType.RIICHI_OPTIONAL,
                DecisionType.TSUMO_OPTIONAL,
                DecisionType.RON_OPTIONAL,
                DecisionType.ANKAN_OPTIONAL,
                DecisionType.KAKAN_OPTIONAL,
                DecisionType.KYUUSHU_OPTIONAL,
            ):
                _, _, term, _, _ = env.step_response(0)
                if term:
                    break
            else:
                break
    return None


class TestResponsePhaseLegacySemantics:
    """CQ-0297 follow-up: ResponsePhase (legal_mask all-zero) で encoder
    feature が pre-CQ-0297 と一致することを確認する regression guard。
    """

    def test_worker_response_phase_matches_legacy_encode(self, tmp_path):
        from mahjong_rl.env import Stage2Env, DecisionType
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker

        env = Stage2Env(observation_mode="full")
        seed = _drive_until_response(env)
        if seed is None:
            pytest.skip("RESPONSE phase に到達しなかった")
        assert env.decision_type == DecisionType.RESPONSE
        # ResponsePhase なので legal_mask は all-zero
        legal = env.get_legal_mask()
        assert legal.sum() == 0.0

        enc = _encoder_with_riichi_mask()
        worker = Stage2SelfPlayWorker(
            config={},
            output_dir=tmp_path,
            observation_mode="full",
            encoder=enc,
            model=None,
        )
        # CQ-0297 follow-up: ResponsePhase では legal_mask=None 経由の
        # 旧 encode と _encode_env_obs の出力が一致する。
        obs = env._make_observation()
        legacy_features = enc.encode(obs)  # legal_mask=None
        if legacy_features.ndim > 1:
            legacy_features = legacy_features.flatten()
        new_features = worker._encode_env_obs(env)
        assert np.allclose(legacy_features, new_features), (
            "ResponsePhase で worker._encode_env_obs が legacy encode(obs) と "
            "一致しない。discard_ukeire_hint / riichi_discard_mask の "
            "ResponsePhase semantics が破壊された可能性。")

    def test_evaluator_response_phase_matches_legacy_encode(self):
        from mahjong_rl.env import Stage2Env, DecisionType
        from mahjong_rl.models.stage2a_model import Stage2aModel
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator

        env = Stage2Env(observation_mode="full")
        seed = _drive_until_response(env)
        if seed is None:
            pytest.skip("RESPONSE phase に到達しなかった")
        assert env.decision_type == DecisionType.RESPONSE
        legal = env.get_legal_mask()
        assert legal.sum() == 0.0

        enc = _encoder_with_riichi_mask()
        model = Stage2aModel(
            input_dim=enc.metadata().output_shape[0],
            discard_hidden_dims=[16],
            optional_hidden_dims=[16],
            value_hidden_dims=[16],
            candidate_dim=8,
            optional_scorer_hidden=8,
        )
        evaluator = Stage2aEvaluator(
            model=model,
            encoder=enc,
            observation_mode="full",
        )
        obs = env._make_observation()
        legacy_features = enc.encode(obs)
        if legacy_features.ndim > 1:
            legacy_features = legacy_features.flatten()
        new_features = evaluator._encode_env_obs(env)
        assert np.allclose(legacy_features, new_features), (
            "ResponsePhase で evaluator._encode_env_obs が legacy "
            "encode(obs) と一致しない。")
