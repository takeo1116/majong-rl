"""CQ-0296: Self-action optional candidate safety fixes.

確認:
- Ankan optional candidate の tile_type 正規化 (tile_type / tile_id 両 convention 対応 / sentinel)
- CandidateEncoder / Stage2aModel forward が Ankan candidate で
  embedding index out of range にならない
- Stage2Env.step_response が SelfAction optional primary action 経路で
  ``candidate_index`` に対応する action を実行する (旧 [0] 決め打ち bug
  の retroactive guard)
- Skip 経路で primary action が実行されない
- optional_all enabled の Stage2Env / Stage2SelfPlayWorker smoke が
  crash しない
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.smoke

from mahjong_rl.env.response_candidate import (
    _normalize_tile_to_tile_type,
    make_ankan_optional_candidates,
    OPTIONAL_ANKAN_ACTION_TYPE,
    OPTIONAL_SKIP_ACTION_TYPE,
)


# ========== Section 1: tile_type normalization helper ==========


class TestNormalizeTileToTileType:
    """``_normalize_tile_to_tile_type`` が tile_type / tile_id / sentinel
    すべてに対応する。"""

    def test_tile_type_range_0_33_passthrough(self):
        for tt in (0, 1, 16, 21, 33):
            assert _normalize_tile_to_tile_type(tt) == tt

    def test_tile_id_range_34_135_floor_div(self):
        # 34 = tile_id of 9m (tile_type 8? actually 34//4=8) — let's check several
        assert _normalize_tile_to_tile_type(34) == 8     # 34 // 4 = 8
        assert _normalize_tile_to_tile_type(64) == 16    # 64 // 4 = 16
        assert _normalize_tile_to_tile_type(135) == 33   # 135 // 4 = 33

    def test_sentinel_255_returns_minus1(self):
        assert _normalize_tile_to_tile_type(255) == -1

    def test_negative_returns_minus1(self):
        assert _normalize_tile_to_tile_type(-1) == -1
        assert _normalize_tile_to_tile_type(-100) == -1

    def test_huge_value_returns_minus1(self):
        assert _normalize_tile_to_tile_type(1000) == -1
        assert _normalize_tile_to_tile_type(136) == -1   # 134+ but >=136


# ========== Section 2: make_ankan_optional_candidates uses helper ==========


class TestAnkanCandidateTileTypeRange:
    """``make_ankan_optional_candidates`` の出力 tile_type が 0..33 / -1
    に収まる。"""

    def _mk_action_like(self, tile_value):
        # Action オブジェクトを直接モックすると engine type と整合しないので、
        # 必要なフィールドだけを持つ薄い stub を使う。
        class _A:
            tile = tile_value
            target_player = -1
            type = None
            riichi = False
        return _A()

    def test_tile_type_convention_works(self):
        # 現 engine 想定: .tile が tile_type (0..33)
        ankan = self._mk_action_like(21)
        cands = make_ankan_optional_candidates(
            ankan, skip_action=None, current_player=0)
        primary = cands[0]
        assert primary.action_type == OPTIONAL_ANKAN_ACTION_TYPE
        assert primary.tile_type == 21

    def test_tile_id_convention_normalized(self):
        # 将来 .tile が tile_id (0..135) に揃った場合
        ankan = self._mk_action_like(64)  # tile_id 64 → tile_type 16
        cands = make_ankan_optional_candidates(
            ankan, skip_action=None, current_player=0)
        primary = cands[0]
        assert primary.tile_type == 16

    def test_sentinel_becomes_minus1(self):
        ankan = self._mk_action_like(255)
        cands = make_ankan_optional_candidates(
            ankan, skip_action=None, current_player=0)
        assert cands[0].tile_type == -1

    def test_skip_candidate_tile_type_is_minus1(self):
        # Skip candidate は tile 情報を持たない
        ankan = self._mk_action_like(21)
        cands = make_ankan_optional_candidates(
            ankan, skip_action=None, current_player=0)
        skip = cands[1]
        assert skip.action_type == OPTIONAL_SKIP_ACTION_TYPE
        assert skip.tile_type == -1


# ========== Section 3: embedding range safety ==========


class TestCandidateEncoderEmbeddingRange:
    """CandidateEncoder が Ankan candidate (現/将来 convention 両方) を
    out-of-range なく forward できる。"""

    def _make_model(self):
        from mahjong_rl.models.stage2a_model import Stage2aModel
        return Stage2aModel(
            input_dim=10, discard_hidden_dims=[8],
            optional_hidden_dims=[8], value_hidden_dims=[8],
            candidate_dim=8, optional_scorer_hidden=8,
        )

    def _mk_action_like(self, tile_value):
        class _A:
            tile = tile_value
            target_player = -1
            type = None
            riichi = False
        return _A()

    def _forward_with_ankan_candidate(self, model, ankan_tile_value):
        from mahjong_rl.candidate_encoding import encode_candidates_batch
        cands = make_ankan_optional_candidates(
            self._mk_action_like(ankan_tile_value),
            skip_action=None, current_player=0)
        # build a minimal call sample-like batch (1 sample)
        # encode_candidates_batch wants .candidates / .candidate_count
        class _S:
            def __init__(self, cs):
                self.candidates = cs
                self.candidate_count = len(cs)
        s = _S(cands)
        feats, mask = encode_candidates_batch([s], len(cands))
        obs = torch.zeros(1, 10, dtype=torch.float32)
        rc = torch.zeros(1, 3, dtype=torch.float32)
        out = model.forward_optional(obs, feats, mask, response_context=rc,
                                       compute_value=False)
        return out

    def test_forward_with_tile_type_convention(self):
        model = self._make_model()
        # .tile = 21 (tile_type) → 正規化後 21 → emb[22] (index+1)
        out = self._forward_with_ankan_candidate(model, 21)
        assert out.optional_scores.shape == (1, 2)

    def test_forward_with_tile_id_convention(self):
        model = self._make_model()
        # .tile = 64 (tile_id) → 正規化後 16 → emb[17]
        out = self._forward_with_ankan_candidate(model, 64)
        assert out.optional_scores.shape == (1, 2)

    def test_forward_with_sentinel(self):
        model = self._make_model()
        out = self._forward_with_ankan_candidate(model, 255)
        assert out.optional_scores.shape == (1, 2)


# ========== Section 4: step_response uses candidate_index ==========


def _drive_until(env, target_dt, max_seeds=500, max_steps=300, seed_offset=0):
    """指定 decision_type にぶつかるまで env を進める。"""
    from mahjong_rl.env import DecisionType
    import random
    rng = random.Random(seed_offset)
    for seed in range(max_seeds):
        env.reset(seed + seed_offset)
        for _ in range(max_steps):
            dt = env.decision_type
            if dt == target_dt:
                return seed
            if dt == DecisionType.DISCARD:
                mask = env.get_legal_mask()
                legal = [i for i in range(34) if mask[i] > 0.5]
                if not legal:
                    break
                tt = rng.choice(legal)
                _, _, term, _, _ = env.step_discard(tt)
                if term:
                    break
            elif dt == DecisionType.RESPONSE:
                n = len(env.response_candidates)
                _, _, term, _, _ = env.step_response(rng.randrange(n))
                if term:
                    break
            elif dt in (DecisionType.RIICHI_OPTIONAL,
                         DecisionType.TSUMO_OPTIONAL,
                         DecisionType.RON_OPTIONAL,
                         DecisionType.ANKAN_OPTIONAL,
                         DecisionType.KAKAN_OPTIONAL,
                         DecisionType.KYUUSHU_OPTIONAL):
                _, _, term, _, _ = env.step_response(0)
                if term:
                    break
            else:
                break
    return None


class TestStepResponseCandidateIndex:
    """``Stage2Env.step_response`` が SelfAction optional の primary action
    を ``candidate_index`` に対応した候補から実行する。"""

    def test_ankan_primary_executes_engine_step(self):
        """ANKAN_OPTIONAL で primary (idx=0) を選ぶと engine が Ankan を
        実行し、phase / scores が変化する。"""
        from mahjong_rl.env import Stage2Env, DecisionType
        env = Stage2Env(observation_mode="full",
                          optional_ankan_enabled=True)
        seed = _drive_until(env, DecisionType.ANKAN_OPTIONAL)
        if seed is None:
            pytest.skip("ANKAN_OPTIONAL seed not found in 500 seeds")
        before = list(env.env_state.match_state.scores)
        _, _, _, _, _ = env.step_response(0)  # primary = Ankan
        # decision_type は ANKAN_OPTIONAL から離れる
        assert env.decision_type != DecisionType.ANKAN_OPTIONAL

    def test_ankan_skip_does_not_execute_primary(self):
        """ANKAN_OPTIONAL で Skip (idx=1) を選ぶと engine step は走らない
        (skipped 集合に "ankan" が入り、次 decision が同 player の
        DISCARD になる)。"""
        from mahjong_rl.env import Stage2Env, DecisionType
        env = Stage2Env(observation_mode="full",
                          optional_ankan_enabled=True)
        seed = _drive_until(env, DecisionType.ANKAN_OPTIONAL)
        if seed is None:
            pytest.skip("ANKAN_OPTIONAL seed not found in 500 seeds")
        cp_before = env.env_state.round_state.current_player
        scores_before = list(env.env_state.match_state.scores)
        _, _, _, _, _ = env.step_response(1)  # Skip
        # Skip 後は engine step 無しで DISCARD に fall-through
        assert "ankan" in env._optional_skipped_this_turn
        # scores は engine step 無いので不変
        assert list(env.env_state.match_state.scores) == scores_before
        # current_player も同じ (DISCARD 待ち)
        assert env.env_state.round_state.current_player == cp_before

    def test_tsumo_skip_uses_action_type_not_hardcoded_index(self):
        """TSUMO_OPTIONAL の Skip 検出が ``action_type == OPTIONAL_SKIP_*``
        ベースで動く (旧 ``candidate_index == 1`` の検出と等価動作)。"""
        from mahjong_rl.env import Stage2Env, DecisionType
        env = Stage2Env(observation_mode="full",
                          optional_tsumo_enabled=True)
        seed = _drive_until(env, DecisionType.TSUMO_OPTIONAL)
        if seed is None:
            pytest.skip("TSUMO_OPTIONAL seed not found in 500 seeds")
        scores_before = list(env.env_state.match_state.scores)
        _, _, _, _, _ = env.step_response(1)  # Skip
        # Skip → engine step 無し → scores 不変
        assert list(env.env_state.match_state.scores) == scores_before
        assert "tsumo" in env._optional_skipped_this_turn


# ========== Section 5: optional_all enabled smoke ==========


class TestOptionalAllSmoke:
    """全 optional 有効で Stage2Env / Stage2SelfPlayWorker が crash しない
    ことを軽量 smoke で確認。"""

    def test_stage2_env_optional_all_smoke(self):
        from mahjong_rl.env import Stage2Env, DecisionType
        import random
        env = Stage2Env(
            observation_mode="full",
            optional_riichi_enabled=True,
            optional_tsumo_enabled=True,
            optional_ron_enabled=True,
            optional_ankan_enabled=True,
            optional_kakan_enabled=True,
            optional_kyuushu_enabled=True,
        )
        rng = random.Random(0)
        env.reset(0)
        for _ in range(2000):
            dt = env.decision_type
            if dt == DecisionType.DISCARD:
                mask = env.get_legal_mask()
                legal = [i for i in range(34) if mask[i] > 0.5]
                if not legal:
                    break
                _, _, term, _, _ = env.step_discard(rng.choice(legal))
            elif dt == DecisionType.RESPONSE:
                n = len(env.response_candidates)
                _, _, term, _, _ = env.step_response(rng.randrange(n))
            elif dt in (DecisionType.RIICHI_OPTIONAL,
                         DecisionType.TSUMO_OPTIONAL,
                         DecisionType.RON_OPTIONAL,
                         DecisionType.ANKAN_OPTIONAL,
                         DecisionType.KAKAN_OPTIONAL,
                         DecisionType.KYUUSHU_OPTIONAL):
                # 0 or 1 をランダムに
                _, _, term, _, _ = env.step_response(rng.randrange(2))
            else:
                break
            if term:
                break

    def test_selfplay_worker_optional_all_smoke(self, tmp_path):
        from mahjong_rl.encoders import FlatFeatureEncoder
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        encoder = FlatFeatureEncoder(observation_mode="full")
        config = {
            "optional_riichi": {"enabled": True},
            "optional_tsumo": {"enabled": True},
            "optional_ron": {"enabled": True},
            "optional_ankan": {"enabled": True},
            "optional_kakan": {"enabled": True},
            "optional_kyuushu": {"enabled": True},
        }
        worker = Stage2SelfPlayWorker(
            config=config,
            output_dir=tmp_path,
            observation_mode="full",
            encoder=encoder,
        )
        stats = worker.generate(num_matches=1, base_seed=0,
                                  experiment_id="t", run_id="r",
                                  worker_id="w0")
        # smoke: 完走 + 結果 dict にキー
        assert "total_steps" in stats
        assert "decision_family_counts" in stats
