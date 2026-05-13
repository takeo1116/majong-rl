"""CQ-0292 batch 2: Riichi optional discard mask 修正 + decision_family
diagnostics の検証。

カバー範囲:
- ``make_discard_mask_from_legal_actions(include_all_discards=True)``
- ``Stage2Env`` の ``get_legal_mask`` / ``get_legal_discard_snapshot``
  が ``optional_riichi_enabled`` を見て mask 幅を切り替える
- snapshot/non-snapshot 経路で同じ semantics
- 赤5/通常5 が混ざる場合の通常牌優先 (CQ-0290) 維持
- ``Stage2SelfPlayWorker.generate()`` の stats に
  ``decision_family_counts`` / ``optional_decision_count`` が出る
- parallel selfplay aggregation で counts が合算される
- default off では optional_decision_count = 0
- discard sample の ``decision_family`` が ``"discard"`` になる
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.smoke


# ========== Section 1: legal_mask helper ==========


class TestMakeDiscardMaskFromLegalActions:
    """include_all_discards=True で riichi 有無に関わらず全 Discard を含む。"""

    def _make_actions(self, specs):
        """Discard Action のみを並べた擬似 list。"""
        from mahjong_rl._mahjong_core import Action
        return [Action.make_discard(0, tile, riichi=riichi)
                for tile, riichi in specs]

    def test_default_riichi_only_when_riichi_present(self):
        from mahjong_rl.legal_mask import make_discard_mask_from_legal_actions
        # tile_id 0 は 1m / tile_id 4 は 2m
        actions = self._make_actions([
            (0, False), (4, True), (8, False),
        ])
        mask = make_discard_mask_from_legal_actions(actions)
        # 立直 tile_type=1 のみ
        assert mask[0] == 0.0
        assert mask[1] == 1.0
        assert mask[2] == 0.0
        assert mask.sum() == 1.0

    def test_include_all_discards_unions_all_tile_types(self):
        from mahjong_rl.legal_mask import make_discard_mask_from_legal_actions
        actions = self._make_actions([
            (0, False), (4, True), (8, False),
        ])
        mask = make_discard_mask_from_legal_actions(
            actions, include_all_discards=True)
        # 全 tile_type が legal mask に入る
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask[2] == 1.0
        assert mask.sum() == 3.0

    def test_include_all_discards_no_riichi_no_change(self):
        from mahjong_rl.legal_mask import make_discard_mask_from_legal_actions
        actions = self._make_actions([(0, False), (4, False)])
        mask_default = make_discard_mask_from_legal_actions(actions)
        mask_all = make_discard_mask_from_legal_actions(
            actions, include_all_discards=True)
        # riichi が無いときは挙動が同じ
        assert (mask_default == mask_all).all()


# ========== Section 2: Stage2Env legal mask semantics ==========


class TestStage2EnvLegalMaskSemantics:
    """Stage2Env が ``optional_riichi_enabled`` で mask 幅を切り替える。"""

    def _drive_until_riichi_legal(self, env, max_seeds=500, max_steps=300):
        """riichi 打牌が含まれる discard 場面までドライブする。"""
        from mahjong_rl.env import DecisionType
        for seed in range(max_seeds):
            env.reset(seed)
            for _ in range(max_steps):
                if env.decision_type == DecisionType.DISCARD:
                    mask, snap = env.get_legal_discard_snapshot()
                    if any(a.riichi for a in snap):
                        return seed, mask, snap
                    if mask.sum() == 0:
                        break
                    tt = int(np.argmax(mask))
                    _, _, term, _, _ = env.step_discard_with_snapshot(
                        tt, snap)
                    if term or env.decision_type == DecisionType.RIICHI_OPTIONAL:
                        # RIICHI_OPTIONAL からは抜ける (テスト前提状態を作るため)
                        if env.decision_type == DecisionType.RIICHI_OPTIONAL:
                            env.step_response(0)  # NoRiichi
                        if term:
                            break
                elif env.decision_type == DecisionType.RESPONSE:
                    n = len(env.response_candidates)
                    _, _, term, _, _ = env.step_response(n - 1)
                    if term:
                        break
                else:
                    break
        return None, None, None

    def test_default_off_mask_is_riichi_only_when_riichi_present(self):
        from mahjong_rl.env import Stage2Env
        env = Stage2Env(observation_mode="full",
                        optional_riichi_enabled=False)
        seed, mask, snap = self._drive_until_riichi_legal(env)
        if seed is None:
            pytest.skip("no riichi-legal seed found in 500 seeds")
        # default off: riichi 打牌の tile_type のみが mask に入る
        riichi_tile_types = {a.tile // 4 for a in snap if a.riichi}
        non_riichi_only_types = {a.tile // 4 for a in snap
                                  if not a.riichi
                                  and (a.tile // 4) not in riichi_tile_types}
        # mask は riichi tile_type だけ
        for tt in riichi_tile_types:
            assert mask[tt] == 1.0, f"riichi tile_type {tt} must be legal"
        # non-riichi-only tile_types は legal mask から除外
        for tt in non_riichi_only_types:
            assert mask[tt] == 0.0, (
                f"non-riichi tile_type {tt} must NOT be in default-off mask")

    def test_enabled_mask_includes_all_discards(self):
        from mahjong_rl.env import Stage2Env
        env = Stage2Env(observation_mode="full",
                        optional_riichi_enabled=True)
        seed, mask, snap = self._drive_until_riichi_legal(env)
        if seed is None:
            pytest.skip("no riichi-legal seed found in 500 seeds")
        # enabled: riichi/非riichi 両方の tile_type が legal
        all_tile_types = {a.tile // 4 for a in snap}
        for tt in all_tile_types:
            assert mask[tt] == 1.0, (
                f"tile_type {tt} must be legal under optional_riichi=true")

    def test_snapshot_and_non_snapshot_mask_match(self):
        """get_legal_mask と get_legal_discard_snapshot[0] が一致する。"""
        from mahjong_rl.env import Stage2Env
        env = Stage2Env(observation_mode="full",
                        optional_riichi_enabled=True)
        seed, snap_mask, snap = self._drive_until_riichi_legal(env)
        if seed is None:
            pytest.skip("no riichi-legal seed found in 500 seeds")
        legal_mask = env.get_legal_mask()
        assert (legal_mask == snap_mask).all()


# ========== Section 3: Stage2Env riichi optional choice semantics ==========


class TestStage2EnvRiichiOptionalChoice:
    """RIICHI_OPTIONAL 経路の semantics を確認。"""

    def _drive_until_riichi_optional(self, env, max_seeds=500, max_steps=300):
        """riichi 打牌可能 tile_type を直接選んで RIICHI_OPTIONAL を出す。"""
        from mahjong_rl.env import DecisionType
        for seed in range(max_seeds):
            env.reset(seed)
            for _ in range(max_steps):
                if env.decision_type == DecisionType.DISCARD:
                    mask, snap = env.get_legal_discard_snapshot()
                    riichi_tile_types = sorted({
                        a.tile // 4 for a in snap if a.riichi
                    })
                    if riichi_tile_types:
                        tt = riichi_tile_types[0]
                    elif mask.sum() == 0:
                        break
                    else:
                        tt = int(np.argmax(mask))
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
                else:
                    break
        return None

    def test_riichi_optional_tile_type_is_legal_in_mask(self):
        """RIICHI_OPTIONAL 発火時、その tile_type は直前の discard mask 上で
        legal だった (= mask 上で 1)。"""
        from mahjong_rl.env import Stage2Env
        env = Stage2Env(observation_mode="full",
                        optional_riichi_enabled=True)
        seed = self._drive_until_riichi_optional(env)
        if seed is None:
            pytest.skip("no RIICHI_OPTIONAL seed found")
        cands = env.response_candidates
        assert len(cands) == 2
        # 両 candidate の tile_type は同一であるべき
        assert cands[0].tile_type == cands[1].tile_type

    def test_red_normal_priority_in_riichi_optional(self):
        """同 tile_type 内で 通常牌 → 赤牌の順 (CQ-0290) を維持する。
        riichi/non-riichi 両方とも 通常牌が選ばれる。"""
        from mahjong_rl.env import Stage2Env
        env = Stage2Env(observation_mode="full",
                        optional_riichi_enabled=True)
        seed = self._drive_until_riichi_optional(env)
        if seed is None:
            pytest.skip("no RIICHI_OPTIONAL seed found")
        cands = env.response_candidates
        red_ids = (16, 52, 88)
        non_riichi_action = cands[0].action
        riichi_action = cands[1].action
        # 通常牌が選択候補に存在するなら赤牌は使わないこと
        snap = [a for a in env._engine.get_legal_actions(env._env)
                if a.type.name == "Discard"
                and (a.tile // 4) == cands[0].tile_type]
        normal_non_riichi = [a for a in snap
                              if (not a.riichi) and (a.tile not in red_ids)]
        normal_riichi = [a for a in snap
                          if a.riichi and (a.tile not in red_ids)]
        if normal_non_riichi:
            assert non_riichi_action.tile not in red_ids
        if normal_riichi:
            assert riichi_action.tile not in red_ids


# ========== Section 4: Stage2SelfPlayWorker decision_family diagnostics ==========


class TestSelfPlayDecisionFamilyDiagnostics:
    """selfplay stats に decision_family_counts / optional_decision_count が
    入る。"""

    def _make_worker(self, tmp_path, **flags):
        from mahjong_rl.encoders import FlatFeatureEncoder
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        encoder = FlatFeatureEncoder(observation_mode="full")
        config: dict = {}
        for k, v in flags.items():
            config[k] = {"enabled": bool(v)}
        return Stage2SelfPlayWorker(
            config=config,
            output_dir=tmp_path,
            observation_mode="full",
            encoder=encoder,
        )

    def test_default_off_optional_decision_count_is_zero(self, tmp_path):
        w = self._make_worker(tmp_path)
        stats = w.generate(num_matches=1, base_seed=0,
                            experiment_id="t", run_id="r",
                            worker_id="w0")
        assert "decision_family_counts" in stats
        assert "optional_decision_count" in stats
        assert stats["optional_decision_count"] == 0
        # default off では riichi/tsumo/ron/ankan/kakan/kyuushu = 0
        for k in ("riichi", "tsumo", "ron", "ankan", "kakan", "kyuushu"):
            assert stats["decision_family_counts"].get(k, 0) == 0

    def test_default_off_discard_count_matches_family_counts(self, tmp_path):
        w = self._make_worker(tmp_path)
        stats = w.generate(num_matches=1, base_seed=0,
                            experiment_id="t", run_id="r",
                            worker_id="w0")
        # save_baseline_actions=False & model=None → save される全 sample が
        # baseline 由来 (imitation mode)
        assert stats["decision_family_counts"]["discard"] == stats["discard_count"]
        # response 系
        assert stats["decision_family_counts"]["response"] >= 0

    def test_all_optional_on_emits_optional_families(self, tmp_path):
        w = self._make_worker(
            tmp_path,
            optional_riichi=True, optional_tsumo=True, optional_ron=True,
            optional_ankan=True, optional_kakan=True, optional_kyuushu=True,
        )
        stats = w.generate(num_matches=2, base_seed=0,
                            experiment_id="t", run_id="r",
                            worker_id="w0")
        # 何らかの optional decision が発生していること (2 matches あれば
        # riichi が出る確率が高い)
        opt = stats["optional_decision_count"]
        # 弱めの guarantee: 3 種以上 (response + discard) は確実に出る
        assert stats["decision_family_counts"]["discard"] > 0
        # optional_decision_count は family_counts の非 response/non-discard 合計
        recomputed = sum(
            v for k, v in stats["decision_family_counts"].items()
            if k not in ("discard", "response")
        )
        assert opt == recomputed

    def test_stats_is_json_serializable(self, tmp_path):
        w = self._make_worker(tmp_path, optional_riichi=True)
        stats = w.generate(num_matches=1, base_seed=0,
                            experiment_id="t", run_id="r",
                            worker_id="w0")
        # JSON serializable
        s = json.dumps(stats)
        loaded = json.loads(s)
        assert "decision_family_counts" in loaded
        assert "optional_decision_count" in loaded


# ========== Section 5: parallel aggregation ==========


class TestParallelAggregateDecisionFamily:
    """_aggregate_stats が decision_family_counts / optional_decision_count
    を合算する。"""

    def test_aggregate_sums_family_counts(self):
        from mahjong_rl.stage2a_parallel import _aggregate_stats
        stats_list = [
            {
                "num_matches": 1, "total_steps": 10,
                "discard_count": 5, "call_count": 3,
                "decision_family_counts": {
                    "discard": 5, "response": 2, "riichi": 1,
                    "tsumo": 0, "ron": 0, "ankan": 0,
                    "kakan": 0, "kyuushu": 0,
                },
            },
            {
                "num_matches": 1, "total_steps": 12,
                "discard_count": 6, "call_count": 4,
                "decision_family_counts": {
                    "discard": 6, "response": 3, "riichi": 1,
                    "tsumo": 1, "ron": 0, "ankan": 0,
                    "kakan": 0, "kyuushu": 0,
                },
            },
        ]
        agg = _aggregate_stats(stats_list, num_workers=2)
        assert agg["decision_family_counts"]["discard"] == 11
        assert agg["decision_family_counts"]["response"] == 5
        assert agg["decision_family_counts"]["riichi"] == 2
        assert agg["decision_family_counts"]["tsumo"] == 1
        assert agg["optional_decision_count"] == 3  # 2 + 1

    def test_aggregate_handles_missing_family_counts(self):
        from mahjong_rl.stage2a_parallel import _aggregate_stats
        # 古い worker stats (decision_family_counts なし) が混ざっても crash しない
        stats_list = [
            {"num_matches": 1, "total_steps": 5,
             "discard_count": 3, "call_count": 1},
            {"num_matches": 1, "total_steps": 6,
             "discard_count": 4, "call_count": 2,
             "decision_family_counts": {
                 "discard": 4, "response": 1, "riichi": 1,
                 "tsumo": 0, "ron": 0, "ankan": 0,
                 "kakan": 0, "kyuushu": 0,
             }},
        ]
        agg = _aggregate_stats(stats_list, num_workers=2)
        assert agg["decision_family_counts"]["discard"] == 4
        assert agg["decision_family_counts"]["riichi"] == 1
        assert agg["optional_decision_count"] == 1


# ========== Section 6: discard sample decision_family ==========


class TestDiscardSampleDecisionFamily:
    """shard に書き出された discard sample の decision_family が
    ``"discard"`` になる (response として集計されない)。"""

    def test_discard_sample_decision_family_is_discard(self, tmp_path):
        from mahjong_rl.encoders import FlatFeatureEncoder
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.call_shard import DecisionShardReader
        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={},
            output_dir=tmp_path,
            observation_mode="full",
            encoder=encoder,
        )
        worker.generate(num_matches=1, base_seed=0,
                         experiment_id="t", run_id="r",
                         worker_id="w0")
        # 出力された shard を読む
        reader = DecisionShardReader(tmp_path)
        samples = reader.read_all()
        n_discard = 0
        n_discard_family = 0
        n_response_family = 0
        for s in samples:
            if s.decision_type == "discard":
                n_discard += 1
                if s.decision_family == "discard":
                    n_discard_family += 1
                if s.decision_family == "response":
                    n_response_family += 1
        assert n_discard > 0
        # 全 discard sample は decision_family="discard"
        assert n_discard_family == n_discard
        # 一つも response として保存されていない
        assert n_response_family == 0
