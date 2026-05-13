"""CQ-0290: 34種打牌の concrete tile 解決で赤牌より通常牌を優先する

確認:
- engine `get_self_actions` が、同一 tile_type 内で通常牌 → 赤牌の順に
  Discard actions を列挙する
- `Stage2Env._resolve_discard_from_snapshot` が同一 tile_type 内で
  通常牌 → 赤牌を優先する
- `Stage2Env._resolve_discard` が同様
- `Stage1Env._resolve_discard` が同様
- 通常牌が無い (赤牌のみ) 場合は赤牌を選べる
- riichi 候補の中でも通常牌 → 赤牌の順
- riichi 優先 (riichi+通常 > riichi+赤 > non-riichi+通常 > non-riichi+赤)
"""
from __future__ import annotations

import pytest
import numpy as np

pytestmark = pytest.mark.smoke

from mahjong_rl import (
    GameEngine, EnvironmentState, Action, ActionType, Phase,
    NUM_TILE_TYPES,
)
from mahjong_rl._mahjong_core import Tile
from mahjong_rl.env.stage1_env import Stage1Env, _is_red_tile_id
from mahjong_rl.env.stage2_env import (
    Stage2Env,
    _is_red_tile_id as _is_red_tile_id_stage2,
)


# 赤牌の TileId
RED_5M = 16
RED_5P = 52
RED_5S = 88
TYPE_5M = 4
TYPE_5P = 13
TYPE_5S = 22


# ========== 1. helper unit tests ==========


class TestIsRedTileId:
    """`_is_red_tile_id` helper の仕様"""

    @pytest.mark.parametrize("tile_id", [16, 52, 88])
    def test_red_ids(self, tile_id):
        assert _is_red_tile_id(tile_id) is True
        assert _is_red_tile_id_stage2(tile_id) is True
        assert Tile.is_red_id(tile_id) is True

    @pytest.mark.parametrize("tile_id", [0, 17, 18, 19, 53, 54, 89, 90, 135])
    def test_non_red_ids(self, tile_id):
        assert _is_red_tile_id(tile_id) is False
        assert _is_red_tile_id_stage2(tile_id) is False
        assert Tile.is_red_id(tile_id) is False


# ========== 2. _resolve_discard_from_snapshot (Stage2 静的) ==========


def _mk_discard(tile_id: int, riichi: bool = False) -> "Action":
    """tile_id を捨てる Discard Action を作る"""
    # Action.make_discard(player_id, tile, riichi=False)
    return Action.make_discard(0, tile_id, riichi)


class TestStage2SnapshotResolveDiscard:
    """`Stage2Env._resolve_discard_from_snapshot` の通常牌優先"""

    def test_red_and_normal_5m_picks_normal(self):
        """赤5m と通常5m が両方候補にあるとき、通常5m を選ぶ"""
        actions = [
            _mk_discard(RED_5M),    # 赤5m (id=16)
            _mk_discard(17),         # 通常5m (id=17)
            _mk_discard(18),         # 通常5m (id=18)
        ]
        a = Stage2Env._resolve_discard_from_snapshot(TYPE_5M, actions)
        assert _is_red_tile_id(a.tile) is False, (
            f"赤牌 {a.tile} が選ばれた; 通常牌を選ぶべき")
        assert a.tile // 4 == TYPE_5M

    def test_red_only_picks_red(self):
        """通常牌が無く赤牌だけならば赤牌を選ぶ"""
        actions = [_mk_discard(RED_5M)]
        a = Stage2Env._resolve_discard_from_snapshot(TYPE_5M, actions)
        assert a.tile == RED_5M

    def test_normal_first_in_input_still_picks_normal(self):
        """入力順が通常 → 赤でも通常を選ぶ (ordering 不変)"""
        actions = [
            _mk_discard(17),         # 通常5m
            _mk_discard(RED_5M),    # 赤5m
        ]
        a = Stage2Env._resolve_discard_from_snapshot(TYPE_5M, actions)
        assert _is_red_tile_id(a.tile) is False

    def test_red_first_in_input_still_picks_normal(self):
        """入力順が赤 → 通常でも通常を選ぶ"""
        actions = [
            _mk_discard(RED_5M),    # 赤5m が先頭
            _mk_discard(17),
        ]
        a = Stage2Env._resolve_discard_from_snapshot(TYPE_5M, actions)
        assert a.tile == 17

    def test_red5p_and_normal_5p(self):
        """5p でも同じ"""
        actions = [
            _mk_discard(RED_5P),    # 赤5p (52)
            _mk_discard(53),         # 通常5p
        ]
        a = Stage2Env._resolve_discard_from_snapshot(TYPE_5P, actions)
        assert a.tile == 53

    def test_red5s_and_normal_5s(self):
        """5s でも同じ"""
        actions = [
            _mk_discard(RED_5S),    # 赤5s (88)
            _mk_discard(89),         # 通常5s
        ]
        a = Stage2Env._resolve_discard_from_snapshot(TYPE_5S, actions)
        assert a.tile == 89

    def test_riichi_candidate_prefers_normal(self):
        """riichi 候補でも同一 tile_type 内で通常牌を優先"""
        actions = [
            _mk_discard(RED_5M, riichi=True),   # riichi + 赤5m
            _mk_discard(17, riichi=True),        # riichi + 通常5m
        ]
        a = Stage2Env._resolve_discard_from_snapshot(TYPE_5M, actions)
        assert a.riichi is True
        assert a.tile == 17  # 通常5m

    def test_riichi_priority_over_non_riichi(self):
        """riichi+通常 > non-riichi+通常 > non-riichi+赤 (CQ-0290 後も維持)"""
        actions = [
            _mk_discard(17, riichi=False),       # non-riichi + 通常5m
            _mk_discard(RED_5M, riichi=False),  # non-riichi + 赤5m
            _mk_discard(17, riichi=True),        # riichi + 通常5m
        ]
        a = Stage2Env._resolve_discard_from_snapshot(TYPE_5M, actions)
        assert a.riichi is True
        assert a.tile == 17

    def test_riichi_red_beats_non_riichi_normal(self):
        """riichi+赤 > non-riichi+通常 (riichi 優先が tile 色より優先)"""
        actions = [
            _mk_discard(17, riichi=False),       # non-riichi + 通常5m
            _mk_discard(RED_5M, riichi=True),   # riichi + 赤5m (riichi+normal 無し)
        ]
        a = Stage2Env._resolve_discard_from_snapshot(TYPE_5M, actions)
        assert a.riichi is True
        assert a.tile == RED_5M

    def test_unknown_tile_type_raises(self):
        actions = [_mk_discard(0)]
        with pytest.raises(ValueError, match="snapshot 不整合"):
            Stage2Env._resolve_discard_from_snapshot(TYPE_5M, actions)


# ========== 3. engine self action 列挙順 ==========


class TestEngineSelfActionOrder:
    """`engine.get_legal_actions` で同 tile_type 内 normal → red 順"""

    def test_engine_emits_normal_before_red(self):
        """engine self action 列挙で同 tile_type 内 normal が red より前に出る

        実 match を回し、player.hand に赤牌+同 tile_type 通常牌が同時にある
        場面で、`engine.get_legal_actions` が通常牌を先に返すことを確認する。
        """
        # Stage1Env (= 同じ engine を内包) で deterministic に進める
        env = Stage1Env(observation_mode="full")
        found = False
        for seed in range(2000):
            env.reset(seed)
            for _ in range(50):
                rs = env.env_state.round_state
                cp = rs.current_player
                hand = list(rs.players[cp].hand)
                for red_id, type_id in [
                    (RED_5M, TYPE_5M),
                    (RED_5P, TYPE_5P),
                    (RED_5S, TYPE_5S),
                ]:
                    if red_id not in hand:
                        continue
                    if not any(tid != red_id and (tid // 4) == type_id
                               for tid in hand):
                        continue
                    actions = env._engine.get_legal_actions(env._env)
                    red_idx = None
                    normal_idx = None
                    for i, a in enumerate(actions):
                        if a.type != ActionType.Discard:
                            continue
                        if a.riichi:
                            # 立直系は別ストリーム; 通常 Discard だけで判定
                            continue
                        if a.tile == red_id:
                            if red_idx is None:
                                red_idx = i
                        elif (a.tile // 4) == type_id:
                            if normal_idx is None:
                                normal_idx = i
                    if red_idx is not None and normal_idx is not None:
                        found = True
                        assert normal_idx < red_idx, (
                            f"seed={seed}: tile_type={type_id} で "
                            f"赤(idx={red_idx}) が 通常(idx={normal_idx}) "
                            f"より先に列挙された")
                        return
                # 進める
                mask = env.get_legal_mask()
                if mask.sum() == 0:
                    break
                tile_type = int(np.argmax(mask))
                try:
                    _, _, term, _, _ = env.step(tile_type)
                except Exception:
                    break
                if term:
                    break
        if not found:
            pytest.skip(
                "赤牌+通常牌が同一 tile_type で同時に手にある seed が "
                "サンプル範囲で見つからなかった (engine の deal 性質)")


# ========== 4. Stage1Env._resolve_discard / Stage2Env._resolve_discard ==========


class _FakeAction:
    """Stage1Env / Stage2Env の `_resolve_discard` で使うため、
    `type`, `tile`, `riichi` 属性を持つ minimal stub"""
    def __init__(self, tile_id: int, riichi: bool = False,
                 action_type=None):
        self.tile = tile_id
        self.riichi = riichi
        self.type = action_type if action_type is not None else ActionType.Discard


class _FakeEngine:
    """`get_legal_actions(env)` が固定リストを返す stub"""
    def __init__(self, actions):
        self._actions = actions

    def get_legal_actions(self, env):
        return self._actions


class TestStage1EnvResolveDiscard:
    """`Stage1Env._resolve_discard` の通常牌優先 (engine を stub 化)"""

    def _make_env_with_actions(self, actions):
        env = Stage1Env.__new__(Stage1Env)
        env._engine = _FakeEngine(actions)
        env._env = None
        return env

    def test_red_and_normal_5m_picks_normal(self):
        env = self._make_env_with_actions([
            _FakeAction(RED_5M),
            _FakeAction(17),
        ])
        a = env._resolve_discard(TYPE_5M)
        assert a.tile == 17

    def test_red_only_picks_red(self):
        env = self._make_env_with_actions([_FakeAction(RED_5M)])
        a = env._resolve_discard(TYPE_5M)
        assert a.tile == RED_5M

    def test_riichi_normal_beats_riichi_red(self):
        env = self._make_env_with_actions([
            _FakeAction(RED_5M, riichi=True),
            _FakeAction(17, riichi=True),
        ])
        a = env._resolve_discard(TYPE_5M)
        assert a.riichi is True
        assert a.tile == 17

    def test_riichi_priority(self):
        """riichi+通常 > non-riichi+通常 > non-riichi+赤"""
        env = self._make_env_with_actions([
            _FakeAction(RED_5M, riichi=False),
            _FakeAction(17, riichi=False),
            _FakeAction(17, riichi=True),
        ])
        a = env._resolve_discard(TYPE_5M)
        assert a.riichi is True
        assert a.tile == 17


class TestStage2EnvResolveDiscard:
    """`Stage2Env._resolve_discard` の通常牌優先 (engine を stub 化)"""

    def _make_env_with_actions(self, actions):
        env = Stage2Env.__new__(Stage2Env)
        env._engine = _FakeEngine(actions)
        env._env = None
        return env

    def test_red_and_normal_5m_picks_normal(self):
        env = self._make_env_with_actions([
            _FakeAction(RED_5M),
            _FakeAction(17),
        ])
        a = env._resolve_discard(TYPE_5M)
        assert a.tile == 17

    def test_red_only_picks_red(self):
        env = self._make_env_with_actions([_FakeAction(RED_5M)])
        a = env._resolve_discard(TYPE_5M)
        assert a.tile == RED_5M

    def test_riichi_normal_beats_riichi_red(self):
        env = self._make_env_with_actions([
            _FakeAction(RED_5M, riichi=True),
            _FakeAction(17, riichi=True),
        ])
        a = env._resolve_discard(TYPE_5M)
        assert a.riichi is True
        assert a.tile == 17


# ========== 5. integration: 実 selfplay で同 tile_type 内 ordering ==========


class TestIntegrationLiveEngine:
    """実 engine を回し、red+normal が手にある match で通常を選ぶ"""

    def test_resolve_discard_from_snapshot_in_real_match(self):
        """Stage2Env で実 match を回し、player が赤+通常 5 を同時に持つ
        場面で snapshot resolution が通常牌を選ぶ"""
        env = Stage2Env(observation_mode="full")
        for seed in range(500):
            env.reset(seed)
            # discard phase で player.hand を確認
            for _ in range(50):
                if env._decision_type.value != "discard":
                    break
                cp = env.env_state.round_state.current_player
                hand = list(env.env_state.round_state.players[cp].hand)
                # 赤+通常が同 tile_type で同時にあれば検証
                for red_id, type_id in [
                    (RED_5M, TYPE_5M),
                    (RED_5P, TYPE_5P),
                    (RED_5S, TYPE_5S),
                ]:
                    if red_id in hand and any(
                            (tid // 4) == type_id and tid != red_id
                            for tid in hand):
                        mask, snap = env.get_legal_discard_snapshot()
                        if mask[type_id] == 0.0:
                            continue
                        # snapshot resolve で通常牌が選ばれる
                        a = Stage2Env._resolve_discard_from_snapshot(
                            type_id, snap)
                        assert not _is_red_tile_id(a.tile), (
                            f"seed={seed}: {type_id} の resolve が "
                            f"赤牌 {a.tile} を選んだ; 通常牌を選ぶべき")
                        # 1 回検証できれば十分
                        return
                # 進める (1m を試す等は危険なので、1 つの合法 tile で進める)
                mask = env.get_legal_mask()
                tile_type = int(np.argmax(mask))
                _, _, term, _, _ = env.step_discard(tile_type)
                if term:
                    break
        pytest.skip("赤+通常が同 tile_type で同時に手にある場面が "
                    "サンプル範囲で見つからなかった")
