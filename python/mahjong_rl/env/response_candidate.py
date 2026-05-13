"""CQ-0221: Response phase の legal candidate 表現

response phase の legal action 群を、学習用 candidate として扱うための
型安全な Python 表現。engine の Action をそのまま候補として使う。

CQ-0291 (batch 1): Riichi optional decision の candidate factory も提供する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mahjong_rl._mahjong_core import Action, ActionType


# Stage2a で学習対象とする ActionType
STAGE2A_LEARNABLE_TYPES = frozenset({
    ActionType.Chi,
    ActionType.Pon,
    ActionType.Daiminkan,
    ActionType.Skip,
})

# CQ-0291: Riichi optional 用 synthetic action_type 値
# (engine の ActionType と被らない値を使う; candidate_encoding.ACTION_TYPE_MAP と一致)
OPTIONAL_NORIICHI_ACTION_TYPE = 100
OPTIONAL_RIICHI_ACTION_TYPE = 101

# CQ-0291 batch 2: 和了 optional 用 action_type 値
# (engine の ActionType の int をそのまま流用)
OPTIONAL_TSUMO_ACTION_TYPE = int(ActionType.TsumoWin)  # = 1
OPTIONAL_RON_ACTION_TYPE = int(ActionType.Ron)         # = 2
# 和了 optional の Skip は engine の Skip (=8)
OPTIONAL_SKIP_ACTION_TYPE = int(ActionType.Skip)       # = 8

# CQ-0291 batch 3: Kan/Kyuushu optional 用 action_type 値
OPTIONAL_KAKAN_ACTION_TYPE = int(ActionType.Kakan)     # = 6
OPTIONAL_ANKAN_ACTION_TYPE = int(ActionType.Ankan)     # = 7
OPTIONAL_KYUUSHU_ACTION_TYPE = int(ActionType.Kyuushu) # = 9


def _normalize_tile_to_tile_type(raw_tile: int) -> int:
    """CQ-0296: ``action.tile`` を 0..33 の tile_type に正規化する。

    現 engine では:
    - Ankan action: ``.tile`` が既に tile_type (0..33) 値
    - Kakan / Discard action: ``.tile`` が tile_id (0..135) 値
    の convention が混在している。将来 convention が tile_id に揃った
    場合でも安全に動くよう、defensive に正規化する。

    Args:
        raw_tile: engine action の ``tile`` フィールド (int)

    Returns:
        - 0..33: tile_type
        - 0..135 (但し >=34): tile_id とみなして ``// 4``
        - それ以外 / sentinel (255 など): ``-1``
    """
    raw = int(raw_tile)
    if 0 <= raw < 34:
        return raw
    if 34 <= raw < 136:
        return raw // 4
    return -1


@dataclass(frozen=True)
class ResponseCandidate:
    """1つの response candidate

    Attributes:
        action: engine 側の Action (step に直接渡せる)
        action_type: ActionType (Chi/Pon/Daiminkan/Skip)
        tile_type: 関連牌種 (TileType, Skip は -1)
        target_rel_seat: 鳴き先の相対席 (current から見た距離, Skip は -1)
        consumed_tile_ids: 消費牌の TileId リスト (Skip は空)
    """
    action: Action
    action_type: ActionType
    tile_type: int
    target_rel_seat: int
    consumed_tile_ids: tuple[int, ...]


def extract_response_candidates(
    legal_actions: Sequence[Action],
    current_player: int,
    num_players: int = 4,
    learnable_only: bool = True,
) -> list[ResponseCandidate]:
    """engine の legal_actions から response candidate リストを生成する

    Args:
        legal_actions: engine.get_legal_actions() の返り値
        current_player: 現在の応答者
        num_players: プレイヤー数 (通常 4)
        learnable_only: True なら Stage2a 学習対象のみ返す

    Returns:
        ResponseCandidate のリスト。Skip が含まれる場合はリスト末尾に配置。
    """
    candidates: list[ResponseCandidate] = []
    skip_candidate: ResponseCandidate | None = None

    for action in legal_actions:
        at = action.type

        if learnable_only and at not in STAGE2A_LEARNABLE_TYPES:
            continue

        if at == ActionType.Skip:
            skip_candidate = ResponseCandidate(
                action=action,
                action_type=at,
                tile_type=-1,
                target_rel_seat=-1,
                consumed_tile_ids=(),
            )
            continue

        # Chi / Pon / Daiminkan
        tile_type = action.tile // 4 if action.tile < 255 else -1
        target = action.target_player
        if target >= num_players:
            # Chi は target_player 未設定 (255)。上家 = (actor - 1) % 4
            if at == ActionType.Chi:
                target = (action.actor - 1) % num_players
            else:
                target = current_player  # fallback
        rel_seat = (target - current_player) % num_players
        consumed = tuple(t for t in action.consumed_tiles if t != 255)

        candidates.append(ResponseCandidate(
            action=action,
            action_type=at,
            tile_type=tile_type,
            target_rel_seat=rel_seat,
            consumed_tile_ids=consumed,
        ))

    # Skip は末尾に配置
    if skip_candidate is not None:
        candidates.append(skip_candidate)

    return candidates


def make_riichi_optional_candidates(
    non_riichi_action: Action,
    riichi_action: Action,
    current_player: int,
) -> list[ResponseCandidate]:
    """CQ-0291 (batch 1): Riichi optional decision の candidates を作る

    Order: [NoRiichi (idx=0), Riichi (idx=1)]
    両方の Action は engine の `Discard` (ActionType.Discard) で、riichi flag が
    異なるだけ。`action_type` には CQ-0291 専用の synthetic 値
    (`OPTIONAL_NORIICHI_ACTION_TYPE` / `OPTIONAL_RIICHI_ACTION_TYPE`) を入れて、
    candidate encoder で区別する。
    """
    tile_type = non_riichi_action.tile // 4
    return [
        ResponseCandidate(
            action=non_riichi_action,
            action_type=OPTIONAL_NORIICHI_ACTION_TYPE,
            tile_type=tile_type,
            target_rel_seat=-1,
            consumed_tile_ids=(),
        ),
        ResponseCandidate(
            action=riichi_action,
            action_type=OPTIONAL_RIICHI_ACTION_TYPE,
            tile_type=tile_type,
            target_rel_seat=-1,
            consumed_tile_ids=(),
        ),
    ]


def make_tsumo_optional_candidates(
    tsumo_action: Action,
    skip_action: Action | None,
    current_player: int,
) -> list[ResponseCandidate]:
    """CQ-0291 (batch 2): TsumoWin optional decision の candidates を作る

    Order: [TsumoWin (idx=0), Skip (idx=1)]

    SelfActionPhase で TsumoWin が合法なときに使う。``skip_action`` が None の
    場合は (engine が SelfActionPhase で明示的な Skip action を提供しないため)
    ``Skip = 1`` が選ばれた際の挙動は wrapper 側で制御する (= 後段の DISCARD
    decision に fall-through)。Action フィールドはそのまま wrapper に
    引き渡される。
    """
    return [
        ResponseCandidate(
            action=tsumo_action,
            action_type=OPTIONAL_TSUMO_ACTION_TYPE,
            tile_type=-1,
            target_rel_seat=-1,
            consumed_tile_ids=(),
        ),
        ResponseCandidate(
            action=skip_action if skip_action is not None else tsumo_action,
            action_type=OPTIONAL_SKIP_ACTION_TYPE,
            tile_type=-1,
            target_rel_seat=-1,
            consumed_tile_ids=(),
        ),
    ]


def make_ron_optional_candidates(
    ron_action: Action,
    skip_action: Action,
    current_player: int,
    num_players: int = 4,
) -> list[ResponseCandidate]:
    """CQ-0291 (batch 2): Ron optional decision の candidates を作る

    Order: [Ron (idx=0), Skip (idx=1)]

    ResponsePhase で Ron が合法なときに使う。Ron 候補のみ + Skip という
    binary optional に絞る (Pon/Chi/Daiminkan が同時に合法な稀なケースは
    Ron を選ぶか Skip を選ぶかに簡略化。バッチ 3 以降で再検討)。
    """
    tile_type = ron_action.tile // 4 if ron_action.tile < 255 else -1
    target = ron_action.target_player
    if 0 <= target < num_players:
        rel_seat = (target - current_player) % num_players
    else:
        rel_seat = -1
    return [
        ResponseCandidate(
            action=ron_action,
            action_type=OPTIONAL_RON_ACTION_TYPE,
            tile_type=tile_type,
            target_rel_seat=rel_seat,
            consumed_tile_ids=(),
        ),
        ResponseCandidate(
            action=skip_action,
            action_type=OPTIONAL_SKIP_ACTION_TYPE,
            tile_type=-1,
            target_rel_seat=-1,
            consumed_tile_ids=(),
        ),
    ]


def _make_self_action_optional_candidates(
    primary_action: Action,
    primary_action_type_code: int,
    skip_action: Action | None,
    primary_tile_type: int = -1,
) -> list[ResponseCandidate]:
    """CQ-0291 batch 3: SelfActionPhase の optional decision 用共通 helper

    Order: [primary (idx=0), Skip (idx=1)]
    SelfActionPhase で engine が明示的な Skip action を提供しない場合は
    ``skip_action=None`` を許容する (Skip 選択は wrapper 側で discard
    fall-through する)。
    """
    return [
        ResponseCandidate(
            action=primary_action,
            action_type=primary_action_type_code,
            tile_type=primary_tile_type,
            target_rel_seat=-1,
            consumed_tile_ids=(),
        ),
        ResponseCandidate(
            action=(skip_action if skip_action is not None else primary_action),
            action_type=OPTIONAL_SKIP_ACTION_TYPE,
            tile_type=-1,
            target_rel_seat=-1,
            consumed_tile_ids=(),
        ),
    ]


def make_ankan_optional_candidates(
    ankan_action: Action,
    skip_action: Action | None,
    current_player: int,
) -> list[ResponseCandidate]:
    """CQ-0291 (batch 3): Ankan optional decision の candidates を作る

    Order: [Ankan (idx=0), Skip (idx=1)]
    複数 Ankan tile_type が同時に合法な稀ケースは、wrapper 側で先頭の
    Ankan を渡す。

    CQ-0296: ``ankan_action.tile`` は現 engine では tile_type (0..33) 値
    だが、将来 tile_id convention に揃う可能性に備えて
    ``_normalize_tile_to_tile_type`` で defensive に正規化する。
    """
    tile_type = _normalize_tile_to_tile_type(ankan_action.tile)
    return _make_self_action_optional_candidates(
        primary_action=ankan_action,
        primary_action_type_code=OPTIONAL_ANKAN_ACTION_TYPE,
        skip_action=skip_action,
        primary_tile_type=tile_type,
    )


def make_kakan_optional_candidates(
    kakan_action: Action,
    skip_action: Action | None,
    current_player: int,
) -> list[ResponseCandidate]:
    """CQ-0291 (batch 3): Kakan optional decision の candidates を作る

    Order: [Kakan (idx=0), Skip (idx=1)]
    """
    tile_type = (kakan_action.tile // 4) if kakan_action.tile < 255 else -1
    return _make_self_action_optional_candidates(
        primary_action=kakan_action,
        primary_action_type_code=OPTIONAL_KAKAN_ACTION_TYPE,
        skip_action=skip_action,
        primary_tile_type=tile_type,
    )


def make_kyuushu_optional_candidates(
    kyuushu_action: Action,
    skip_action: Action | None,
    current_player: int,
) -> list[ResponseCandidate]:
    """CQ-0291 (batch 3): Kyuushu optional decision の candidates を作る

    Order: [Kyuushu (idx=0), Skip (idx=1)]
    """
    return _make_self_action_optional_candidates(
        primary_action=kyuushu_action,
        primary_action_type_code=OPTIONAL_KYUUSHU_ACTION_TYPE,
        skip_action=skip_action,
        primary_tile_type=-1,
    )
