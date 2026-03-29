"""CQ-0221: Response phase の legal candidate 表現

response phase の legal action 群を、学習用 candidate として扱うための
型安全な Python 表現。engine の Action をそのまま候補として使う。
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
