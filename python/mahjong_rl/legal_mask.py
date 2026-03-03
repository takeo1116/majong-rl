"""Stage 1 用 legal mask 生成"""
import numpy as np
from mahjong_rl._mahjong_core import ActionType, NUM_TILE_TYPES


def make_discard_mask(hand_tile_ids: list[int]) -> np.ndarray:
    """手牌 TileId リストから 34 種打牌マスクを生成する

    Args:
        hand_tile_ids: 手牌の TileId 列（0-135）

    Returns:
        (34,) の float32 配列、手牌に含まれる牌種が 1.0
    """
    mask = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    for tile_id in hand_tile_ids:
        mask[tile_id // 4] = 1.0
    return mask


def make_discard_mask_from_legal_actions(legal_actions) -> np.ndarray:
    """合法アクションリストから 34 種打牌マスクを生成する

    立直打牌がある場合は立直打牌の牌種のみを含む。
    立直打牌がない場合は通常打牌の牌種を含む。

    Args:
        legal_actions: エンジンの get_legal_actions() の結果

    Returns:
        (34,) の float32 配列
    """
    mask = np.zeros(NUM_TILE_TYPES, dtype=np.float32)

    # 立直打牌があるか確認
    riichi_discards = [a for a in legal_actions
                       if a.type == ActionType.Discard and a.riichi]
    if riichi_discards:
        for a in riichi_discards:
            mask[a.tile // 4] = 1.0
    else:
        for a in legal_actions:
            if a.type == ActionType.Discard:
                mask[a.tile // 4] = 1.0

    return mask
