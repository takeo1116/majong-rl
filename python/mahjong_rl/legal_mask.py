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


def make_discard_mask_from_legal_actions(
    legal_actions, include_all_discards: bool = False,
) -> np.ndarray:
    """合法アクションリストから 34 種打牌マスクを生成する

    Args:
        legal_actions: エンジンの get_legal_actions() の結果
        include_all_discards: ``False`` (default) は既存挙動。
            立直打牌がある場合は立直打牌の牌種のみを含む。
            立直打牌がない場合は通常打牌の牌種を含む。
            ``True`` (CQ-0292 batch 2) は立直打牌の有無に関わらず、
            全 Discard tile_type を mask に含める (Stage2a optional
            riichi 用)。

    Returns:
        (34,) の float32 配列
    """
    mask = np.zeros(NUM_TILE_TYPES, dtype=np.float32)

    if include_all_discards:
        for a in legal_actions:
            if a.type == ActionType.Discard:
                mask[a.tile // 4] = 1.0
        return mask

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


def make_teacher_discard_mask_from_legal_actions(legal_actions) -> np.ndarray:
    """CQ-0294: rulebase teacher / baseline 用 discard mask。

    旧 auto-riichi 優先 semantics を常に維持する。すなわち legal actions
    に riichi discard が含まれているなら riichi 打牌の tile_type のみを
    返し、そうでなければ通常 discard の tile_type を返す。

    Stage2a optional_riichi 有効 (= policy mask が全打牌を含む) の場合
    でも、teacher / baseline 経路はこの mask を使うことで「立直可能
    なら立直打牌だけから選ぶ」誘導を維持する。

    実装上は ``make_discard_mask_from_legal_actions(legal_actions,
    include_all_discards=False)`` と同じ。意図を明示する別名として
    提供する。
    """
    return make_discard_mask_from_legal_actions(
        legal_actions, include_all_discards=False)


def make_riichi_discard_mask_from_legal_actions(legal_actions) -> np.ndarray:
    """CQ-0294: ``riichi_discard_mask`` feature 用の 34 次元 flag。

    その tile_type を切ると riichi discard action が存在するなら 1.0、
    存在しないなら 0.0。立直不可局面 / 既に立直済み / 副露あり等で
    riichi discard が legal actions に含まれない場合は全 0 になる。
    """
    mask = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    for a in legal_actions:
        if a.type == ActionType.Discard and a.riichi:
            mask[a.tile // 4] = 1.0
    return mask
