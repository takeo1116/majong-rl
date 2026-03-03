"""RuleBasedBaseline: シャンテン数最小化ベースの打牌選択"""
from __future__ import annotations

import numpy as np

from mahjong_rl._mahjong_core import NUM_TILE_TYPES
from .shanten import compute_shanten


class RuleBasedBaseline:
    """ルールベースベースライン

    各合法打牌候補でシャンテン数を計算し、最小を選ぶ。
    同シャンテン数の場合は受け入れ枚数で比較する。
    """

    def select_discard(
        self,
        hand_tile_ids: list[int],
        legal_mask: np.ndarray,
    ) -> int:
        """打牌を選択する

        Args:
            hand_tile_ids: 手牌の TileId リスト (0-135)
            legal_mask: 34種の合法手マスク (1=合法, 0=非合法)

        Returns:
            選択された牌種 (TileType, 0-33)
        """
        # 手牌を34種カウントに変換
        counts = [0] * NUM_TILE_TYPES
        for tid in hand_tile_ids:
            counts[tid // 4] += 1

        best_type = -1
        best_shanten = 999
        best_acceptance = -1

        for t in range(NUM_TILE_TYPES):
            if legal_mask[t] < 0.5:
                continue
            if counts[t] <= 0:
                continue

            # t を切った後のシャンテン数
            counts[t] -= 1
            sh = compute_shanten(counts)

            if sh < best_shanten or (sh == best_shanten and self._count_acceptance(counts, sh) > best_acceptance):
                best_shanten = sh
                best_acceptance = self._count_acceptance(counts, sh)
                best_type = t

            counts[t] += 1

        return best_type

    @staticmethod
    def _count_acceptance(counts: list[int], shanten: int) -> int:
        """受け入れ枚数を計算する

        現在のシャンテン数が下がる牌種の残り枚数合計。
        """
        total = 0
        for t in range(NUM_TILE_TYPES):
            if counts[t] >= 4:
                continue
            counts[t] += 1
            new_sh = compute_shanten(counts)
            counts[t] -= 1
            if new_sh < shanten:
                total += 4 - counts[t]  # 残り枚数の概算
        return total
