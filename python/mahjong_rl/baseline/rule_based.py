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
        """打牌を選択する (fast path)

        best-set 収集（2パス目）を省略し、最良候補のみ返す。
        評価規則は _find_best_score() で一本化されている (CQ-0128, CQ-0129)。

        Args:
            hand_tile_ids: 手牌の TileId リスト (0-135)
            legal_mask: 34種の合法手マスク (1=合法, 0=非合法)

        Returns:
            選択された牌種 (TileType, 0-33)
        """
        counts, best_shanten, best_acceptance = self._find_best_score(
            hand_tile_ids, legal_mask)
        # best に一致する最初の候補を返す（1パスのみ）
        for t in range(NUM_TILE_TYPES):
            if legal_mask[t] < 0.5 or counts[t] <= 0:
                continue
            counts[t] -= 1
            sh = compute_shanten(counts)
            acc = self._count_acceptance(counts, sh)
            counts[t] += 1
            if sh == best_shanten and acc == best_acceptance:
                return t
        return -1

    def select_discard_with_best_set(
        self,
        hand_tile_ids: list[int],
        legal_mask: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """打牌を選択し、同率最良候補の mask も返す (CQ-0125)

        評価規則は _find_best_score() で一本化されている (CQ-0128, CQ-0129)。
        best-set 収集のため2パス目を実行する。

        Args:
            hand_tile_ids: 手牌の TileId リスト (0-135)
            legal_mask: 34種の合法手マスク (1=合法, 0=非合法)

        Returns:
            (best_tile_type, best_mask):
              best_tile_type: 選択された牌種 (TileType, 0-33)
              best_mask: (34,) float32, 同率最良候補=1.0
        """
        counts, best_shanten, best_acceptance = self._find_best_score(
            hand_tile_ids, legal_mask)

        # 2パス目: 同率候補を収集
        best_mask = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        best_type = -1
        for t in range(NUM_TILE_TYPES):
            if legal_mask[t] < 0.5 or counts[t] <= 0:
                continue
            counts[t] -= 1
            sh = compute_shanten(counts)
            acc = self._count_acceptance(counts, sh)
            if sh == best_shanten and acc == best_acceptance:
                best_mask[t] = 1.0
                if best_type == -1:
                    best_type = t
            counts[t] += 1

        return best_type, best_mask

    def _find_best_score(
        self,
        hand_tile_ids: list[int],
        legal_mask: np.ndarray,
    ) -> tuple[list[int], int, int]:
        """全候補を1パス評価し、最良 (shanten, acceptance) を返す (CQ-0129)

        評価規則: シャンテン数最小 → 受け入れ枚数最大

        Returns:
            (counts, best_shanten, best_acceptance):
              counts: 手牌の牌種カウント配列（呼び出し元で再利用可能）
        """
        counts = [0] * NUM_TILE_TYPES
        for tid in hand_tile_ids:
            counts[tid // 4] += 1

        best_shanten = 999
        best_acceptance = -1
        for t in range(NUM_TILE_TYPES):
            if legal_mask[t] < 0.5 or counts[t] <= 0:
                continue
            counts[t] -= 1
            sh = compute_shanten(counts)
            acc = self._count_acceptance(counts, sh)
            if sh < best_shanten or (sh == best_shanten and acc > best_acceptance):
                best_shanten = sh
                best_acceptance = acc
            counts[t] += 1

        return counts, best_shanten, best_acceptance

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
