#pragma once

#include <array>

namespace mahjong {

constexpr int kNumTileTypesShanten = 34;

/// シャンテン数を計算する (通常形・七対子・国士の最小値)
int compute_shanten(const std::array<int, kNumTileTypesShanten>& counts);

/// 打牌候補の一括分析結果
struct DiscardAnalysis {
    std::array<int, 34> shanten_after;   ///< 打牌後シャンテン (非候補=-1)
    std::array<int, 34> acceptance;      ///< 受け入れ枚数 (非候補=0)
    std::array<float, 34> ukeire_norm;   ///< 正規化受け入れ (max基準, [0,1])
    std::array<float, 34> shanten_sign;  ///< delta sign (-1.0/0.0)
};

/// 全打牌候補のシャンテン数・受け入れ枚数・shanten_sign を一括計算
/// @param counts 34種の手牌カウント
/// @param legal_mask 合法手マスク (>=1 で合法)。手牌にない牌種は自動スキップ。
DiscardAnalysis analyze_discards(
    const std::array<int, 34>& counts,
    const std::array<int, 34>& legal_mask);

/// 最善打牌の選択結果
struct BestDiscardResult {
    int best_shanten;       ///< 最善シャンテン数
    int best_acceptance;    ///< 最善受け入れ枚数
    int best_tile;          ///< 最善打牌 (tie-break: 最小 index)。候補なしなら -1
    std::array<int, 34> best_mask;  ///< 同率最良候補=1, それ以外=0
};

/// 最善打牌を選択する (シャンテン最小 → 受け入れ最大)
BestDiscardResult find_best_discard(
    const std::array<int, 34>& counts,
    const std::array<int, 34>& legal_mask);

/// 手牌形状ヒント (順子/塔子/嵌張の binary multihot)
struct ShapeHint {
    std::array<float, 21> chi;           ///< 順子 (3スート × 7中心牌)
    std::array<float, 24> outside_wait;  ///< 塔子 (3スート × 8隣接ペア)
    std::array<float, 21> inside_wait;   ///< 嵌張 (3スート × 7中心牌)
};

/// 手牌形状ヒントを計算する
ShapeHint compute_shape_hint(const std::array<int, 34>& counts);

}  // namespace mahjong
