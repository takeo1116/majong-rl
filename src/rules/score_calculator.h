#pragma once

#include "core/tile.h"
#include "rules/agari.h"
#include "rules/yaku.h"
#include "rules/fu_calculator.h"
#include <array>
#include <vector>

namespace mahjong {
namespace score_calculator {

// 支払い情報
struct PaymentInfo {
    int from_non_dealer;   // 非親からの支払い（ツモ時）
    int from_dealer;       // 親からの支払い（ツモ時）
    int from_ron;          // ロン時の支払い
};

// 点数計算結果
struct ScoreResult {
    int total_han;       // 翻合計
    int fu;              // 符
    int base_point;      // 基本点
    PaymentInfo payment; // 支払い情報（積み棒込み）
    std::vector<YakuResult> yakus;  // 成立した役
    int dora_count;
    int akadora_count;
    int uradora_count;
    AgariDecomposition best_decomposition;  // 高点法で選ばれた構成
    WaitType best_wait_type;                // 選ばれた待ち形
    bool valid;          // 役ありで有効な和了か
};

// 和了の点数を計算する（高点法）
// decompositions: 全面子構成
// ctx: 和了コンテキスト
// is_dealer: 和了者が親か
// honba: 積み棒数
ScoreResult calculate_win_score(
    const std::vector<AgariDecomposition>& decompositions,
    const WinContext& ctx,
    bool is_dealer,
    int honba);

// 基本点を計算する（翻と符から）
int calculate_base_point(int han, int fu);

// 100点単位切り上げ
int ceil100(int value);

}  // namespace score_calculator
}  // namespace mahjong
