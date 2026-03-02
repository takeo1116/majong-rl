#pragma once

#include "core/tile.h"
#include "rules/agari.h"
#include "rules/yaku.h"

namespace mahjong {
namespace fu_calculator {

// 符計算結果
struct FuResult {
    int base_fu;       // 副底 (20 or 30)
    int mentsu_fu;     // 面子符
    int jantai_fu;     // 雀頭符
    int wait_fu;       // 待ち符
    int tsumo_fu;      // ツモ符
    int total_fu;      // 合計符（10符単位切り上げ後）
};

// 符計算を行う
// decomp: 面子構成
// wait_type: 待ちの種類
// agari_tile: 和了牌の TileType
// is_tsumo: ツモか
// is_menzen: 門前か
// is_pinfu: 平和が成立しているか
// bakaze: 場風の TileType
// jikaze: 自風の TileType
FuResult calculate_fu(
    const AgariDecomposition& decomp,
    WaitType wait_type,
    TileType agari_tile,
    bool is_tsumo,
    bool is_menzen,
    bool is_pinfu,
    TileType bakaze,
    TileType jikaze);

}  // namespace fu_calculator
}  // namespace mahjong
