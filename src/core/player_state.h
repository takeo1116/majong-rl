#pragma once

#include "core/tile.h"
#include "core/types.h"
#include "core/meld.h"
#include <array>
#include <vector>

namespace mahjong {

// 河に捨てられた牌の情報
struct DiscardInfo {
    TileId tile;
    bool riichi_discard;  // 立直宣言打牌かどうか
    bool called;          // 他家に鳴かれたかどうか
};

// プレイヤー状態
// 値コピー可能な構造体
struct PlayerState {
    // 手牌（実牌IDの動的配列、通常13枚、ツモ時14枚）
    std::vector<TileId> hand;

    // 副露一覧
    std::vector<Meld> melds;

    // 河
    std::vector<DiscardInfo> discards;

    // 持ち点
    int32_t score = 25000;

    // 立直済みフラグ
    bool is_riichi = false;

    // ダブル立直フラグ
    bool is_double_riichi = false;

    // 一発有効フラグ
    bool ippatsu = false;

    // 門前状態
    bool is_menzen = true;

    // 自風（0=東, 1=南, 2=西, 3=北）
    Wind jikaze = Wind::East;

    // フリテン状態
    bool is_furiten = false;

    // 同巡内フリテン
    bool is_temporary_furiten = false;

    // 立直後見逃しフリテン
    bool is_riichi_furiten = false;

    // 嶺上ツモフラグ（嶺上開花の判定用）
    bool rinshan_draw = false;

    // 初期化
    void reset(Wind wind, int32_t initial_score = 25000);

    // 手牌枚数を返す
    int hand_count() const { return static_cast<int>(hand.size()); }

    // 副露数を返す
    int meld_count() const { return static_cast<int>(melds.size()); }
};

}  // namespace mahjong
