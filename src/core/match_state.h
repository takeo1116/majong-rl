#pragma once

#include "core/tile.h"
#include "core/types.h"
#include <array>

namespace mahjong {

// 半荘状態
// 値コピー可能な構造体
struct MatchState {
    // 現在の局番号（0=東1局, 1=東2局, ..., 7=南4局, 8=延長局）
    uint8_t round_number = 0;

    // 各プレイヤーの持ち点
    std::array<int32_t, kNumPlayers> scores = {25000, 25000, 25000, 25000};

    // 起家（最初の親）
    PlayerId first_dealer = 0;

    // 現在の親
    PlayerId current_dealer = 0;

    // 本場
    uint8_t honba = 0;

    // 供託本数
    uint8_t kyotaku = 0;

    // 延長局突入済みフラグ
    bool is_extra_round = false;

    // 半荘終了フラグ
    bool is_match_over = false;

    // 最終順位（0-based, scores が同点の場合は起家に近い順で上位）
    std::array<uint8_t, kNumPlayers> final_ranking = {0, 1, 2, 3};

    // 場風（0=東, 1=南）
    Wind bakaze() const;

    // 現在がオーラスかどうか
    bool is_oorasu() const;

    // 比較
    bool operator==(const MatchState&) const = default;

    // 初期化
    void reset(PlayerId dealer = 0);

    // 順位を計算して final_ranking に反映する
    void compute_ranking();
};

}  // namespace mahjong
