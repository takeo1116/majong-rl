#pragma once

#include "core/tile.h"
#include "core/types.h"
#include <array>
#include <string>

namespace mahjong {

// 副露（鳴き）を表現する構造体
struct Meld {
    MeldType type;                 // 副露の種別
    std::array<TileId, 4> tiles;   // 構成牌のID（使用しない枠は255）
    uint8_t tile_count;            // 構成牌の枚数（チー/ポン: 3, 槓: 4）
    PlayerId from_player;          // 鳴いた元のプレイヤー（暗槓の場合は自分自身）
    TileId called_tile;            // 鳴いた牌のID

    // 構築ヘルパー
    static Meld make_chi(TileId called, TileId t1, TileId t2, PlayerId from);
    static Meld make_pon(TileId called, TileId t1, TileId t2, PlayerId from);
    static Meld make_daiminkan(TileId called, TileId t1, TileId t2, TileId t3, PlayerId from);
    static Meld make_ankan(TileId t1, TileId t2, TileId t3, TileId t4, PlayerId self);
    static Meld make_kakan(const Meld& pon, TileId added);

    // 副露に含まれる TileType を返す（チーの場合は最小の type）
    TileType base_type() const;

    // 比較
    bool operator==(const Meld&) const = default;

    // 文字列表現
    std::string to_string() const;
};

}  // namespace mahjong
