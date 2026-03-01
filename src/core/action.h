#pragma once

#include "core/tile.h"
#include "core/types.h"
#include <array>
#include <string>

namespace mahjong {

// 行動を表現する完全構造体
struct Action {
    ActionType type;              // 行動種別
    PlayerId actor;               // 行動者のプレイヤーID
    TileId tile;                  // 対象牌（不要な場合は255）
    PlayerId target_player;       // 対象プレイヤー（不要な場合は255）
    MeldType meld_type;           // 副露種別（鳴き時のみ有効）
    bool riichi;                  // 立直宣言フラグ（Discard時のみ有効）
    std::array<TileId, 2> consumed_tiles;  // 鳴き時に手牌から出す牌（最大2枚、不要な枠は255）

    // 構築ヘルパー
    static Action make_discard(PlayerId actor, TileId tile, bool riichi = false);
    static Action make_tsumo_win(PlayerId actor);
    static Action make_ron(PlayerId actor, PlayerId target);
    static Action make_chi(PlayerId actor, TileId called, TileId t1, TileId t2);
    static Action make_pon(PlayerId actor, TileId called, TileId t1, TileId t2, PlayerId from);
    static Action make_daiminkan(PlayerId actor, TileId called, PlayerId from);
    static Action make_kakan(PlayerId actor, TileId added);
    static Action make_ankan(PlayerId actor, TileType tile_type);
    static Action make_skip(PlayerId actor);
    static Action make_kyuushu(PlayerId actor);

    // 文字列表現
    std::string to_string() const;

    // 比較
    bool operator==(const Action& other) const;
    bool operator!=(const Action& other) const { return !(*this == other); }
};

}  // namespace mahjong
