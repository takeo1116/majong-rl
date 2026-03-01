#pragma once

#include "core/tile.h"
#include "core/types.h"
#include <array>
#include <string>
#include <vector>

namespace mahjong {

// イベント構造体
// 各種イベントを統一的に扱う
struct Event {
    EventType type;
    PlayerId actor;                // 行動者（該当しない場合は255）
    PlayerId target;               // 対象プレイヤー（該当しない場合は255）
    TileId tile;                   // 関連する牌（該当しない場合は255）
    MeldType meld_type;            // 副露種別（鳴きイベント時のみ有効）
    bool riichi;                   // 立直フラグ
    RoundEndReason round_end_reason;  // 局終了理由（RoundEnd時のみ有効）

    // 構築ヘルパー
    static Event make_round_start();
    static Event make_deal(PlayerId player);
    static Event make_draw(PlayerId player, TileId tile);
    static Event make_discard(PlayerId player, TileId tile, bool riichi);
    static Event make_chi(PlayerId actor, TileId called);
    static Event make_pon(PlayerId actor, TileId called, PlayerId from);
    static Event make_kan(PlayerId actor, MeldType kan_type);
    static Event make_dora_reveal(TileId indicator);
    static Event make_ron(PlayerId winner, PlayerId loser);
    static Event make_tsumo(PlayerId winner);
    static Event make_abortive_draw();
    static Event make_exhaustive_draw();
    static Event make_round_end(RoundEndReason reason);
    static Event make_match_end();

    // 文字列表現
    std::string to_string() const;
};

}  // namespace mahjong
