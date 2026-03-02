#include "core/event.h"

namespace mahjong {

Event Event::make_round_start() {
    Event e{};
    e.type = EventType::RoundStart;
    e.actor = 255;
    e.target = 255;
    e.tile = 255;
    e.riichi = false;
    e.round_end_reason = RoundEndReason::None;
    return e;
}

Event Event::make_deal(PlayerId player) {
    Event e{};
    e.type = EventType::Deal;
    e.actor = player;
    e.target = 255;
    e.tile = 255;
    e.riichi = false;
    e.round_end_reason = RoundEndReason::None;
    return e;
}

Event Event::make_draw(PlayerId player, TileId tile) {
    Event e{};
    e.type = EventType::Draw;
    e.actor = player;
    e.target = 255;
    e.tile = tile;
    e.riichi = false;
    e.round_end_reason = RoundEndReason::None;
    return e;
}

Event Event::make_discard(PlayerId player, TileId tile, bool riichi) {
    Event e{};
    e.type = EventType::Discard;
    e.actor = player;
    e.target = 255;
    e.tile = tile;
    e.riichi = riichi;
    e.round_end_reason = RoundEndReason::None;
    return e;
}

Event Event::make_chi(PlayerId actor, TileId called) {
    Event e{};
    e.type = EventType::Chi;
    e.actor = actor;
    e.target = 255;
    e.tile = called;
    e.meld_type = MeldType::Chi;
    e.riichi = false;
    e.round_end_reason = RoundEndReason::None;
    return e;
}

Event Event::make_pon(PlayerId actor, TileId called, PlayerId from) {
    Event e{};
    e.type = EventType::Pon;
    e.actor = actor;
    e.target = from;
    e.tile = called;
    e.meld_type = MeldType::Pon;
    e.riichi = false;
    e.round_end_reason = RoundEndReason::None;
    return e;
}

Event Event::make_kan(PlayerId actor, MeldType kan_type, TileId tile) {
    Event e{};
    e.type = EventType::Kan;
    e.actor = actor;
    e.target = 255;
    e.tile = tile;
    e.meld_type = kan_type;
    e.riichi = false;
    e.round_end_reason = RoundEndReason::None;
    return e;
}

Event Event::make_dora_reveal(TileId indicator) {
    Event e{};
    e.type = EventType::DoraReveal;
    e.actor = 255;
    e.target = 255;
    e.tile = indicator;
    e.riichi = false;
    e.round_end_reason = RoundEndReason::None;
    return e;
}

Event Event::make_ron(PlayerId winner, PlayerId loser) {
    Event e{};
    e.type = EventType::Ron;
    e.actor = winner;
    e.target = loser;
    e.tile = 255;
    e.riichi = false;
    e.round_end_reason = RoundEndReason::None;
    return e;
}

Event Event::make_tsumo(PlayerId winner) {
    Event e{};
    e.type = EventType::Tsumo;
    e.actor = winner;
    e.target = 255;
    e.tile = 255;
    e.riichi = false;
    e.round_end_reason = RoundEndReason::None;
    return e;
}

Event Event::make_abortive_draw() {
    Event e{};
    e.type = EventType::AbortiveDraw;
    e.actor = 255;
    e.target = 255;
    e.tile = 255;
    e.riichi = false;
    e.round_end_reason = RoundEndReason::None;
    return e;
}

Event Event::make_exhaustive_draw() {
    Event e{};
    e.type = EventType::ExhaustiveDraw;
    e.actor = 255;
    e.target = 255;
    e.tile = 255;
    e.riichi = false;
    e.round_end_reason = RoundEndReason::None;
    return e;
}

Event Event::make_round_end(RoundEndReason reason) {
    Event e{};
    e.type = EventType::RoundEnd;
    e.actor = 255;
    e.target = 255;
    e.tile = 255;
    e.riichi = false;
    e.round_end_reason = reason;
    return e;
}

Event Event::make_match_end() {
    Event e{};
    e.type = EventType::MatchEnd;
    e.actor = 255;
    e.target = 255;
    e.tile = 255;
    e.riichi = false;
    e.round_end_reason = RoundEndReason::None;
    return e;
}

std::string Event::to_string() const {
    std::string result = mahjong::to_string(type);
    if (actor != 255) {
        result += "(actor=" + std::to_string(actor);
        if (target != 255) {
            result += ",target=" + std::to_string(target);
        }
        if (tile != 255) {
            result += ",tile=" + Tile::from_id(tile).to_string();
        }
        if (riichi) {
            result += ",riichi";
        }
        result += ")";
    } else if (tile != 255) {
        result += "(tile=" + Tile::from_id(tile).to_string() + ")";
    }
    return result;
}

}  // namespace mahjong
