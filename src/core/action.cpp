#include "core/action.h"

namespace mahjong {

Action Action::make_discard(PlayerId actor, TileId tile, bool riichi) {
    Action a{};
    a.type = ActionType::Discard;
    a.actor = actor;
    a.tile = tile;
    a.target_player = 255;
    a.meld_type = MeldType::Chi;  // 未使用
    a.riichi = riichi;
    a.consumed_tiles = {255, 255};
    return a;
}

Action Action::make_tsumo_win(PlayerId actor) {
    Action a{};
    a.type = ActionType::TsumoWin;
    a.actor = actor;
    a.tile = 255;
    a.target_player = 255;
    a.meld_type = MeldType::Chi;
    a.riichi = false;
    a.consumed_tiles = {255, 255};
    return a;
}

Action Action::make_ron(PlayerId actor, PlayerId target) {
    Action a{};
    a.type = ActionType::Ron;
    a.actor = actor;
    a.tile = 255;
    a.target_player = target;
    a.meld_type = MeldType::Chi;
    a.riichi = false;
    a.consumed_tiles = {255, 255};
    return a;
}

Action Action::make_chi(PlayerId actor, TileId called, TileId t1, TileId t2) {
    Action a{};
    a.type = ActionType::Chi;
    a.actor = actor;
    a.tile = called;
    a.target_player = 255;
    a.meld_type = MeldType::Chi;
    a.riichi = false;
    a.consumed_tiles = {t1, t2};
    return a;
}

Action Action::make_pon(PlayerId actor, TileId called, TileId t1, TileId t2, PlayerId from) {
    Action a{};
    a.type = ActionType::Pon;
    a.actor = actor;
    a.tile = called;
    a.target_player = from;
    a.meld_type = MeldType::Pon;
    a.riichi = false;
    a.consumed_tiles = {t1, t2};
    return a;
}

Action Action::make_daiminkan(PlayerId actor, TileId called, PlayerId from) {
    Action a{};
    a.type = ActionType::Daiminkan;
    a.actor = actor;
    a.tile = called;
    a.target_player = from;
    a.meld_type = MeldType::Daiminkan;
    a.riichi = false;
    a.consumed_tiles = {255, 255};
    return a;
}

Action Action::make_kakan(PlayerId actor, TileId added) {
    Action a{};
    a.type = ActionType::Kakan;
    a.actor = actor;
    a.tile = added;
    a.target_player = 255;
    a.meld_type = MeldType::Kakan;
    a.riichi = false;
    a.consumed_tiles = {255, 255};
    return a;
}

Action Action::make_ankan(PlayerId actor, TileType tile_type) {
    Action a{};
    a.type = ActionType::Ankan;
    a.actor = actor;
    a.tile = tile_type;  // TileType を格納（実牌IDではなく種別）
    a.target_player = 255;
    a.meld_type = MeldType::Ankan;
    a.riichi = false;
    a.consumed_tiles = {255, 255};
    return a;
}

Action Action::make_skip(PlayerId actor) {
    Action a{};
    a.type = ActionType::Skip;
    a.actor = actor;
    a.tile = 255;
    a.target_player = 255;
    a.meld_type = MeldType::Chi;
    a.riichi = false;
    a.consumed_tiles = {255, 255};
    return a;
}

Action Action::make_kyuushu(PlayerId actor) {
    Action a{};
    a.type = ActionType::Kyuushu;
    a.actor = actor;
    a.tile = 255;
    a.target_player = 255;
    a.meld_type = MeldType::Chi;
    a.riichi = false;
    a.consumed_tiles = {255, 255};
    return a;
}

std::string Action::to_string() const {
    std::string result = mahjong::to_string(type);
    result += "(actor=" + std::to_string(actor);
    if (tile != 255) {
        if (type == ActionType::Ankan) {
            result += ",type=" + Tile::type_to_string(tile);
        } else {
            result += ",tile=" + Tile::from_id(tile).to_string();
        }
    }
    if (target_player != 255) {
        result += ",target=" + std::to_string(target_player);
    }
    if (riichi) {
        result += ",riichi";
    }
    result += ")";
    return result;
}

bool Action::operator==(const Action& other) const {
    return type == other.type
        && actor == other.actor
        && tile == other.tile
        && target_player == other.target_player
        && riichi == other.riichi
        && consumed_tiles == other.consumed_tiles;
}

}  // namespace mahjong
