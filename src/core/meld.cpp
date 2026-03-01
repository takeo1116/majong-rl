#include "core/meld.h"
#include <algorithm>
#include <cassert>

namespace mahjong {

Meld Meld::make_chi(TileId called, TileId t1, TileId t2, PlayerId from) {
    Meld m;
    m.type = MeldType::Chi;
    m.tiles = {called, t1, t2, 255};
    m.tile_count = 3;
    m.from_player = from;
    m.called_tile = called;
    return m;
}

Meld Meld::make_pon(TileId called, TileId t1, TileId t2, PlayerId from) {
    Meld m;
    m.type = MeldType::Pon;
    m.tiles = {called, t1, t2, 255};
    m.tile_count = 3;
    m.from_player = from;
    m.called_tile = called;
    return m;
}

Meld Meld::make_daiminkan(TileId called, TileId t1, TileId t2, TileId t3, PlayerId from) {
    Meld m;
    m.type = MeldType::Daiminkan;
    m.tiles = {called, t1, t2, t3};
    m.tile_count = 4;
    m.from_player = from;
    m.called_tile = called;
    return m;
}

Meld Meld::make_ankan(TileId t1, TileId t2, TileId t3, TileId t4, PlayerId self) {
    Meld m;
    m.type = MeldType::Ankan;
    m.tiles = {t1, t2, t3, t4};
    m.tile_count = 4;
    m.from_player = self;
    m.called_tile = t1;  // 暗槓では特に鳴き元はないが、代表牌を入れておく
    return m;
}

Meld Meld::make_kakan(const Meld& pon, TileId added) {
    assert(pon.type == MeldType::Pon);
    Meld m;
    m.type = MeldType::Kakan;
    m.tiles = {pon.tiles[0], pon.tiles[1], pon.tiles[2], added};
    m.tile_count = 4;
    m.from_player = pon.from_player;
    m.called_tile = pon.called_tile;
    return m;
}

TileType Meld::base_type() const {
    // 構成牌の最小 TileType を返す
    TileType min_type = 255;
    for (uint8_t i = 0; i < tile_count; ++i) {
        TileType t = tiles[i] / 4;
        if (t < min_type) min_type = t;
    }
    return min_type;
}

std::string Meld::to_string() const {
    std::string result = mahjong::to_string(type) + "[";
    for (uint8_t i = 0; i < tile_count; ++i) {
        if (i > 0) result += ",";
        result += Tile::from_id(tiles[i]).to_string();
    }
    result += "]";
    return result;
}

}  // namespace mahjong
