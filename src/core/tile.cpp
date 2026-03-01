#include "core/tile.h"
#include <cassert>

namespace mahjong {

// 136枚の牌配置:
// 各種別に4枚ずつ: id = type * 4 + 0..3
// 赤牌は以下の位置:
//   赤5m: id = 4*4 + 0 = 16  (5m の最初の1枚を赤とする)
//   赤5p: id = 13*4 + 0 = 52 (5p の最初の1枚を赤とする)
//   赤5s: id = 22*4 + 0 = 88 (5s の最初の1枚を赤とする)

bool Tile::is_red_id(TileId id) {
    return id == 16 || id == 52 || id == 88;
}

Tile Tile::from_id(TileId id) {
    assert(id < kNumTiles);
    Tile t;
    t.id = id;
    t.type = id / 4;
    t.is_red = is_red_id(id);
    return t;
}

Suit Tile::suit_of(TileType type) {
    assert(type < kNumTileTypes);
    if (type < 9) return Suit::Man;
    if (type < 18) return Suit::Pin;
    if (type < 27) return Suit::Sou;
    return Suit::Ji;
}

int Tile::number_of(TileType type) {
    assert(type < kNumTileTypes);
    if (type >= 27) return 0;  // 字牌
    return (type % 9) + 1;
}

bool Tile::is_jihai(TileType type) {
    return type >= 27;
}

bool Tile::is_yaochu(TileType type) {
    if (type >= 27) return true;  // 字牌は全て么九牌
    int num = (type % 9) + 1;
    return num == 1 || num == 9;
}

bool Tile::is_sangenpai(TileType type) {
    return type == tile_type::HAKU || type == tile_type::HATSU || type == tile_type::CHUN;
}

bool Tile::is_kazehai(TileType type) {
    return type >= tile_type::TON && type <= tile_type::PEI;
}

TileType Tile::next_dora(TileType indicator) {
    assert(indicator < kNumTileTypes);
    if (indicator < 27) {
        // 数牌: 同スート内で循環 (9 → 1)
        int suit_base = (indicator / 9) * 9;
        int num = indicator % 9;
        return suit_base + (num + 1) % 9;
    }
    // 字牌
    if (indicator <= tile_type::PEI) {
        // 風牌: 東→南→西→北→東
        return tile_type::TON + (indicator - tile_type::TON + 1) % 4;
    }
    // 三元牌: 白→發→中→白
    return tile_type::HAKU + (indicator - tile_type::HAKU + 1) % 3;
}

std::string Tile::to_string() const {
    std::string result;
    if (is_red) {
        result += "r";
    }
    result += type_to_string(type);
    return result;
}

std::string Tile::type_to_string(TileType type) {
    assert(type < kNumTileTypes);
    static const char* names[] = {
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
        "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
        "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
        "東", "南", "西", "北", "白", "發", "中"
    };
    return names[type];
}

namespace {
std::array<Tile, kNumTiles> build_all_tiles() {
    std::array<Tile, kNumTiles> tiles;
    for (int i = 0; i < kNumTiles; ++i) {
        tiles[i] = Tile::from_id(static_cast<TileId>(i));
    }
    return tiles;
}
}  // namespace

const std::array<Tile, kNumTiles>& all_tiles() {
    static const auto tiles = build_all_tiles();
    return tiles;
}

}  // namespace mahjong
