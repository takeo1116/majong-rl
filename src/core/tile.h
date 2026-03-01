#pragma once

#include <cstdint>
#include <string>
#include <array>

namespace mahjong {

// 牌のスート（種類）
enum class Suit : uint8_t {
    Man = 0,  // 萬子
    Pin = 1,  // 筒子
    Sou = 2,  // 索子
    Ji  = 3,  // 字牌
};

// 字牌の種別
enum class Wind : uint8_t {
    East  = 0,  // 東
    South = 1,  // 南
    West  = 2,  // 西
    North = 3,  // 北
};

// 136枚の実牌数
constexpr int kNumTiles = 136;
// 34種の牌種数
constexpr int kNumTileTypes = 34;
// プレイヤー数
constexpr int kNumPlayers = 4;

// 34種インデックス（0-33）
// 0-8: 1m-9m, 9-17: 1p-9p, 18-26: 1s-9s, 27-33: 東南西北白發中
using TileType = uint8_t;

// 136枚の実牌ID（0-135）
using TileId = uint8_t;

// プレイヤーID（0-3）
using PlayerId = uint8_t;

// TileType の定数
namespace tile_type {
    // 萬子
    constexpr TileType M1 = 0;
    constexpr TileType M2 = 1;
    constexpr TileType M3 = 2;
    constexpr TileType M4 = 3;
    constexpr TileType M5 = 4;
    constexpr TileType M6 = 5;
    constexpr TileType M7 = 6;
    constexpr TileType M8 = 7;
    constexpr TileType M9 = 8;
    // 筒子
    constexpr TileType P1 = 9;
    constexpr TileType P2 = 10;
    constexpr TileType P3 = 11;
    constexpr TileType P4 = 12;
    constexpr TileType P5 = 13;
    constexpr TileType P6 = 14;
    constexpr TileType P7 = 15;
    constexpr TileType P8 = 16;
    constexpr TileType P9 = 17;
    // 索子
    constexpr TileType S1 = 18;
    constexpr TileType S2 = 19;
    constexpr TileType S3 = 20;
    constexpr TileType S4 = 21;
    constexpr TileType S5 = 22;
    constexpr TileType S6 = 23;
    constexpr TileType S7 = 24;
    constexpr TileType S8 = 25;
    constexpr TileType S9 = 26;
    // 字牌
    constexpr TileType TON   = 27;  // 東
    constexpr TileType NAN   = 28;  // 南
    constexpr TileType SHA   = 29;  // 西
    constexpr TileType PEI   = 30;  // 北
    constexpr TileType HAKU  = 31;  // 白
    constexpr TileType HATSU = 32;  // 發
    constexpr TileType CHUN  = 33;  // 中
}  // namespace tile_type

// 牌構造体
struct Tile {
    TileId id;       // 実牌ID (0-135)
    TileType type;   // 34種インデックス (0-33)
    bool is_red;     // 赤牌フラグ

    // TileId から Tile を生成する
    static Tile from_id(TileId id);

    // TileType からスートを取得する
    static Suit suit_of(TileType type);

    // TileType から数字を取得する（字牌は0を返す）
    static int number_of(TileType type);

    // 字牌かどうか
    static bool is_jihai(TileType type);

    // 么九牌かどうか
    static bool is_yaochu(TileType type);

    // 三元牌かどうか
    static bool is_sangenpai(TileType type);

    // 風牌かどうか
    static bool is_kazehai(TileType type);

    // 赤牌のTileIdかどうか
    static bool is_red_id(TileId id);

    // ドラ次牌を返す（表示牌 → ドラ牌の種別変換）
    static TileType next_dora(TileType indicator);

    // 文字列表現
    std::string to_string() const;

    // TileType の文字列表現
    static std::string type_to_string(TileType type);
};

// 136枚の全牌テーブル（起動時に初期化）
const std::array<Tile, kNumTiles>& all_tiles();

}  // namespace mahjong
