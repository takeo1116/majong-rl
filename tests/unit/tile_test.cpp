#include <gtest/gtest.h>
#include "core/tile.h"

using namespace mahjong;

// 全牌テーブルの基本検証
TEST(TileTest, AllTilesCount) {
    const auto& tiles = all_tiles();
    EXPECT_EQ(tiles.size(), kNumTiles);
}

// TileId から Tile への変換
TEST(TileTest, FromId) {
    Tile t0 = Tile::from_id(0);
    EXPECT_EQ(t0.id, 0);
    EXPECT_EQ(t0.type, tile_type::M1);
    EXPECT_FALSE(t0.is_red);

    Tile t135 = Tile::from_id(135);
    EXPECT_EQ(t135.id, 135);
    EXPECT_EQ(t135.type, tile_type::CHUN);
    EXPECT_FALSE(t135.is_red);
}

// 赤牌の判定
TEST(TileTest, RedTiles) {
    // 赤5m: id=16
    EXPECT_TRUE(Tile::is_red_id(16));
    Tile red5m = Tile::from_id(16);
    EXPECT_TRUE(red5m.is_red);
    EXPECT_EQ(red5m.type, tile_type::M5);

    // 赤5p: id=52
    EXPECT_TRUE(Tile::is_red_id(52));
    Tile red5p = Tile::from_id(52);
    EXPECT_TRUE(red5p.is_red);
    EXPECT_EQ(red5p.type, tile_type::P5);

    // 赤5s: id=88
    EXPECT_TRUE(Tile::is_red_id(88));
    Tile red5s = Tile::from_id(88);
    EXPECT_TRUE(red5s.is_red);
    EXPECT_EQ(red5s.type, tile_type::S5);

    // 赤でない牌
    EXPECT_FALSE(Tile::is_red_id(0));
    EXPECT_FALSE(Tile::is_red_id(17));
    EXPECT_FALSE(Tile::is_red_id(53));
}

// 赤牌は各種1枚ずつ（計3枚）
TEST(TileTest, RedTileCount) {
    const auto& tiles = all_tiles();
    int red_count = 0;
    for (const auto& t : tiles) {
        if (t.is_red) ++red_count;
    }
    EXPECT_EQ(red_count, 3);
}

// スート判定
TEST(TileTest, SuitOf) {
    EXPECT_EQ(Tile::suit_of(tile_type::M1), Suit::Man);
    EXPECT_EQ(Tile::suit_of(tile_type::M9), Suit::Man);
    EXPECT_EQ(Tile::suit_of(tile_type::P1), Suit::Pin);
    EXPECT_EQ(Tile::suit_of(tile_type::P9), Suit::Pin);
    EXPECT_EQ(Tile::suit_of(tile_type::S1), Suit::Sou);
    EXPECT_EQ(Tile::suit_of(tile_type::S9), Suit::Sou);
    EXPECT_EQ(Tile::suit_of(tile_type::TON), Suit::Ji);
    EXPECT_EQ(Tile::suit_of(tile_type::CHUN), Suit::Ji);
}

// 数字取得
TEST(TileTest, NumberOf) {
    EXPECT_EQ(Tile::number_of(tile_type::M1), 1);
    EXPECT_EQ(Tile::number_of(tile_type::M9), 9);
    EXPECT_EQ(Tile::number_of(tile_type::P5), 5);
    EXPECT_EQ(Tile::number_of(tile_type::S1), 1);
    EXPECT_EQ(Tile::number_of(tile_type::TON), 0);  // 字牌
}

// 字牌判定
TEST(TileTest, IsJihai) {
    EXPECT_FALSE(Tile::is_jihai(tile_type::M1));
    EXPECT_FALSE(Tile::is_jihai(tile_type::S9));
    EXPECT_TRUE(Tile::is_jihai(tile_type::TON));
    EXPECT_TRUE(Tile::is_jihai(tile_type::CHUN));
}

// 么九牌判定
TEST(TileTest, IsYaochu) {
    EXPECT_TRUE(Tile::is_yaochu(tile_type::M1));
    EXPECT_TRUE(Tile::is_yaochu(tile_type::M9));
    EXPECT_FALSE(Tile::is_yaochu(tile_type::M2));
    EXPECT_FALSE(Tile::is_yaochu(tile_type::M5));
    EXPECT_TRUE(Tile::is_yaochu(tile_type::TON));
    EXPECT_TRUE(Tile::is_yaochu(tile_type::HAKU));
}

// 三元牌判定
TEST(TileTest, IsSangenpai) {
    EXPECT_FALSE(Tile::is_sangenpai(tile_type::TON));
    EXPECT_TRUE(Tile::is_sangenpai(tile_type::HAKU));
    EXPECT_TRUE(Tile::is_sangenpai(tile_type::HATSU));
    EXPECT_TRUE(Tile::is_sangenpai(tile_type::CHUN));
}

// 風牌判定
TEST(TileTest, IsKazehai) {
    EXPECT_TRUE(Tile::is_kazehai(tile_type::TON));
    EXPECT_TRUE(Tile::is_kazehai(tile_type::NAN));
    EXPECT_TRUE(Tile::is_kazehai(tile_type::SHA));
    EXPECT_TRUE(Tile::is_kazehai(tile_type::PEI));
    EXPECT_FALSE(Tile::is_kazehai(tile_type::HAKU));
}

// ドラ次牌の変換
TEST(TileTest, NextDora) {
    // 数牌: 1→2, 9→1
    EXPECT_EQ(Tile::next_dora(tile_type::M1), tile_type::M2);
    EXPECT_EQ(Tile::next_dora(tile_type::M9), tile_type::M1);
    EXPECT_EQ(Tile::next_dora(tile_type::P5), tile_type::P6);
    EXPECT_EQ(Tile::next_dora(tile_type::S9), tile_type::S1);

    // 風牌: 東→南→西→北→東
    EXPECT_EQ(Tile::next_dora(tile_type::TON), tile_type::NAN);
    EXPECT_EQ(Tile::next_dora(tile_type::PEI), tile_type::TON);

    // 三元牌: 白→發→中→白
    EXPECT_EQ(Tile::next_dora(tile_type::HAKU), tile_type::HATSU);
    EXPECT_EQ(Tile::next_dora(tile_type::CHUN), tile_type::HAKU);
}

// 文字列表現
TEST(TileTest, ToString) {
    Tile t = Tile::from_id(0);
    EXPECT_EQ(t.to_string(), "1m");

    Tile red = Tile::from_id(16);
    EXPECT_EQ(red.to_string(), "r5m");

    EXPECT_EQ(Tile::type_to_string(tile_type::TON), "東");
    EXPECT_EQ(Tile::type_to_string(tile_type::CHUN), "中");
}

// 各 TileType に対して4枚ずつ存在する
TEST(TileTest, FourTilesPerType) {
    const auto& tiles = all_tiles();
    std::array<int, kNumTileTypes> counts{};
    for (const auto& t : tiles) {
        counts[t.type]++;
    }
    for (int i = 0; i < kNumTileTypes; ++i) {
        EXPECT_EQ(counts[i], 4) << "TileType " << i << " has " << counts[i] << " tiles";
    }
}
