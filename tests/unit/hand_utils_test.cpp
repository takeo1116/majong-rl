#include <gtest/gtest.h>
#include "engine/hand_utils.h"
#include "core/tile.h"

using namespace mahjong;
using namespace mahjong::hand_utils;

// ヘルパー: TileType 配列からカウントを作成
static std::array<int, kNumTileTypes> counts_from_types(const std::vector<TileType>& types) {
    std::array<int, kNumTileTypes> counts{};
    for (auto t : types) counts[t]++;
    return counts;
}

// 和了形チェック
TEST(HandUtilsTest, IsAgariBasic) {
    // 1m1m1m 2m3m4m 5m6m7m 8m9m9m 9m9m → 和了
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M1, tile_type::M1,
        tile_type::M2, tile_type::M3, tile_type::M4,
        tile_type::M5, tile_type::M6, tile_type::M7,
        tile_type::M8, tile_type::M9, tile_type::M9,
        tile_type::M9, tile_type::M8
    });
    EXPECT_TRUE(is_agari(counts));
}

TEST(HandUtilsTest, IsAgariAllKoutsu) {
    // 1m*3 2m*3 3m*3 4m*3 5m*2 → 和了
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M1, tile_type::M1,
        tile_type::M2, tile_type::M2, tile_type::M2,
        tile_type::M3, tile_type::M3, tile_type::M3,
        tile_type::M4, tile_type::M4, tile_type::M4,
        tile_type::M5, tile_type::M5
    });
    EXPECT_TRUE(is_agari(counts));
}

TEST(HandUtilsTest, IsAgariNotComplete) {
    // 不完全な手
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M2, tile_type::M3,
        tile_type::M4, tile_type::M5, tile_type::M6,
        tile_type::M7, tile_type::M8, tile_type::M9,
        tile_type::P1, tile_type::P2, tile_type::P3,
        tile_type::S1, tile_type::S2
    });
    EXPECT_FALSE(is_agari(counts));
}

TEST(HandUtilsTest, IsAgariWithMelds) {
    // 副露ありの和了（5枚: 1面子 + 1雀頭）
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M2, tile_type::M3,
        tile_type::P1, tile_type::P1
    });
    EXPECT_TRUE(is_agari(counts));
}

TEST(HandUtilsTest, IsAgariJustPair) {
    // 雀頭のみ（2枚）
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M1
    });
    EXPECT_TRUE(is_agari(counts));
}

// テンパイチェック
TEST(HandUtilsTest, IsTenpaiBasic) {
    // 1m2m3m 4m5m6m 7m8m9m 1p2p3p 5p → 5p待ちテンパイ
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M2, tile_type::M3,
        tile_type::M4, tile_type::M5, tile_type::M6,
        tile_type::M7, tile_type::M8, tile_type::M9,
        tile_type::P1, tile_type::P2, tile_type::P3,
        tile_type::P5
    });
    EXPECT_TRUE(is_tenpai(counts));
}

TEST(HandUtilsTest, IsNotTenpai) {
    // バラバラの手
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M3, tile_type::M5,
        tile_type::P2, tile_type::P4, tile_type::P6,
        tile_type::S1, tile_type::S3, tile_type::S5,
        tile_type::TON, tile_type::NAN, tile_type::SHA,
        tile_type::PEI
    });
    EXPECT_FALSE(is_tenpai(counts));
}

// 待ち牌
TEST(HandUtilsTest, GetWaits) {
    // 1m2m3m 4m5m6m 7m8m9m 1p1p1p 東 → 東待ち
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M2, tile_type::M3,
        tile_type::M4, tile_type::M5, tile_type::M6,
        tile_type::M7, tile_type::M8, tile_type::M9,
        tile_type::P1, tile_type::P1, tile_type::P1,
        tile_type::TON
    });
    auto waits = get_waits(counts);
    EXPECT_EQ(waits.size(), 1);
    EXPECT_EQ(waits[0], tile_type::TON);
}

TEST(HandUtilsTest, GetWaitsMultiple) {
    // 1m2m3m 4m5m6m 7m8m9m 1p2p3p 4p → 4p-7p 待ちではなく...
    // 正しく: 1m2m3m 4m5m6m 7m8m9m 1p2p3p 4p → 4p を雀頭にすれば和了
    // 4p 待ちのシャンポン？ いいえ、これは 4p + X で和了になるXを探す
    // 実際: 4p が雀頭で 4面子完成 → 和了（14枚ならis_agariだが13枚でtenpai）
    // 4p + 4p → 4p雀頭 + 4面子 → テンパイ
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M2, tile_type::M3,
        tile_type::M4, tile_type::M5, tile_type::M6,
        tile_type::M7, tile_type::M8, tile_type::M9,
        tile_type::P1, tile_type::P2, tile_type::P3,
        tile_type::P4
    });
    auto waits = get_waits(counts);
    EXPECT_FALSE(waits.empty());
    // 4p 待ち（単騎）
    bool has_p4 = std::find(waits.begin(), waits.end(), tile_type::P4) != waits.end();
    EXPECT_TRUE(has_p4);
}

// 九種九牌のカウント
TEST(HandUtilsTest, CountYaochuTypes) {
    // 全て么九牌
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M9, tile_type::P1, tile_type::P9,
        tile_type::S1, tile_type::S9, tile_type::TON, tile_type::NAN,
        tile_type::SHA, tile_type::PEI, tile_type::HAKU, tile_type::HATSU,
        tile_type::CHUN, tile_type::M1
    });
    EXPECT_EQ(count_yaochu_types(counts), 13);
}

TEST(HandUtilsTest, CountYaochuTypesLow) {
    // 么九牌が少ない
    auto counts = counts_from_types({
        tile_type::M2, tile_type::M3, tile_type::M4,
        tile_type::M5, tile_type::M6, tile_type::M7,
        tile_type::P2, tile_type::P3, tile_type::P4,
        tile_type::P5, tile_type::P6, tile_type::P7,
        tile_type::S2, tile_type::S3
    });
    EXPECT_EQ(count_yaochu_types(counts), 0);
}

// TileId から type counts
TEST(HandUtilsTest, MakeTypeCounts) {
    std::vector<TileId> hand = {0, 1, 2, 4, 5};
    // 0,1,2 → type 0 (1m) x3
    // 4,5 → type 1 (2m) x2
    auto counts = make_type_counts(hand);
    EXPECT_EQ(counts[tile_type::M1], 3);
    EXPECT_EQ(counts[tile_type::M2], 2);
    EXPECT_EQ(counts[tile_type::M3], 0);
}

// 面子分解
TEST(HandUtilsTest, CanDecomposeMentsu) {
    // 1m2m3m 4m5m6m → 2面子
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M2, tile_type::M3,
        tile_type::M4, tile_type::M5, tile_type::M6
    });
    EXPECT_TRUE(can_decompose_mentsu(counts));
}

TEST(HandUtilsTest, CannotDecomposeMentsu) {
    // 1m2m4m → 面子にならない
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M2, tile_type::M4
    });
    EXPECT_FALSE(can_decompose_mentsu(counts));
}
