#include <gtest/gtest.h>
#include "core/meld.h"

using namespace mahjong;

TEST(MeldTest, MakeChi) {
    // 1m(id=0) + 2m(id=4) + 3m(id=8) のチー
    Meld m = Meld::make_chi(0, 4, 8, 3);
    EXPECT_EQ(m.type, MeldType::Chi);
    EXPECT_EQ(m.tile_count, 3);
    EXPECT_EQ(m.from_player, 3);
    EXPECT_EQ(m.called_tile, 0);
    EXPECT_EQ(m.tiles[0], 0);
    EXPECT_EQ(m.tiles[1], 4);
    EXPECT_EQ(m.tiles[2], 8);
}

TEST(MeldTest, MakePon) {
    // 1m のポン
    Meld m = Meld::make_pon(0, 1, 2, 1);
    EXPECT_EQ(m.type, MeldType::Pon);
    EXPECT_EQ(m.tile_count, 3);
    EXPECT_EQ(m.from_player, 1);
}

TEST(MeldTest, MakeDaiminkan) {
    Meld m = Meld::make_daiminkan(0, 1, 2, 3, 2);
    EXPECT_EQ(m.type, MeldType::Daiminkan);
    EXPECT_EQ(m.tile_count, 4);
    EXPECT_EQ(m.from_player, 2);
}

TEST(MeldTest, MakeAnkan) {
    Meld m = Meld::make_ankan(0, 1, 2, 3, 0);
    EXPECT_EQ(m.type, MeldType::Ankan);
    EXPECT_EQ(m.tile_count, 4);
    EXPECT_EQ(m.from_player, 0);
}

TEST(MeldTest, MakeKakan) {
    Meld pon = Meld::make_pon(0, 1, 2, 1);
    Meld kakan = Meld::make_kakan(pon, 3);
    EXPECT_EQ(kakan.type, MeldType::Kakan);
    EXPECT_EQ(kakan.tile_count, 4);
    EXPECT_EQ(kakan.tiles[3], 3);
    EXPECT_EQ(kakan.from_player, 1);  // 元のポンの鳴き元を引き継ぐ
}

TEST(MeldTest, BaseType) {
    // 1m-2m-3m のチー → base_type は M1
    Meld chi = Meld::make_chi(0, 4, 8, 3);
    EXPECT_EQ(chi.base_type(), tile_type::M1);

    // 5p のポン → base_type は P5
    Meld pon = Meld::make_pon(52, 53, 54, 1);  // 5p の ID
    EXPECT_EQ(pon.base_type(), tile_type::P5);
}

TEST(MeldTest, ToString) {
    Meld m = Meld::make_pon(0, 1, 2, 1);
    std::string s = m.to_string();
    EXPECT_FALSE(s.empty());
    // "Pon[...]" のような形式
    EXPECT_NE(s.find("Pon"), std::string::npos);
}
