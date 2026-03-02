#include <gtest/gtest.h>
#include "rules/fu_calculator.h"

using namespace mahjong;

static AgariDecomposition make_decomp(TileType jantai,
    std::initializer_list<MentsuInfo> mentsu) {
    AgariDecomposition d;
    d.jantai = jantai;
    d.mentsu_list = mentsu;
    return d;
}

TEST(FuCalculatorTest, PinfuTsumo20) {
    // 平和ツモ → 20符固定
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Ryanmen, tile_type::S8,
        true, true, true,  // tsumo, menzen, pinfu
        tile_type::TON, tile_type::NAN);
    EXPECT_EQ(result.total_fu, 20);
}

TEST(FuCalculatorTest, PinfuRon30) {
    // 平和ロン → 20 + 10(門前ロン) = 30符
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Ryanmen, tile_type::S8,
        false, true, false,  // ron, menzen, 平和はロンでは符計算は平和計算不要
        tile_type::TON, tile_type::NAN);
    EXPECT_EQ(result.total_fu, 30);
}

TEST(FuCalculatorTest, MenzenRonKafu) {
    // 門前ロン: 20 + 10(門前ロン) + 面子符 → 切り上げ
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Koutsu, tile_type::M1, false},  // 么九暗刻 = 8符
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Ryanmen, tile_type::S8,
        false, true, false,
        tile_type::TON, tile_type::NAN);
    // 20 + 10 + 8 = 38 → 40符
    EXPECT_EQ(result.total_fu, 40);
}

TEST(FuCalculatorTest, TsumoFu) {
    // ツモ（平和以外）: 20 + 2(ツモ符) + 面子符
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Koutsu, tile_type::M2, false},  // 中張暗刻 = 4符
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Shanpon, tile_type::M2,
        true, true, false,
        tile_type::TON, tile_type::NAN);
    // 20 + 2 + 4 = 26 → 30符
    EXPECT_EQ(result.total_fu, 30);
    EXPECT_EQ(result.tsumo_fu, 2);
}

TEST(FuCalculatorTest, YaochuAnkou) {
    // 么九暗刻 = 8符
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Koutsu, tile_type::M9, false},  // 么九暗刻
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Ryanmen, tile_type::S8,
        false, true, false,
        tile_type::TON, tile_type::NAN);
    EXPECT_EQ(result.mentsu_fu, 8);
}

TEST(FuCalculatorTest, OpenKoutsu) {
    // 中張明刻 = 2符
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Koutsu, tile_type::M2, true},  // 中張明刻
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Ryanmen, tile_type::S8,
        false, false, false,
        tile_type::TON, tile_type::NAN);
    EXPECT_EQ(result.mentsu_fu, 2);
}

TEST(FuCalculatorTest, Kantsu) {
    // 中張暗槓 = 16符
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Kantsu, tile_type::M2, false},  // 中張暗槓
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Ryanmen, tile_type::S8,
        true, true, false,
        tile_type::TON, tile_type::NAN);
    EXPECT_EQ(result.mentsu_fu, 16);
}

TEST(FuCalculatorTest, OpenKantsu) {
    // 么九明槓 = 16符
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Kantsu, tile_type::M1, true},  // 么九明槓
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Ryanmen, tile_type::S8,
        false, false, false,
        tile_type::TON, tile_type::NAN);
    EXPECT_EQ(result.mentsu_fu, 16);
}

TEST(FuCalculatorTest, YaochuAnkanFu) {
    // 么九暗槓 = 32符
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Kantsu, tile_type::M1, false},  // 么九暗槓
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Ryanmen, tile_type::S8,
        true, true, false,
        tile_type::TON, tile_type::NAN);
    EXPECT_EQ(result.mentsu_fu, 32);
}

TEST(FuCalculatorTest, JantaiFu) {
    // 三元牌雀頭 = 2符
    auto decomp = make_decomp(tile_type::HAKU, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Shanpon, tile_type::S6,
        true, true, false,
        tile_type::TON, tile_type::NAN);
    EXPECT_EQ(result.jantai_fu, 2);
}

TEST(FuCalculatorTest, BakazeJikazeJantaiFu) {
    // 場風=東、自風=東の雀頭 → 4符
    auto decomp = make_decomp(tile_type::TON, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Shanpon, tile_type::S6,
        true, true, false,
        tile_type::TON, tile_type::TON);
    EXPECT_EQ(result.jantai_fu, 4);
}

TEST(FuCalculatorTest, KanchanWaitFu) {
    // 嵌張待ち = 2符
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Kanchan, tile_type::M2,
        false, true, false,
        tile_type::TON, tile_type::NAN);
    EXPECT_EQ(result.wait_fu, 2);
}

TEST(FuCalculatorTest, PenchanWaitFu) {
    // 辺張待ち = 2符
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Penchan, tile_type::M3,
        false, true, false,
        tile_type::TON, tile_type::NAN);
    EXPECT_EQ(result.wait_fu, 2);
}

TEST(FuCalculatorTest, TankiWaitFu) {
    // 単騎待ち = 2符
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Tanki, tile_type::M5,
        false, true, false,
        tile_type::TON, tile_type::NAN);
    EXPECT_EQ(result.wait_fu, 2);
}

TEST(FuCalculatorTest, KuiPinfuForm30) {
    // 喰い平和形（副底20符、符加算なし、非門前ロン）→ 30符
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, true},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Ryanmen, tile_type::S8,
        false, false, false,
        tile_type::TON, tile_type::NAN);
    EXPECT_EQ(result.total_fu, 30);
}

TEST(FuCalculatorTest, RoundUp) {
    // 22符 → 30符
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    // 門前ツモ、非平和（嵌張待ち）
    auto result = fu_calculator::calculate_fu(
        decomp, WaitType::Kanchan, tile_type::M2,
        true, true, false,
        tile_type::TON, tile_type::NAN);
    // 20 + 2(ツモ) + 2(嵌張) = 24 → 30符
    EXPECT_EQ(result.total_fu, 30);
}
