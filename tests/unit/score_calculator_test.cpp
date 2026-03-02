#include <gtest/gtest.h>
#include "rules/score_calculator.h"
#include "rules/agari.h"
#include "engine/hand_utils.h"

using namespace mahjong;

TEST(ScoreCalculatorTest, Ceil100) {
    EXPECT_EQ(score_calculator::ceil100(100), 100);
    EXPECT_EQ(score_calculator::ceil100(101), 200);
    EXPECT_EQ(score_calculator::ceil100(1920), 2000);
    EXPECT_EQ(score_calculator::ceil100(2000), 2000);
    EXPECT_EQ(score_calculator::ceil100(0), 0);
}

TEST(ScoreCalculatorTest, BasePointNormal) {
    // 1翻30符 → 30 * 2^3 = 240
    EXPECT_EQ(score_calculator::calculate_base_point(1, 30), 240);
    // 2翻30符 → 30 * 2^4 = 480
    EXPECT_EQ(score_calculator::calculate_base_point(2, 30), 480);
    // 3翻30符 → 30 * 2^5 = 960
    EXPECT_EQ(score_calculator::calculate_base_point(3, 30), 960);
    // 4翻30符 → 30 * 2^6 = 1920 (切り上げ満貫なし)
    EXPECT_EQ(score_calculator::calculate_base_point(4, 30), 1920);
    // 3翻60符 → 60 * 2^5 = 1920
    EXPECT_EQ(score_calculator::calculate_base_point(3, 60), 1920);
}

TEST(ScoreCalculatorTest, BasePointMangan) {
    // 5翻 → 満貫 = 2000
    EXPECT_EQ(score_calculator::calculate_base_point(5, 30), 2000);
}

TEST(ScoreCalculatorTest, BasePointHaneman) {
    // 6翻 → 跳満 = 3000
    EXPECT_EQ(score_calculator::calculate_base_point(6, 30), 3000);
    EXPECT_EQ(score_calculator::calculate_base_point(7, 30), 3000);
}

TEST(ScoreCalculatorTest, BasePointBaiman) {
    // 8翻 → 倍満 = 4000
    EXPECT_EQ(score_calculator::calculate_base_point(8, 30), 4000);
    EXPECT_EQ(score_calculator::calculate_base_point(10, 30), 4000);
}

TEST(ScoreCalculatorTest, BasePointSanbaiman) {
    // 11翻 → 三倍満 = 6000
    EXPECT_EQ(score_calculator::calculate_base_point(11, 30), 6000);
    EXPECT_EQ(score_calculator::calculate_base_point(12, 30), 6000);
}

TEST(ScoreCalculatorTest, BasePointKazoeYakuman) {
    // 13翻以上 → 数え役満 = 8000
    EXPECT_EQ(score_calculator::calculate_base_point(13, 30), 8000);
    EXPECT_EQ(score_calculator::calculate_base_point(20, 30), 8000);
}

TEST(ScoreCalculatorTest, NoKiriageMangan) {
    // 4翻30符 → 基本点1920、切り上げ満貫なし
    EXPECT_EQ(score_calculator::calculate_base_point(4, 30), 1920);
    // 子ロン: 1920 * 4 = 7680 → ceil100 = 7700
    // 満貫なら8000だが、切り上げ満貫なしなので7700
}

TEST(ScoreCalculatorTest, ManganBoundary) {
    // 4翻40符 → 40 * 64 = 2560 → 満貫(2000)
    EXPECT_EQ(score_calculator::calculate_base_point(4, 40), 2000);
}

TEST(ScoreCalculatorTest, HighPointMethodBasic) {
    // 1m2m3m 4m5m6m 7m8m9m 白白白 中中 → 役牌(白)
    auto counts = hand_utils::make_type_counts({
        0, 4, 8, 12, 16, 20, 24, 28, 32,  // 1m-9m
        124, 125, 126,  // 白×3
        132, 133        // 中×2
    });
    auto decomps = agari::enumerate_decompositions(counts, {});
    ASSERT_FALSE(decomps.empty());

    WinContext ctx{};
    ctx.agari_tile = tile_type::CHUN;  // 中をツモ
    ctx.is_tsumo = true;
    ctx.is_menzen = true;
    ctx.is_riichi = false;
    ctx.is_ippatsu = false;
    ctx.is_rinshan = false;
    ctx.is_chankan = false;
    ctx.is_haitei = false;
    ctx.is_houtei = false;
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::NAN;
    ctx.all_tile_ids = {0, 4, 8, 12, 16, 20, 24, 28, 32, 124, 125, 126, 132, 133};

    auto result = score_calculator::calculate_win_score(decomps, ctx, false, 0);
    EXPECT_TRUE(result.valid);
    EXPECT_GE(result.total_han, 2);  // 白 + 門前ツモ
}

TEST(ScoreCalculatorTest, DealerRon) {
    // 親ロン: 白(1翻), 40符(20+10門前ロン+8白暗刻+2単騎=40)
    // base = 40*2^3 = 320, dealer ron = ceil100(320*6) = ceil100(1920) = 2000
    AgariDecomposition decomp;
    decomp.jantai = tile_type::M5;
    decomp.mentsu_list = {
        {MentsuKind::Koutsu, tile_type::HAKU, false},
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
    };

    WinContext ctx{};
    ctx.agari_tile = tile_type::M5;
    ctx.is_tsumo = false;
    ctx.is_menzen = true;
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::NAN;

    auto result = score_calculator::calculate_win_score({decomp}, ctx, true, 0);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.payment.from_ron, 2000);
}

TEST(ScoreCalculatorTest, NonDealerRon) {
    // 子ロン: 白(1翻), 50符(20+10門前ロン+8白暗刻+4S3暗刻+2単騎=44→50)
    // base = 50*2^3 = 400, non-dealer ron = ceil100(400*4) = 1600
    AgariDecomposition decomp;
    decomp.jantai = tile_type::M5;
    decomp.mentsu_list = {
        {MentsuKind::Koutsu, tile_type::HAKU, false},
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Koutsu, tile_type::S3, false},
    };

    WinContext ctx{};
    ctx.agari_tile = tile_type::M5;
    ctx.is_tsumo = false;
    ctx.is_menzen = true;
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::NAN;

    auto result = score_calculator::calculate_win_score({decomp}, ctx, false, 0);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.payment.from_ron, 1600);
}

TEST(ScoreCalculatorTest, HonbaRon) {
    // 積み棒テスト: 2本場のロン → +600
    AgariDecomposition decomp;
    decomp.jantai = tile_type::M5;
    decomp.mentsu_list = {
        {MentsuKind::Koutsu, tile_type::HAKU, false},
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
    };

    WinContext ctx{};
    ctx.agari_tile = tile_type::M5;
    ctx.is_tsumo = false;
    ctx.is_menzen = true;
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::NAN;

    auto result_0 = score_calculator::calculate_win_score({decomp}, ctx, true, 0);
    auto result_2 = score_calculator::calculate_win_score({decomp}, ctx, true, 2);
    EXPECT_TRUE(result_0.valid);
    EXPECT_TRUE(result_2.valid);
    EXPECT_EQ(result_2.payment.from_ron - result_0.payment.from_ron, 600);
}

TEST(ScoreCalculatorTest, TsumoPayment) {
    // 子ツモ: 門前ツモ(1翻)+白(1翻)=2翻, 40符(20+8白暗刻+2ツモ+2単騎=32→40)
    // base = 40*2^4 = 640
    // 親: ceil100(640*2) = 1300, 子: ceil100(640) = 700
    AgariDecomposition decomp;
    decomp.jantai = tile_type::M5;
    decomp.mentsu_list = {
        {MentsuKind::Koutsu, tile_type::HAKU, false},
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
    };

    WinContext ctx{};
    ctx.agari_tile = tile_type::M5;
    ctx.is_tsumo = true;
    ctx.is_menzen = true;
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::NAN;

    auto result = score_calculator::calculate_win_score({decomp}, ctx, false, 0);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.total_han, 2);
    EXPECT_EQ(result.fu, 40);
    EXPECT_EQ(result.payment.from_dealer, 1300);
    EXPECT_EQ(result.payment.from_non_dealer, 700);
}

TEST(ScoreCalculatorTest, NoYakuInvalid) {
    // 役なし → valid = false
    AgariDecomposition decomp;
    decomp.jantai = tile_type::M1;
    decomp.mentsu_list = {
        {MentsuKind::Shuntsu, tile_type::M3, false},
        {MentsuKind::Shuntsu, tile_type::P1, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    };

    WinContext ctx{};
    ctx.agari_tile = tile_type::M4;
    ctx.is_tsumo = false;
    ctx.is_menzen = true;
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::NAN;

    auto result = score_calculator::calculate_win_score({decomp}, ctx, false, 0);
    EXPECT_FALSE(result.valid);
}

TEST(ScoreCalculatorTest, DealerTsumo) {
    // 親ツモ: 門前ツモ(1)+白(1)+場風東(1)+自風東(1)=4翻
    // 40符(20+8白暗刻+8東暗刻+2ツモ+2単騎=40)
    // base = 40*2^6 = 2560 → 満貫(2000)
    // 親ツモ満貫: 各子 ceil100(2000*2) = 4000
    AgariDecomposition decomp;
    decomp.jantai = tile_type::M5;
    decomp.mentsu_list = {
        {MentsuKind::Koutsu, tile_type::HAKU, false},
        {MentsuKind::Koutsu, tile_type::TON, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
    };

    WinContext ctx{};
    ctx.agari_tile = tile_type::M5;
    ctx.is_tsumo = true;
    ctx.is_menzen = true;
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::TON;

    auto result = score_calculator::calculate_win_score({decomp}, ctx, true, 0);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.total_han, 4);
    EXPECT_EQ(result.payment.from_non_dealer, 4000);
}

TEST(ScoreCalculatorTest, HonbaTsumo) {
    // 積み棒3本場のツモ → 各者から+100*3 = +300
    AgariDecomposition decomp;
    decomp.jantai = tile_type::M5;
    decomp.mentsu_list = {
        {MentsuKind::Koutsu, tile_type::HAKU, false},
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
    };

    WinContext ctx{};
    ctx.agari_tile = tile_type::M5;
    ctx.is_tsumo = true;
    ctx.is_menzen = true;
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::NAN;

    auto result_0 = score_calculator::calculate_win_score({decomp}, ctx, false, 0);
    auto result_3 = score_calculator::calculate_win_score({decomp}, ctx, false, 3);
    EXPECT_TRUE(result_0.valid);
    EXPECT_TRUE(result_3.valid);
    // 各者から+100*3/3 = +100 per person...
    // honba_total = 300*3 = 900, per person = 300
    EXPECT_EQ(result_3.payment.from_dealer - result_0.payment.from_dealer, 300);
    EXPECT_EQ(result_3.payment.from_non_dealer - result_0.payment.from_non_dealer, 300);
}
