#include <gtest/gtest.h>
#include "rules/yaku.h"
#include "rules/agari.h"

using namespace mahjong;

// ヘルパー: 基本的な WinContext を作る
static WinContext make_basic_ctx(TileType agari_tile, bool is_tsumo, bool is_menzen) {
    WinContext ctx{};
    ctx.agari_tile = agari_tile;
    ctx.is_tsumo = is_tsumo;
    ctx.is_menzen = is_menzen;
    ctx.is_riichi = false;
    ctx.is_ippatsu = false;
    ctx.is_rinshan = false;
    ctx.is_chankan = false;
    ctx.is_haitei = false;
    ctx.is_houtei = false;
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::TON;
    return ctx;
}

// ヘルパー: 面子構成を手動で作る
static AgariDecomposition make_decomp(TileType jantai,
    std::initializer_list<MentsuInfo> mentsu) {
    AgariDecomposition d;
    d.jantai = jantai;
    d.mentsu_list = mentsu;
    return d;
}

TEST(YakuTest, MenzenTsumo) {
    // 門前ツモ: 全順子 + 役牌でない雀頭
    auto decomp = make_decomp(tile_type::M1, {
        {MentsuKind::Shuntsu, tile_type::M2, false},
        {MentsuKind::Shuntsu, tile_type::P1, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::M5, false},
    });
    auto ctx = make_basic_ctx(tile_type::M5, true, true);
    ctx.jikaze = tile_type::NAN;  // 平和にならないよう

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Shanpon, ctx);
    bool found = false;
    for (const auto& y : eval.yakus) {
        if (y.type == YakuType::MenzenTsumo) { found = true; break; }
    }
    EXPECT_TRUE(found);
}

TEST(YakuTest, MenzenTsumoNotForRon) {
    auto decomp = make_decomp(tile_type::M1, {
        {MentsuKind::Shuntsu, tile_type::M2, false},
        {MentsuKind::Shuntsu, tile_type::P1, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::M5, false},
    });
    auto ctx = make_basic_ctx(tile_type::M5, false, true);  // ロン
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Shanpon, ctx);
    for (const auto& y : eval.yakus) {
        EXPECT_NE(y.type, YakuType::MenzenTsumo);
    }
}

TEST(YakuTest, Riichi) {
    auto decomp = make_decomp(tile_type::M1, {
        {MentsuKind::Shuntsu, tile_type::M2, false},
        {MentsuKind::Shuntsu, tile_type::P1, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::M5, false},
    });
    auto ctx = make_basic_ctx(tile_type::M5, false, true);
    ctx.is_riichi = true;
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Shanpon, ctx);
    bool found = false;
    for (const auto& y : eval.yakus) {
        if (y.type == YakuType::Riichi) { found = true; break; }
    }
    EXPECT_TRUE(found);
}

TEST(YakuTest, Ippatsu) {
    auto decomp = make_decomp(tile_type::M1, {
        {MentsuKind::Shuntsu, tile_type::M2, false},
        {MentsuKind::Shuntsu, tile_type::P1, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::M5, false},
    });
    auto ctx = make_basic_ctx(tile_type::M5, false, true);
    ctx.is_riichi = true;
    ctx.is_ippatsu = true;
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Shanpon, ctx);
    bool found_riichi = false, found_ippatsu = false;
    for (const auto& y : eval.yakus) {
        if (y.type == YakuType::Riichi) found_riichi = true;
        if (y.type == YakuType::Ippatsu) found_ippatsu = true;
    }
    EXPECT_TRUE(found_riichi);
    EXPECT_TRUE(found_ippatsu);
}

TEST(YakuTest, IppatsuRequiresRiichi) {
    // is_riichi=false && is_ippatsu=true の不整合入力を検証
    auto decomp = make_decomp(tile_type::M1, {
        {MentsuKind::Koutsu, tile_type::HAKU, false},
        {MentsuKind::Shuntsu, tile_type::P1, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::M5, false},
    });
    auto ctx = make_basic_ctx(tile_type::M5, false, true);
    ctx.is_riichi = false;
    ctx.is_ippatsu = true;  // 不整合: 立直なしで一発
    ctx.jikaze = tile_type::NAN;

#ifdef NDEBUG
    // Release: assert 無効 → Ippatsu が付かないことを確認
    auto eval = yaku::evaluate_yaku(decomp, WaitType::Shanpon, ctx);
    for (const auto& y : eval.yakus) {
        EXPECT_NE(y.type, YakuType::Ippatsu);
    }
#else
    // Debug: assert で異常終了することを確認
    EXPECT_DEATH(
        yaku::evaluate_yaku(decomp, WaitType::Shanpon, ctx),
        "ippatsu without riichi is invalid"
    );
#endif
}

TEST(YakuTest, Tanyao) {
    // 断么九: 全て中張牌
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M2, false},
        {MentsuKind::Shuntsu, tile_type::P3, false},
        {MentsuKind::Shuntsu, tile_type::S4, false},
        {MentsuKind::Koutsu, tile_type::P6, false},
    });
    auto ctx = make_basic_ctx(tile_type::P6, true, true);
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Shanpon, ctx);
    bool found = false;
    for (const auto& y : eval.yakus) {
        if (y.type == YakuType::Tanyao) { found = true; break; }
    }
    EXPECT_TRUE(found);
}

TEST(YakuTest, TanyaoFailsWithYaochu) {
    // 么九牌を含む → 断么九不成立
    auto decomp = make_decomp(tile_type::M1, {
        {MentsuKind::Shuntsu, tile_type::M2, false},
        {MentsuKind::Shuntsu, tile_type::P3, false},
        {MentsuKind::Shuntsu, tile_type::S4, false},
        {MentsuKind::Koutsu, tile_type::P6, false},
    });
    auto ctx = make_basic_ctx(tile_type::P6, true, true);
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Shanpon, ctx);
    for (const auto& y : eval.yakus) {
        EXPECT_NE(y.type, YakuType::Tanyao);
    }
}

TEST(YakuTest, YakuhaiSangenpai) {
    // 白の刻子
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M2, false},
        {MentsuKind::Shuntsu, tile_type::P3, false},
        {MentsuKind::Shuntsu, tile_type::S4, false},
        {MentsuKind::Koutsu, tile_type::HAKU, false},
    });
    auto ctx = make_basic_ctx(tile_type::M5, false, true);
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Tanki, ctx);
    int yakuhai_count = 0;
    for (const auto& y : eval.yakus) {
        if (y.type == YakuType::Yakuhai) yakuhai_count++;
    }
    EXPECT_EQ(yakuhai_count, 1);
}

TEST(YakuTest, YakuhaiBakazeAndJikaze) {
    // 東が場風かつ自風 → 2翻
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M2, false},
        {MentsuKind::Shuntsu, tile_type::P3, false},
        {MentsuKind::Shuntsu, tile_type::S4, false},
        {MentsuKind::Koutsu, tile_type::TON, false},
    });
    auto ctx = make_basic_ctx(tile_type::M5, false, true);
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::TON;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Tanki, ctx);
    int yakuhai_count = 0;
    for (const auto& y : eval.yakus) {
        if (y.type == YakuType::Yakuhai) yakuhai_count++;
    }
    EXPECT_EQ(yakuhai_count, 2);  // 場風+自風
}

TEST(YakuTest, Pinfu) {
    // 全順子 + 非役牌雀頭 + 両面待ち
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    auto ctx = make_basic_ctx(tile_type::S8, true, true);
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Ryanmen, ctx);
    bool found = false;
    for (const auto& y : eval.yakus) {
        if (y.type == YakuType::Pinfu) { found = true; break; }
    }
    EXPECT_TRUE(found);
}

TEST(YakuTest, PinfuFailsWithKoutsu) {
    // 刻子あり → 平和不成立
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::S6, false},
    });
    auto ctx = make_basic_ctx(tile_type::S6, true, true);
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Shanpon, ctx);
    for (const auto& y : eval.yakus) {
        EXPECT_NE(y.type, YakuType::Pinfu);
    }
}

TEST(YakuTest, PinfuFailsWithYakuhaiJantai) {
    // 役牌雀頭 → 平和不成立
    auto decomp = make_decomp(tile_type::HAKU, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    auto ctx = make_basic_ctx(tile_type::S8, true, true);
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Ryanmen, ctx);
    for (const auto& y : eval.yakus) {
        EXPECT_NE(y.type, YakuType::Pinfu);
    }
}

TEST(YakuTest, Iipeiko) {
    // 同一順子2組: 1m2m3m × 2
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Koutsu, tile_type::S5, false},
    });
    auto ctx = make_basic_ctx(tile_type::S5, true, true);
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Shanpon, ctx);
    bool found = false;
    for (const auto& y : eval.yakus) {
        if (y.type == YakuType::Iipeiko) { found = true; break; }
    }
    EXPECT_TRUE(found);
}

TEST(YakuTest, IipeikoRequiresMenzen) {
    // 門前でない → 一盃口不成立
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, true},  // 副露
        {MentsuKind::Koutsu, tile_type::S5, false},
    });
    auto ctx = make_basic_ctx(tile_type::S5, true, false);  // 非門前
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Shanpon, ctx);
    for (const auto& y : eval.yakus) {
        EXPECT_NE(y.type, YakuType::Iipeiko);
    }
}

TEST(YakuTest, HaiteiTsumo) {
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Koutsu, tile_type::HAKU, false},
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
    });
    auto ctx = make_basic_ctx(tile_type::M5, true, true);
    ctx.is_haitei = true;
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Tanki, ctx);
    bool found = false;
    for (const auto& y : eval.yakus) {
        if (y.type == YakuType::HaiteiTsumo) { found = true; break; }
    }
    EXPECT_TRUE(found);
}

TEST(YakuTest, HouteiRon) {
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Koutsu, tile_type::HAKU, false},
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
    });
    auto ctx = make_basic_ctx(tile_type::M5, false, true);
    ctx.is_houtei = true;
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Tanki, ctx);
    bool found = false;
    for (const auto& y : eval.yakus) {
        if (y.type == YakuType::HouteiRon) { found = true; break; }
    }
    EXPECT_TRUE(found);
}

TEST(YakuTest, RinshanKaihou) {
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Koutsu, tile_type::HAKU, false},
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Kantsu, tile_type::S5, false},
    });
    auto ctx = make_basic_ctx(tile_type::M5, true, true);
    ctx.is_rinshan = true;
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Tanki, ctx);
    bool found = false;
    for (const auto& y : eval.yakus) {
        if (y.type == YakuType::RinshanKaihou) { found = true; break; }
    }
    EXPECT_TRUE(found);
}

TEST(YakuTest, Chankan) {
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Koutsu, tile_type::HAKU, false},
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
    });
    auto ctx = make_basic_ctx(tile_type::M5, false, true);
    ctx.is_chankan = true;
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Tanki, ctx);
    bool found = false;
    for (const auto& y : eval.yakus) {
        if (y.type == YakuType::Chankan) { found = true; break; }
    }
    EXPECT_TRUE(found);
}

TEST(YakuTest, NoYakuReturnsEmpty) {
    // 役なし: 門前ロン、全順子だが平和でない（嵌張待ち）、么九牌あり
    auto decomp = make_decomp(tile_type::M1, {
        {MentsuKind::Shuntsu, tile_type::M3, false},
        {MentsuKind::Shuntsu, tile_type::P1, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Shuntsu, tile_type::S6, false},
    });
    auto ctx = make_basic_ctx(tile_type::M4, false, true);
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Kanchan, ctx);
    EXPECT_TRUE(eval.yakus.empty());
    EXPECT_EQ(eval.total_han, 0);
}

TEST(YakuTest, DoraCount) {
    // ドラ表示牌が M4 → ドラは M5
    std::vector<TileId> tile_ids = {16, 17, 18, 19};  // M5 の4枚
    std::vector<TileType> indicators = {tile_type::M4};
    EXPECT_EQ(yaku::count_dora(tile_ids, indicators), 4);
}

TEST(YakuTest, AkadoraCount) {
    // 赤牌は id=16(赤5m), 52(赤5p), 88(赤5s)
    std::vector<TileId> tile_ids = {16, 52, 88, 0, 4};
    EXPECT_EQ(yaku::count_akadora(tile_ids), 3);
}

TEST(YakuTest, UradoraOnlyWithRiichi) {
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Koutsu, tile_type::HAKU, false},
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P2, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
    });
    auto ctx = make_basic_ctx(tile_type::M5, false, true);
    ctx.is_riichi = false;
    ctx.uradora_indicators = {tile_type::M4};  // 裏ドラ表示牌
    ctx.all_tile_ids = {16};  // M5 赤
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Tanki, ctx);
    // 立直なし → 裏ドラなし
    EXPECT_EQ(eval.uradora_count, 0);
}

TEST(YakuTest, MultipleYakuCombined) {
    // 立直 + 一発 + 門前ツモ + 断么九 + 平和 = 5翻
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M2, false},
        {MentsuKind::Shuntsu, tile_type::P3, false},
        {MentsuKind::Shuntsu, tile_type::S4, false},
        {MentsuKind::Shuntsu, tile_type::S5, false},
    });
    auto ctx = make_basic_ctx(tile_type::S7, true, true);
    ctx.is_riichi = true;
    ctx.is_ippatsu = true;
    ctx.bakaze = tile_type::TON;
    ctx.jikaze = tile_type::NAN;

    auto eval = yaku::evaluate_yaku(decomp, WaitType::Ryanmen, ctx);
    EXPECT_GE(eval.total_han, 5);
}

// 待ち判定テスト
TEST(WaitTypeTest, RyanmenWait) {
    // 5m6m+7m → 両面
    auto decomp = make_decomp(tile_type::M1, {
        {MentsuKind::Shuntsu, tile_type::M5, false},
        {MentsuKind::Shuntsu, tile_type::P1, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::HAKU, false},
    });
    auto waits = yaku::get_possible_waits(decomp, tile_type::M7);
    bool has_ryanmen = false;
    for (auto w : waits) {
        if (w == WaitType::Ryanmen) has_ryanmen = true;
    }
    EXPECT_TRUE(has_ryanmen);
}

TEST(WaitTypeTest, KanchanWait) {
    // 1m3m+2m → 嵌張
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P1, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::HAKU, false},
    });
    auto waits = yaku::get_possible_waits(decomp, tile_type::M2);
    bool has_kanchan = false;
    for (auto w : waits) {
        if (w == WaitType::Kanchan) has_kanchan = true;
    }
    EXPECT_TRUE(has_kanchan);
}

TEST(WaitTypeTest, PenchanWait) {
    // 1m2m+3m → 辺張
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P1, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::HAKU, false},
    });
    auto waits = yaku::get_possible_waits(decomp, tile_type::M3);
    bool has_penchan = false;
    for (auto w : waits) {
        if (w == WaitType::Penchan) has_penchan = true;
    }
    EXPECT_TRUE(has_penchan);
}

TEST(WaitTypeTest, TankiWait) {
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P1, false},
        {MentsuKind::Shuntsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::HAKU, false},
    });
    auto waits = yaku::get_possible_waits(decomp, tile_type::M5);
    bool has_tanki = false;
    for (auto w : waits) {
        if (w == WaitType::Tanki) has_tanki = true;
    }
    EXPECT_TRUE(has_tanki);
}

TEST(WaitTypeTest, ShanponWait) {
    // 刻子+和了牌 → 双碰
    auto decomp = make_decomp(tile_type::M5, {
        {MentsuKind::Shuntsu, tile_type::M1, false},
        {MentsuKind::Shuntsu, tile_type::P1, false},
        {MentsuKind::Koutsu, tile_type::S3, false},
        {MentsuKind::Koutsu, tile_type::HAKU, false},
    });
    auto waits = yaku::get_possible_waits(decomp, tile_type::HAKU);
    bool has_shanpon = false;
    for (auto w : waits) {
        if (w == WaitType::Shanpon) has_shanpon = true;
    }
    EXPECT_TRUE(has_shanpon);
}

TEST(YakuTest, ToString) {
    EXPECT_EQ(yaku::to_string(YakuType::Riichi), "立直");
    EXPECT_EQ(yaku::to_string(YakuType::Tanyao), "断么九");
    EXPECT_EQ(yaku::to_string(YakuType::Pinfu), "平和");
}
