#include <gtest/gtest.h>
#include "rules/agari.h"
#include "engine/hand_utils.h"

using namespace mahjong;

// ヘルパー: TileType 列からカウント配列を作る
static std::array<int, kNumTileTypes> counts_from_types(std::initializer_list<TileType> types) {
    std::array<int, kNumTileTypes> counts{};
    for (auto t : types) counts[t]++;
    return counts;
}

TEST(AgariTest, BasicMentsuHand) {
    // 1m2m3m 4m5m6m 7m8m9m 1p1p1p 2p2p → 4面子1雀頭
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M2, tile_type::M3,
        tile_type::M4, tile_type::M5, tile_type::M6,
        tile_type::M7, tile_type::M8, tile_type::M9,
        tile_type::P1, tile_type::P1, tile_type::P1,
        tile_type::P2, tile_type::P2
    });
    auto decomps = agari::enumerate_decompositions(counts, {});
    EXPECT_GE(decomps.size(), 1u);
    // 構成を確認: 雀頭=P2, 面子4つ
    bool found = false;
    for (const auto& d : decomps) {
        if (d.jantai == tile_type::P2 && d.mentsu_list.size() == 4) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(AgariTest, NoAgariReturnsEmpty) {
    // 和了形でない手
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M2, tile_type::M3,
        tile_type::M4, tile_type::M5, tile_type::M6,
        tile_type::M7, tile_type::M8, tile_type::M9,
        tile_type::P1, tile_type::P2, tile_type::P3,
        tile_type::S1, tile_type::S3
    });
    auto decomps = agari::enumerate_decompositions(counts, {});
    EXPECT_TRUE(decomps.empty());
}

TEST(AgariTest, MultipleDecompositions) {
    // 1m2m3m 1m2m3m 1m2m3m 5m5m5m 9m9m
    // → 3順子+1刻子 or 3刻子+1順子 等、複数の構成
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M1, tile_type::M1,
        tile_type::M2, tile_type::M2, tile_type::M2,
        tile_type::M3, tile_type::M3, tile_type::M3,
        tile_type::M5, tile_type::M5, tile_type::M5,
        tile_type::M9, tile_type::M9
    });
    auto decomps = agari::enumerate_decompositions(counts, {});
    // 少なくとも2通りの構成がある
    EXPECT_GE(decomps.size(), 2u);
}

TEST(AgariTest, WithOpenMelds) {
    // 手牌: 1m2m3m 5m5m（雀頭+1面子分）
    // 副露: ポン(P1), チー(S1S2S3), ポン(TON)
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M2, tile_type::M3,
        tile_type::M5, tile_type::M5
    });
    std::vector<Meld> open_melds;
    open_melds.push_back(Meld::make_pon(36, 37, 38, 1));  // P1 ポン
    open_melds.push_back(Meld::make_chi(72, 76, 80, 0));  // S1S2S3 チー
    open_melds.push_back(Meld::make_pon(108, 109, 110, 2));  // 東 ポン

    auto decomps = agari::enumerate_decompositions(counts, open_melds);
    EXPECT_GE(decomps.size(), 1u);

    // 各構成は面子4つ（閉1+副露3）を持つ
    for (const auto& d : decomps) {
        EXPECT_EQ(d.mentsu_list.size(), 4u);
        int open_count = 0;
        for (const auto& m : d.mentsu_list) {
            if (m.is_open) open_count++;
        }
        EXPECT_EQ(open_count, 3);
    }
}

TEST(AgariTest, AllKoutsu) {
    // 1m1m1m 5m5m5m 9m9m9m 東東東 白白
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M1, tile_type::M1,
        tile_type::M5, tile_type::M5, tile_type::M5,
        tile_type::M9, tile_type::M9, tile_type::M9,
        tile_type::TON, tile_type::TON, tile_type::TON,
        tile_type::HAKU, tile_type::HAKU
    });
    auto decomps = agari::enumerate_decompositions(counts, {});
    EXPECT_EQ(decomps.size(), 1u);
    EXPECT_EQ(decomps[0].jantai, tile_type::HAKU);
    for (const auto& m : decomps[0].mentsu_list) {
        EXPECT_EQ(m.kind, MentsuKind::Koutsu);
    }
}

TEST(AgariTest, TankiWait) {
    // 1m2m3m 4m5m6m 7m8m9m 1p2p3p 白白
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M2, tile_type::M3,
        tile_type::M4, tile_type::M5, tile_type::M6,
        tile_type::M7, tile_type::M8, tile_type::M9,
        tile_type::P1, tile_type::P2, tile_type::P3,
        tile_type::HAKU, tile_type::HAKU
    });
    auto decomps = agari::enumerate_decompositions(counts, {});
    EXPECT_GE(decomps.size(), 1u);
}

TEST(AgariTest, WrongTotalReturnsEmpty) {
    // 合計が 3k+2 でない → 空
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M2, tile_type::M3
    });
    auto decomps = agari::enumerate_decompositions(counts, {});
    EXPECT_TRUE(decomps.empty());
}

TEST(AgariTest, PinfuHand) {
    // 2m3m4m 5p6p7p 3s4s5s 7s8s9s 1m1m
    auto counts = counts_from_types({
        tile_type::M1, tile_type::M1,
        tile_type::M2, tile_type::M3, tile_type::M4,
        tile_type::P5, tile_type::P6, tile_type::P7,
        tile_type::S3, tile_type::S4, tile_type::S5,
        tile_type::S7, tile_type::S8, tile_type::S9
    });
    auto decomps = agari::enumerate_decompositions(counts, {});
    EXPECT_GE(decomps.size(), 1u);
    // 全面子が順子の構成がある
    bool all_shuntsu = false;
    for (const auto& d : decomps) {
        bool ok = true;
        for (const auto& m : d.mentsu_list) {
            if (m.kind != MentsuKind::Shuntsu) { ok = false; break; }
        }
        if (ok) { all_shuntsu = true; break; }
    }
    EXPECT_TRUE(all_shuntsu);
}
