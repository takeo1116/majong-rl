#include <gtest/gtest.h>
#include "core/player_state.h"
#include "core/round_state.h"
#include "core/match_state.h"

using namespace mahjong;

// PlayerState の初期化テスト
TEST(PlayerStateTest, Reset) {
    PlayerState ps;
    ps.reset(Wind::East);

    EXPECT_TRUE(ps.hand.empty());
    EXPECT_TRUE(ps.melds.empty());
    EXPECT_TRUE(ps.discards.empty());
    EXPECT_EQ(ps.score, 25000);
    EXPECT_FALSE(ps.is_riichi);
    EXPECT_TRUE(ps.is_menzen);
    EXPECT_EQ(ps.jikaze, Wind::East);
    EXPECT_FALSE(ps.is_furiten);
}

// PlayerState の値コピー
TEST(PlayerStateTest, ValueCopy) {
    PlayerState ps1;
    ps1.reset(Wind::South, 30000);
    ps1.hand = {0, 4, 8};
    ps1.is_riichi = true;

    PlayerState ps2 = ps1;
    EXPECT_EQ(ps2.score, 30000);
    EXPECT_EQ(ps2.hand.size(), 3);
    EXPECT_TRUE(ps2.is_riichi);

    // コピー後の変更が元に影響しない
    ps2.score = 20000;
    ps2.hand.push_back(12);
    EXPECT_EQ(ps1.score, 30000);
    EXPECT_EQ(ps1.hand.size(), 3);
}

// RoundState の初期化テスト
TEST(RoundStateTest, Reset) {
    RoundState rs;
    rs.reset(0, 0, 0, 0);

    EXPECT_EQ(rs.round_number, 0);
    EXPECT_EQ(rs.dealer, 0);
    EXPECT_EQ(rs.current_player, 0);
    EXPECT_EQ(rs.phase, Phase::StartRound);
    EXPECT_EQ(rs.end_reason, RoundEndReason::None);
    EXPECT_FALSE(rs.is_round_over());

    // 自風の設定確認（親が0の場合）
    EXPECT_EQ(rs.players[0].jikaze, Wind::East);
    EXPECT_EQ(rs.players[1].jikaze, Wind::South);
    EXPECT_EQ(rs.players[2].jikaze, Wind::West);
    EXPECT_EQ(rs.players[3].jikaze, Wind::North);
}

// 親が1の場合の自風設定
TEST(RoundStateTest, ResetWithDealer1) {
    RoundState rs;
    rs.reset(1, 1, 0, 0);

    EXPECT_EQ(rs.players[0].jikaze, Wind::North);
    EXPECT_EQ(rs.players[1].jikaze, Wind::East);
    EXPECT_EQ(rs.players[2].jikaze, Wind::South);
    EXPECT_EQ(rs.players[3].jikaze, Wind::West);
}

// RoundState の値コピー
TEST(RoundStateTest, ValueCopy) {
    RoundState rs1;
    rs1.reset(3, 2, 1, 2);

    RoundState rs2 = rs1;
    EXPECT_EQ(rs2.round_number, 3);
    EXPECT_EQ(rs2.dealer, 2);
    EXPECT_EQ(rs2.honba, 1);
    EXPECT_EQ(rs2.kyotaku, 2);

    // コピー後の変更が元に影響しない
    rs2.turn_number = 5;
    EXPECT_EQ(rs1.turn_number, 0);
}

// MatchState の初期化テスト
TEST(MatchStateTest, Reset) {
    MatchState ms;
    ms.reset(0);

    EXPECT_EQ(ms.round_number, 0);
    EXPECT_EQ(ms.first_dealer, 0);
    EXPECT_EQ(ms.current_dealer, 0);
    EXPECT_EQ(ms.honba, 0);
    EXPECT_EQ(ms.kyotaku, 0);
    EXPECT_FALSE(ms.is_match_over);
    EXPECT_FALSE(ms.is_extra_round);

    for (int i = 0; i < kNumPlayers; ++i) {
        EXPECT_EQ(ms.scores[i], 25000);
    }
}

// 場風の判定
TEST(MatchStateTest, Bakaze) {
    MatchState ms;
    ms.reset(0);

    ms.round_number = 0;
    EXPECT_EQ(ms.bakaze(), Wind::East);

    ms.round_number = 3;
    EXPECT_EQ(ms.bakaze(), Wind::East);

    ms.round_number = 4;
    EXPECT_EQ(ms.bakaze(), Wind::South);

    ms.round_number = 7;
    EXPECT_EQ(ms.bakaze(), Wind::South);

    ms.round_number = 8;  // 延長局
    EXPECT_EQ(ms.bakaze(), Wind::South);
}

// オーラス判定
TEST(MatchStateTest, IsOorasu) {
    MatchState ms;
    ms.reset(0);

    ms.round_number = 6;
    EXPECT_FALSE(ms.is_oorasu());

    ms.round_number = 7;
    EXPECT_TRUE(ms.is_oorasu());

    ms.round_number = 8;
    EXPECT_TRUE(ms.is_oorasu());
}

// 順位計算
TEST(MatchStateTest, ComputeRanking) {
    MatchState ms;
    ms.reset(0);

    ms.scores = {30000, 25000, 20000, 25000};
    ms.compute_ranking();

    // 1位: player0 (30000)
    // 2位: player1 (25000, 起家に近い)
    // 3位: player3 (25000)
    // 4位: player2 (20000)
    EXPECT_EQ(ms.final_ranking[0], 0);  // 1位
    EXPECT_EQ(ms.final_ranking[1], 1);  // 2位
    EXPECT_EQ(ms.final_ranking[3], 2);  // 3位
    EXPECT_EQ(ms.final_ranking[2], 3);  // 4位
}

// 同点の上家取り
TEST(MatchStateTest, ComputeRankingTiebreak) {
    MatchState ms;
    ms.reset(0);

    ms.scores = {25000, 25000, 25000, 25000};
    ms.compute_ranking();

    // 全員同点 → 起家に近い順
    EXPECT_EQ(ms.final_ranking[0], 0);
    EXPECT_EQ(ms.final_ranking[1], 1);
    EXPECT_EQ(ms.final_ranking[2], 2);
    EXPECT_EQ(ms.final_ranking[3], 3);
}

// 値コピー
TEST(MatchStateTest, ValueCopy) {
    MatchState ms1;
    ms1.reset(1);
    ms1.scores = {30000, 20000, 25000, 25000};
    ms1.round_number = 5;

    MatchState ms2 = ms1;
    EXPECT_EQ(ms2.round_number, 5);
    EXPECT_EQ(ms2.first_dealer, 1);
    EXPECT_EQ(ms2.scores[0], 30000);

    ms2.scores[0] = 10000;
    EXPECT_EQ(ms1.scores[0], 30000);
}
