#include <gtest/gtest.h>
#include "core/event.h"
#include "io/display.h"

using namespace mahjong;

TEST(EventTest, MakeRoundStart) {
    Event e = Event::make_round_start();
    EXPECT_EQ(e.type, EventType::RoundStart);
    EXPECT_EQ(e.actor, 255);
}

TEST(EventTest, MakeDraw) {
    Event e = Event::make_draw(0, 4);
    EXPECT_EQ(e.type, EventType::Draw);
    EXPECT_EQ(e.actor, 0);
    EXPECT_EQ(e.tile, 4);
}

TEST(EventTest, MakeDiscard) {
    Event e = Event::make_discard(1, 8, true);
    EXPECT_EQ(e.type, EventType::Discard);
    EXPECT_EQ(e.actor, 1);
    EXPECT_EQ(e.tile, 8);
    EXPECT_TRUE(e.riichi);
}

TEST(EventTest, MakeRon) {
    Event e = Event::make_ron(2, 0);
    EXPECT_EQ(e.type, EventType::Ron);
    EXPECT_EQ(e.actor, 2);
    EXPECT_EQ(e.target, 0);
}

TEST(EventTest, MakeRoundEnd) {
    Event e = Event::make_round_end(RoundEndReason::Tsumo);
    EXPECT_EQ(e.type, EventType::RoundEnd);
    EXPECT_EQ(e.round_end_reason, RoundEndReason::Tsumo);
}

TEST(EventTest, MakeMatchEnd) {
    Event e = Event::make_match_end();
    EXPECT_EQ(e.type, EventType::MatchEnd);
}

TEST(EventTest, ToString) {
    Event e = Event::make_discard(0, 4, false);
    std::string s = e.to_string();
    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find("Discard"), std::string::npos);
}

// ============================
// CQ-0033: 槓イベントの構築・表示テスト
// ============================

TEST(EventTest, MakeKanAnkan) {
    // 暗槓: tile に TileId が格納される（例: TileId 0 = 1m の最初の牌）
    Event e = Event::make_kan(2, MeldType::Ankan, 0);
    EXPECT_EQ(e.type, EventType::Kan);
    EXPECT_EQ(e.actor, 2);
    EXPECT_EQ(e.meld_type, MeldType::Ankan);
    EXPECT_EQ(e.tile, 0);
}

TEST(EventTest, MakeKanKakan) {
    Event e = Event::make_kan(1, MeldType::Kakan, 20);
    EXPECT_EQ(e.type, EventType::Kan);
    EXPECT_EQ(e.actor, 1);
    EXPECT_EQ(e.meld_type, MeldType::Kakan);
    EXPECT_EQ(e.tile, 20);
}

TEST(EventTest, MakeKanDaiminkan) {
    Event e = Event::make_kan(3, MeldType::Daiminkan, 40);
    EXPECT_EQ(e.type, EventType::Kan);
    EXPECT_EQ(e.actor, 3);
    EXPECT_EQ(e.meld_type, MeldType::Daiminkan);
    EXPECT_EQ(e.tile, 40);
}

TEST(EventTest, KanEventDisplayDoesNotCrash) {
    // 各種槓イベントの display が正常な文字列を返すことを確認（CQ-0033 回帰テスト）
    Event ankan = Event::make_kan(0, MeldType::Ankan, 0);
    Event kakan = Event::make_kan(1, MeldType::Kakan, 20);
    Event daiminkan = Event::make_kan(2, MeldType::Daiminkan, 40);

    std::string s1 = display::event_display(ankan);
    std::string s2 = display::event_display(kakan);
    std::string s3 = display::event_display(daiminkan);

    EXPECT_FALSE(s1.empty());
    EXPECT_FALSE(s2.empty());
    EXPECT_FALSE(s3.empty());
    // 牌名が含まれていることを確認
    EXPECT_NE(s1.find("1m"), std::string::npos) << "暗槓表示に牌名がない: " << s1;
    EXPECT_NE(s3.find("2p"), std::string::npos) << "大明槓表示に牌名がない: " << s3;
}
