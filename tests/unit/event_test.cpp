#include <gtest/gtest.h>
#include "core/event.h"

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
