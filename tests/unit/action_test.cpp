#include <gtest/gtest.h>
#include "core/action.h"

using namespace mahjong;

TEST(ActionTest, MakeDiscard) {
    Action a = Action::make_discard(0, 4);
    EXPECT_EQ(a.type, ActionType::Discard);
    EXPECT_EQ(a.actor, 0);
    EXPECT_EQ(a.tile, 4);
    EXPECT_FALSE(a.riichi);
}

TEST(ActionTest, MakeRiichiDiscard) {
    Action a = Action::make_discard(1, 8, true);
    EXPECT_EQ(a.type, ActionType::Discard);
    EXPECT_EQ(a.actor, 1);
    EXPECT_TRUE(a.riichi);
}

TEST(ActionTest, MakeTsumoWin) {
    Action a = Action::make_tsumo_win(2);
    EXPECT_EQ(a.type, ActionType::TsumoWin);
    EXPECT_EQ(a.actor, 2);
}

TEST(ActionTest, MakeRon) {
    Action a = Action::make_ron(1, 0);
    EXPECT_EQ(a.type, ActionType::Ron);
    EXPECT_EQ(a.actor, 1);
    EXPECT_EQ(a.target_player, 0);
}

TEST(ActionTest, MakeChi) {
    Action a = Action::make_chi(1, 0, 4, 8);
    EXPECT_EQ(a.type, ActionType::Chi);
    EXPECT_EQ(a.tile, 0);
    EXPECT_EQ(a.consumed_tiles[0], 4);
    EXPECT_EQ(a.consumed_tiles[1], 8);
}

TEST(ActionTest, MakeSkip) {
    Action a = Action::make_skip(3);
    EXPECT_EQ(a.type, ActionType::Skip);
    EXPECT_EQ(a.actor, 3);
}

TEST(ActionTest, MakeKyuushu) {
    Action a = Action::make_kyuushu(0);
    EXPECT_EQ(a.type, ActionType::Kyuushu);
    EXPECT_EQ(a.actor, 0);
}

TEST(ActionTest, Equality) {
    Action a1 = Action::make_discard(0, 4);
    Action a2 = Action::make_discard(0, 4);
    Action a3 = Action::make_discard(0, 8);

    EXPECT_EQ(a1, a2);
    EXPECT_NE(a1, a3);
}

TEST(ActionTest, ToString) {
    Action a = Action::make_discard(0, 4, true);
    std::string s = a.to_string();
    EXPECT_NE(s.find("Discard"), std::string::npos);
    EXPECT_NE(s.find("riichi"), std::string::npos);
}
