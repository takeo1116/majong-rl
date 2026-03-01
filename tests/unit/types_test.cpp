#include <gtest/gtest.h>
#include "core/types.h"

using namespace mahjong;

// 列挙型の文字列変換が正しく動作する
TEST(TypesTest, ActionTypeToString) {
    EXPECT_EQ(to_string(ActionType::Discard), "Discard");
    EXPECT_EQ(to_string(ActionType::TsumoWin), "TsumoWin");
    EXPECT_EQ(to_string(ActionType::Ron), "Ron");
    EXPECT_EQ(to_string(ActionType::Skip), "Skip");
    EXPECT_EQ(to_string(ActionType::Kyuushu), "Kyuushu");
}

TEST(TypesTest, MeldTypeToString) {
    EXPECT_EQ(to_string(MeldType::Chi), "Chi");
    EXPECT_EQ(to_string(MeldType::Pon), "Pon");
    EXPECT_EQ(to_string(MeldType::Ankan), "Ankan");
}

TEST(TypesTest, PhaseToString) {
    EXPECT_EQ(to_string(Phase::StartMatch), "StartMatch");
    EXPECT_EQ(to_string(Phase::SelfActionPhase), "SelfActionPhase");
    EXPECT_EQ(to_string(Phase::EndMatch), "EndMatch");
}

TEST(TypesTest, ErrorCodeToString) {
    EXPECT_EQ(to_string(ErrorCode::Ok), "Ok");
    EXPECT_EQ(to_string(ErrorCode::IllegalAction), "IllegalAction");
}

TEST(TypesTest, EventTypeToString) {
    EXPECT_EQ(to_string(EventType::RoundStart), "RoundStart");
    EXPECT_EQ(to_string(EventType::Discard), "Discard");
    EXPECT_EQ(to_string(EventType::MatchEnd), "MatchEnd");
}

TEST(TypesTest, RunModeToString) {
    EXPECT_EQ(to_string(RunMode::Debug), "Debug");
    EXPECT_EQ(to_string(RunMode::Fast), "Fast");
}
