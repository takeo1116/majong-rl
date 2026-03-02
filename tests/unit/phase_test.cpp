#include <gtest/gtest.h>
#include "engine/game_engine.h"

using namespace mahjong;

// フェーズ遷移の検証
TEST(PhaseTest, ValidTransitions) {
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::StartMatch, Phase::StartRound));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::StartRound, Phase::DrawPhase));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::StartRound, Phase::SelfActionPhase));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::DrawPhase, Phase::SelfActionPhase));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::DrawPhase, Phase::ResolveDrawPhase));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::SelfActionPhase, Phase::ResponsePhase));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::SelfActionPhase, Phase::DrawPhase));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::SelfActionPhase, Phase::ResolveWinPhase));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::SelfActionPhase, Phase::ResolveDrawPhase));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::SelfActionPhase, Phase::EndRound));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::ResponsePhase, Phase::ResolveResponsePhase));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::ResolveResponsePhase, Phase::DrawPhase));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::ResolveResponsePhase, Phase::SelfActionPhase));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::ResolveResponsePhase, Phase::ResolveWinPhase));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::ResolveResponsePhase, Phase::EndRound));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::ResolveWinPhase, Phase::EndRound));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::ResolveDrawPhase, Phase::EndRound));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::EndRound, Phase::StartRound));
    EXPECT_TRUE(GameEngine::is_valid_transition(Phase::EndRound, Phase::EndMatch));
}

TEST(PhaseTest, InvalidTransitions) {
    EXPECT_FALSE(GameEngine::is_valid_transition(Phase::StartMatch, Phase::DrawPhase));
    EXPECT_FALSE(GameEngine::is_valid_transition(Phase::EndMatch, Phase::StartRound));
    EXPECT_FALSE(GameEngine::is_valid_transition(Phase::SelfActionPhase, Phase::StartMatch));
    EXPECT_FALSE(GameEngine::is_valid_transition(Phase::ResponsePhase, Phase::SelfActionPhase));
    EXPECT_FALSE(GameEngine::is_valid_transition(Phase::DrawPhase, Phase::ResponsePhase));
}

// フェーズごとの許可アクション
TEST(PhaseTest, AllowedActionTypes) {
    auto self_actions = GameEngine::allowed_action_types(Phase::SelfActionPhase);
    EXPECT_FALSE(self_actions.empty());
    auto has = [&](ActionType t) {
        return std::find(self_actions.begin(), self_actions.end(), t) != self_actions.end();
    };
    EXPECT_TRUE(has(ActionType::Discard));
    EXPECT_TRUE(has(ActionType::TsumoWin));
    EXPECT_TRUE(has(ActionType::Ankan));
    EXPECT_TRUE(has(ActionType::Kakan));
    EXPECT_TRUE(has(ActionType::Kyuushu));
    EXPECT_FALSE(has(ActionType::Ron));

    auto resp_actions = GameEngine::allowed_action_types(Phase::ResponsePhase);
    auto has_r = [&](ActionType t) {
        return std::find(resp_actions.begin(), resp_actions.end(), t) != resp_actions.end();
    };
    EXPECT_TRUE(has_r(ActionType::Ron));
    EXPECT_TRUE(has_r(ActionType::Pon));
    EXPECT_TRUE(has_r(ActionType::Chi));
    EXPECT_TRUE(has_r(ActionType::Skip));
    EXPECT_FALSE(has_r(ActionType::Discard));

    // アクション不可のフェーズ
    auto draw_actions = GameEngine::allowed_action_types(Phase::DrawPhase);
    EXPECT_TRUE(draw_actions.empty());
}
