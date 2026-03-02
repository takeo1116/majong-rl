#include "io/display.h"
#include <algorithm>
#include <sstream>

namespace mahjong {
namespace display {

std::string round_name(uint8_t round_number) {
    if (round_number >= 8) return "延長局";
    // 0-3: 東1〜東4局, 4-7: 南1〜南4局
    const char* wind = (round_number < 4) ? "東" : "南";
    int num = (round_number % 4) + 1;
    return std::string(wind) + std::to_string(num) + "局";
}

std::string round_header(uint8_t round_number, uint8_t honba, uint8_t kyotaku, PlayerId dealer) {
    std::ostringstream oss;
    oss << "=== " << round_name(round_number)
        << " " << static_cast<int>(honba) << "本場"
        << " 供託" << static_cast<int>(kyotaku) << "本"
        << " 親:P" << static_cast<int>(dealer)
        << " ===";
    return oss.str();
}

std::string scores_to_string(const std::array<int32_t, kNumPlayers>& scores) {
    std::ostringstream oss;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p > 0) oss << " ";
        oss << "P" << static_cast<int>(p) << ":" << scores[p];
    }
    return oss.str();
}

std::string hand_to_string(const std::vector<TileId>& hand) {
    // TileId でソートして表示
    auto sorted = hand;
    std::sort(sorted.begin(), sorted.end());
    std::ostringstream oss;
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (i > 0) oss << " ";
        oss << Tile::from_id(sorted[i]).to_string();
    }
    return oss.str();
}

std::string action_display(const Action& action) {
    std::ostringstream oss;
    oss << "[P" << static_cast<int>(action.actor) << "] ";
    switch (action.type) {
        case ActionType::Discard:
            oss << "打牌: " << Tile::from_id(action.tile).to_string();
            if (action.riichi) oss << " (リーチ)";
            break;
        case ActionType::TsumoWin:
            oss << "ツモ和了";
            break;
        case ActionType::Ron:
            oss << "ロン";
            break;
        case ActionType::Chi:
            oss << "チー: " << Tile::from_id(action.tile).to_string();
            break;
        case ActionType::Pon:
            oss << "ポン: " << Tile::from_id(action.tile).to_string();
            break;
        case ActionType::Daiminkan:
            oss << "大明槓: " << Tile::from_id(action.tile).to_string();
            break;
        case ActionType::Kakan:
            oss << "加槓: " << Tile::from_id(action.tile).to_string();
            break;
        case ActionType::Ankan:
            oss << "暗槓: " << Tile::type_to_string(action.tile);
            break;
        case ActionType::Skip:
            oss << "スキップ";
            break;
        case ActionType::Kyuushu:
            oss << "九種九牌";
            break;
    }
    return oss.str();
}

std::string event_display(const Event& event) {
    std::ostringstream oss;
    switch (event.type) {
        case EventType::RoundStart:
            oss << "--- 局開始 ---";
            break;
        case EventType::Deal:
            oss << "[P" << static_cast<int>(event.actor) << "] 配牌";
            break;
        case EventType::Draw:
            oss << "[P" << static_cast<int>(event.actor) << "] ツモ: "
                << Tile::from_id(event.tile).to_string();
            break;
        case EventType::Discard:
            oss << "[P" << static_cast<int>(event.actor) << "] 捨牌: "
                << Tile::from_id(event.tile).to_string();
            if (event.riichi) oss << " (リーチ)";
            break;
        case EventType::Riichi:
            oss << "[P" << static_cast<int>(event.actor) << "] リーチ宣言";
            break;
        case EventType::Chi:
            oss << "[P" << static_cast<int>(event.actor) << "] チー: "
                << Tile::from_id(event.tile).to_string();
            break;
        case EventType::Pon:
            oss << "[P" << static_cast<int>(event.actor) << "] ポン: "
                << Tile::from_id(event.tile).to_string();
            break;
        case EventType::Kan:
            oss << "[P" << static_cast<int>(event.actor) << "] "
                << to_string(event.meld_type) << ": "
                << Tile::from_id(event.tile).to_string();
            break;
        case EventType::DoraReveal:
            oss << "ドラ表示: " << Tile::from_id(event.tile).to_string();
            break;
        case EventType::Ron:
            oss << "[P" << static_cast<int>(event.actor) << "] ロン"
                << " ← P" << static_cast<int>(event.target);
            break;
        case EventType::Tsumo:
            oss << "[P" << static_cast<int>(event.actor) << "] ツモ和了";
            break;
        case EventType::AbortiveDraw:
            oss << "途中流局";
            break;
        case EventType::ExhaustiveDraw:
            oss << "荒牌平局";
            break;
        case EventType::RoundEnd:
            oss << "--- 局終了 ---";
            break;
        case EventType::MatchEnd:
            oss << "=== 半荘終了 ===";
            break;
    }
    return oss.str();
}

std::string round_end_summary(const EnvironmentState& env) {
    const auto& rs = env.round_state;
    std::ostringstream oss;
    oss << "--- " << round_name(rs.round_number) << " 終了 ---\n";

    switch (rs.end_reason) {
        case RoundEndReason::Tsumo:
            oss << "結果: ツモ和了 (P" << static_cast<int>(rs.current_player) << ")\n";
            break;
        case RoundEndReason::Ron:
            oss << "結果: ロン和了\n";
            break;
        case RoundEndReason::ExhaustiveDraw:
            oss << "結果: 荒牌平局\n";
            break;
        case RoundEndReason::AbortiveKyuushu:
            oss << "結果: 九種九牌流局\n";
            break;
        default:
            break;
    }

    oss << "スコア: " << scores_to_string(env.match_state.scores);
    return oss.str();
}

std::string match_end_summary(const EnvironmentState& env) {
    const auto& ms = env.match_state;
    std::ostringstream oss;
    oss << "\n=============================\n";
    oss << "  半荘終了\n";
    oss << "=============================\n";
    oss << "最終スコア: " << scores_to_string(ms.scores) << "\n";
    oss << "順位: ";
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p > 0) oss << " ";
        oss << "P" << static_cast<int>(p) << "=" << static_cast<int>(ms.final_ranking[p] + 1) << "位";
    }
    oss << "\n=============================";
    return oss.str();
}

}  // namespace display
}  // namespace mahjong
