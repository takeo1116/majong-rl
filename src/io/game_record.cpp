#include "io/game_record.h"

namespace mahjong {

void GameRecorder::on_match_start(uint64_t seed, PlayerId first_dealer,
                                  const std::array<int32_t, kNumPlayers>& scores) {
    if (!enabled) return;
    record = GameRecord{};
    record.seed = seed;
    record.first_dealer = first_dealer;
    record.final_scores = scores;
}

void GameRecorder::on_round_start(uint8_t round_number, PlayerId dealer,
                                  uint8_t honba, uint8_t kyotaku,
                                  const std::array<int32_t, kNumPlayers>& scores) {
    if (!enabled) return;
    RoundRecord rr;
    rr.round_number = round_number;
    rr.dealer = dealer;
    rr.honba = honba;
    rr.kyotaku = kyotaku;
    rr.start_scores = scores;
    record.rounds.push_back(std::move(rr));
}

void GameRecorder::on_action(const Action& action) {
    if (!enabled) return;
    if (record.rounds.empty()) return;
    record.rounds.back().actions.push_back(action);
}

void GameRecorder::on_events(const std::vector<Event>& events) {
    if (!enabled) return;
    if (record.rounds.empty()) return;
    auto& ev = record.rounds.back().events;
    ev.insert(ev.end(), events.begin(), events.end());
}

void GameRecorder::on_round_end(RoundEndReason reason,
                                const std::array<int32_t, kNumPlayers>& end_scores) {
    if (!enabled) return;
    if (record.rounds.empty()) return;
    record.rounds.back().end_reason = reason;
    record.rounds.back().end_scores = end_scores;
}

void GameRecorder::on_match_end(const std::array<int32_t, kNumPlayers>& final_scores,
                                const std::array<uint8_t, kNumPlayers>& final_ranking) {
    if (!enabled) return;
    record.final_scores = final_scores;
    record.final_ranking = final_ranking;
    record.is_complete = true;
}

}  // namespace mahjong
