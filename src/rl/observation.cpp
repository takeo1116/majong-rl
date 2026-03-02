#include "rl/observation.h"

namespace mahjong {

PartialObservation make_partial_observation(const EnvironmentState& env, PlayerId observer) {
    const auto& rs = env.round_state;
    const auto& ps = rs.players[observer];
    PartialObservation obs;

    obs.observer = observer;

    // 自家情報
    obs.hand = ps.hand;
    obs.melds = ps.melds;
    obs.is_riichi = ps.is_riichi;
    obs.is_menzen = ps.is_menzen;
    obs.is_furiten = ps.is_furiten;
    obs.is_temporary_furiten = ps.is_temporary_furiten;
    obs.is_riichi_furiten = ps.is_riichi_furiten;

    // 全員の公開情報
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        obs.discards[p] = rs.players[p].discards;
        obs.public_melds[p] = rs.players[p].melds;
        obs.scores[p] = rs.players[p].score;
        obs.riichi_declared[p] = rs.players[p].is_riichi;
    }

    // 局情報
    obs.round_number = rs.round_number;
    obs.dealer = rs.dealer;
    obs.bakaze = env.match_state.bakaze();
    obs.jikaze = ps.jikaze;
    obs.honba = rs.honba;
    obs.kyotaku = rs.kyotaku;
    obs.turn_number = rs.turn_number;
    obs.current_player = rs.current_player;
    obs.phase = rs.phase;
    obs.dora_indicators = rs.dora_indicators;

    return obs;
}

FullObservation make_full_observation(const EnvironmentState& env) {
    const auto& rs = env.round_state;
    FullObservation obs;

    // 全プレイヤー情報
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        obs.hands[p] = rs.players[p].hand;
        obs.melds[p] = rs.players[p].melds;
        obs.discards[p] = rs.players[p].discards;
        obs.scores[p] = rs.players[p].score;
    }

    // 山・王牌
    obs.wall = rs.wall;
    obs.wall_position = rs.wall_position;
    obs.dora_indicators = rs.dora_indicators;
    obs.uradora_indicators = rs.uradora_indicators;

    // 局情報
    obs.round_number = rs.round_number;
    obs.dealer = rs.dealer;
    obs.current_player = rs.current_player;
    obs.phase = rs.phase;
    obs.honba = rs.honba;
    obs.kyotaku = rs.kyotaku;
    obs.turn_number = rs.turn_number;
    obs.end_reason = rs.end_reason;

    // 半荘情報
    obs.match_state = env.match_state;

    return obs;
}

PartialObservation make_observation(const EnvironmentState& env, PlayerId observer) {
    return make_partial_observation(env, observer);
}

FullObservation make_observation(const EnvironmentState& env, FullObservationTag) {
    return make_full_observation(env);
}

}  // namespace mahjong
