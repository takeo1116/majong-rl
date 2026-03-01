#include "core/round_state.h"

namespace mahjong {

// 山のうちツモ可能な範囲: wall_position から (kNumTiles - 14 - rinshan_draw_count) 未満
// 王牌は末尾14枚だが、嶺上ツモで使った分は山側に開放されない
int RoundState::remaining_draws() const {
    int drawable_end = kNumTiles - 14;  // 王牌を除く
    return drawable_end - wall_position;
}

void RoundState::reset(uint8_t round_num, PlayerId dealer_id, uint8_t honba_count, uint8_t kyotaku_count) {
    round_number = round_num;
    dealer = dealer_id;
    current_player = dealer_id;
    wall.fill(0);
    wall_position = 0;
    dora_indicators.clear();
    uradora_indicators.clear();
    pending_kan_dora = false;
    honba = honba_count;
    kyotaku = kyotaku_count;
    turn_number = 0;
    last_discard = 255;
    last_discarder = 255;
    recent_events.clear();
    end_reason = RoundEndReason::None;
    phase = Phase::StartRound;
    response_context = ResponseContext{};
    total_kan_count = 0;
    rinshan_draw_count = 0;
    first_draw.fill(true);

    // 各プレイヤーを初期化
    for (PlayerId i = 0; i < kNumPlayers; ++i) {
        // 自風は親から反時計回りに 東→南→西→北
        Wind wind = static_cast<Wind>((i - dealer_id + kNumPlayers) % kNumPlayers);
        players[i].reset(wind);
    }
}

}  // namespace mahjong
