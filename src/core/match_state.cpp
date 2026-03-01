#include "core/match_state.h"
#include <algorithm>
#include <numeric>

namespace mahjong {

Wind MatchState::bakaze() const {
    // 局番号 0-3: 東場, 4-7: 南場, 8: 延長局（南場扱い）
    if (round_number < 4) return Wind::East;
    return Wind::South;
}

bool MatchState::is_oorasu() const {
    // 南4局（round_number == 7）がオーラス
    // 延長局（round_number == 8）もオーラス扱い
    return round_number >= 7;
}

void MatchState::reset(PlayerId dealer) {
    round_number = 0;
    scores = {25000, 25000, 25000, 25000};
    first_dealer = dealer;
    current_dealer = dealer;
    honba = 0;
    kyotaku = 0;
    is_extra_round = false;
    is_match_over = false;
    final_ranking = {0, 1, 2, 3};
}

void MatchState::compute_ranking() {
    // プレイヤーIDの配列を作る
    std::array<uint8_t, kNumPlayers> indices;
    std::iota(indices.begin(), indices.end(), 0);

    // 点数降順、同点なら起家に近い順（上家取り）
    std::sort(indices.begin(), indices.end(), [this](uint8_t a, uint8_t b) {
        if (scores[a] != scores[b]) {
            return scores[a] > scores[b];
        }
        // 起家からの距離が近い方が上位
        int dist_a = (a - first_dealer + kNumPlayers) % kNumPlayers;
        int dist_b = (b - first_dealer + kNumPlayers) % kNumPlayers;
        return dist_a < dist_b;
    });

    // indices[0] が1位, indices[1] が2位, ...
    // final_ranking[player_id] = 順位(0-based) に変換
    for (int rank = 0; rank < kNumPlayers; ++rank) {
        final_ranking[indices[rank]] = static_cast<uint8_t>(rank);
    }
}

}  // namespace mahjong
