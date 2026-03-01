#include "core/environment_state.h"

namespace mahjong {

void EnvironmentState::reset(uint64_t seed, RunMode mode) {
    rng.seed(seed);

    // 起家を乱数で決定（0-3）
    PlayerId dealer = static_cast<PlayerId>(rng.next_int(kNumPlayers));

    run_mode = mode;
    logging_enabled = (mode == RunMode::Debug);

    match_state.reset(dealer);
    round_state.reset(0, dealer, 0, 0);
}

void EnvironmentState::reset(uint64_t seed, PlayerId first_dealer, RunMode mode) {
    rng.seed(seed);

    run_mode = mode;
    logging_enabled = (mode == RunMode::Debug);

    match_state.reset(first_dealer);
    round_state.reset(0, first_dealer, 0, 0);
}

}  // namespace mahjong
