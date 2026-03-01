#include "core/player_state.h"

namespace mahjong {

void PlayerState::reset(Wind wind, int32_t initial_score) {
    hand.clear();
    melds.clear();
    discards.clear();
    score = initial_score;
    is_riichi = false;
    is_double_riichi = false;
    ippatsu = false;
    is_menzen = true;
    jikaze = wind;
    is_furiten = false;
    is_temporary_furiten = false;
    is_riichi_furiten = false;
    rinshan_draw = false;
}

}  // namespace mahjong
