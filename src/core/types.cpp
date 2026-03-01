#include "core/types.h"

namespace mahjong {

std::string to_string(ActionType type) {
    switch (type) {
        case ActionType::Discard:   return "Discard";
        case ActionType::TsumoWin:  return "TsumoWin";
        case ActionType::Ron:       return "Ron";
        case ActionType::Chi:       return "Chi";
        case ActionType::Pon:       return "Pon";
        case ActionType::Daiminkan: return "Daiminkan";
        case ActionType::Kakan:     return "Kakan";
        case ActionType::Ankan:     return "Ankan";
        case ActionType::Skip:      return "Skip";
        case ActionType::Kyuushu:   return "Kyuushu";
    }
    return "Unknown";
}

std::string to_string(MeldType type) {
    switch (type) {
        case MeldType::Chi:       return "Chi";
        case MeldType::Pon:       return "Pon";
        case MeldType::Daiminkan: return "Daiminkan";
        case MeldType::Kakan:     return "Kakan";
        case MeldType::Ankan:     return "Ankan";
    }
    return "Unknown";
}

std::string to_string(Phase phase) {
    switch (phase) {
        case Phase::StartMatch:           return "StartMatch";
        case Phase::StartRound:           return "StartRound";
        case Phase::DrawPhase:            return "DrawPhase";
        case Phase::SelfActionPhase:      return "SelfActionPhase";
        case Phase::ResponsePhase:        return "ResponsePhase";
        case Phase::ResolveResponsePhase: return "ResolveResponsePhase";
        case Phase::ResolveWinPhase:      return "ResolveWinPhase";
        case Phase::ResolveDrawPhase:     return "ResolveDrawPhase";
        case Phase::EndRound:             return "EndRound";
        case Phase::EndMatch:             return "EndMatch";
    }
    return "Unknown";
}

std::string to_string(ErrorCode code) {
    switch (code) {
        case ErrorCode::Ok:                return "Ok";
        case ErrorCode::IllegalAction:     return "IllegalAction";
        case ErrorCode::WrongPhase:        return "WrongPhase";
        case ErrorCode::InvalidTile:       return "InvalidTile";
        case ErrorCode::InvalidActor:      return "InvalidActor";
        case ErrorCode::InconsistentState: return "InconsistentState";
        case ErrorCode::UnknownError:      return "UnknownError";
    }
    return "Unknown";
}

std::string to_string(EventType type) {
    switch (type) {
        case EventType::RoundStart:     return "RoundStart";
        case EventType::Deal:           return "Deal";
        case EventType::Draw:           return "Draw";
        case EventType::Discard:        return "Discard";
        case EventType::Riichi:         return "Riichi";
        case EventType::Chi:            return "Chi";
        case EventType::Pon:            return "Pon";
        case EventType::Kan:            return "Kan";
        case EventType::DoraReveal:     return "DoraReveal";
        case EventType::Ron:            return "Ron";
        case EventType::Tsumo:          return "Tsumo";
        case EventType::AbortiveDraw:   return "AbortiveDraw";
        case EventType::ExhaustiveDraw: return "ExhaustiveDraw";
        case EventType::RoundEnd:       return "RoundEnd";
        case EventType::MatchEnd:       return "MatchEnd";
    }
    return "Unknown";
}

std::string to_string(RunMode mode) {
    switch (mode) {
        case RunMode::Debug: return "Debug";
        case RunMode::Fast:  return "Fast";
    }
    return "Unknown";
}

}  // namespace mahjong
