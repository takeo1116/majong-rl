// pybind11 バインディング: C++ ゲームエンジンを Python から呼び出す
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// Python.h が定義する NAN マクロと tile_type::NAN の名前衝突を回避
#ifdef NAN
#undef NAN
#endif

#include "core/tile.h"
#include "core/types.h"
#include "core/action.h"
#include "core/event.h"
#include "core/meld.h"
#include "core/player_state.h"
#include "core/round_state.h"
#include "core/match_state.h"
#include "core/environment_state.h"
#include "engine/game_engine.h"
#include "engine/hand_utils.h"
#include "rl/observation.h"
#include "rl/reward_policy.h"

namespace py = pybind11;
using namespace mahjong;

PYBIND11_MODULE(_mahjong_core, m) {
    m.doc() = "麻雀ゲームエンジン Python バインディング";

    // --- 列挙型 ---
    py::enum_<Suit>(m, "Suit")
        .value("Man", Suit::Man)
        .value("Pin", Suit::Pin)
        .value("Sou", Suit::Sou)
        .value("Ji", Suit::Ji);

    py::enum_<Wind>(m, "Wind")
        .value("East", Wind::East)
        .value("South", Wind::South)
        .value("West", Wind::West)
        .value("North", Wind::North);

    py::enum_<ActionType>(m, "ActionType")
        .value("Discard", ActionType::Discard)
        .value("TsumoWin", ActionType::TsumoWin)
        .value("Ron", ActionType::Ron)
        .value("Chi", ActionType::Chi)
        .value("Pon", ActionType::Pon)
        .value("Daiminkan", ActionType::Daiminkan)
        .value("Kakan", ActionType::Kakan)
        .value("Ankan", ActionType::Ankan)
        .value("Skip", ActionType::Skip)
        .value("Kyuushu", ActionType::Kyuushu);

    py::enum_<MeldType>(m, "MeldType")
        .value("Chi", MeldType::Chi)
        .value("Pon", MeldType::Pon)
        .value("Daiminkan", MeldType::Daiminkan)
        .value("Kakan", MeldType::Kakan)
        .value("Ankan", MeldType::Ankan);

    py::enum_<Phase>(m, "Phase")
        .value("StartMatch", Phase::StartMatch)
        .value("StartRound", Phase::StartRound)
        .value("DrawPhase", Phase::DrawPhase)
        .value("SelfActionPhase", Phase::SelfActionPhase)
        .value("ResponsePhase", Phase::ResponsePhase)
        .value("ResolveResponsePhase", Phase::ResolveResponsePhase)
        .value("ResolveWinPhase", Phase::ResolveWinPhase)
        .value("ResolveDrawPhase", Phase::ResolveDrawPhase)
        .value("EndRound", Phase::EndRound)
        .value("EndMatch", Phase::EndMatch);

    py::enum_<ErrorCode>(m, "ErrorCode")
        .value("Ok", ErrorCode::Ok)
        .value("IllegalAction", ErrorCode::IllegalAction)
        .value("WrongPhase", ErrorCode::WrongPhase)
        .value("InvalidTile", ErrorCode::InvalidTile)
        .value("InvalidActor", ErrorCode::InvalidActor)
        .value("InconsistentState", ErrorCode::InconsistentState)
        .value("UnknownError", ErrorCode::UnknownError);

    py::enum_<EventType>(m, "EventType")
        .value("RoundStart", EventType::RoundStart)
        .value("Deal", EventType::Deal)
        .value("Draw", EventType::Draw)
        .value("Discard", EventType::Discard)
        .value("Riichi", EventType::Riichi)
        .value("Chi", EventType::Chi)
        .value("Pon", EventType::Pon)
        .value("Kan", EventType::Kan)
        .value("DoraReveal", EventType::DoraReveal)
        .value("Ron", EventType::Ron)
        .value("Tsumo", EventType::Tsumo)
        .value("AbortiveDraw", EventType::AbortiveDraw)
        .value("ExhaustiveDraw", EventType::ExhaustiveDraw)
        .value("RoundEnd", EventType::RoundEnd)
        .value("MatchEnd", EventType::MatchEnd);

    py::enum_<RunMode>(m, "RunMode")
        .value("Debug", RunMode::Debug)
        .value("Fast", RunMode::Fast);

    py::enum_<RoundEndReason>(m, "RoundEndReason")
        .value("NONE", RoundEndReason::None)
        .value("Tsumo", RoundEndReason::Tsumo)
        .value("Ron", RoundEndReason::Ron)
        .value("ExhaustiveDraw", RoundEndReason::ExhaustiveDraw)
        .value("AbortiveKyuushu", RoundEndReason::AbortiveKyuushu);

    py::enum_<RewardPolicyType>(m, "RewardPolicyType")
        .value("PointDelta", RewardPolicyType::PointDelta)
        .value("FinalRank", RewardPolicyType::FinalRank)
        .value("Combined", RewardPolicyType::Combined);

    // --- 基本構造体 ---
    py::class_<Tile>(m, "Tile")
        .def_readonly("id", &Tile::id)
        .def_readonly("type", &Tile::type)
        .def_readonly("is_red", &Tile::is_red)
        .def_static("from_id", &Tile::from_id)
        .def_static("suit_of", &Tile::suit_of)
        .def_static("number_of", &Tile::number_of)
        .def_static("is_jihai", &Tile::is_jihai)
        .def_static("is_yaochu", &Tile::is_yaochu)
        .def_static("is_sangenpai", &Tile::is_sangenpai)
        .def_static("is_kazehai", &Tile::is_kazehai)
        .def_static("is_red_id", &Tile::is_red_id)
        .def_static("next_dora", &Tile::next_dora)
        .def_static("type_to_string", &Tile::type_to_string)
        .def("to_string", &Tile::to_string)
        .def("__repr__", [](const Tile& t) { return t.to_string(); });

    py::class_<DiscardInfo>(m, "DiscardInfo")
        .def_readonly("tile", &DiscardInfo::tile)
        .def_readonly("riichi_discard", &DiscardInfo::riichi_discard)
        .def_readonly("called", &DiscardInfo::called);

    py::class_<Meld>(m, "Meld")
        .def_readonly("type", &Meld::type)
        .def_property_readonly("tiles", [](const Meld& m) {
            // std::array<TileId, 4> → Python list (tile_count 分のみ)
            std::vector<uint8_t> result;
            for (int i = 0; i < m.tile_count; ++i) {
                result.push_back(m.tiles[i]);
            }
            return result;
        })
        .def_readonly("tile_count", &Meld::tile_count)
        .def_readonly("from_player", &Meld::from_player)
        .def_readonly("called_tile", &Meld::called_tile)
        .def("base_type", &Meld::base_type)
        .def("to_string", &Meld::to_string)
        .def("__repr__", [](const Meld& m) { return m.to_string(); });

    py::class_<Action>(m, "Action")
        .def_readonly("type", &Action::type)
        .def_readonly("actor", &Action::actor)
        .def_readonly("tile", &Action::tile)
        .def_readonly("target_player", &Action::target_player)
        .def_readonly("meld_type", &Action::meld_type)
        .def_readonly("riichi", &Action::riichi)
        .def_property_readonly("consumed_tiles", [](const Action& a) {
            std::vector<uint8_t> result;
            for (auto t : a.consumed_tiles) {
                if (t != 255) result.push_back(t);
            }
            return result;
        })
        .def_static("make_discard", &Action::make_discard,
                     py::arg("actor"), py::arg("tile"), py::arg("riichi") = false)
        .def_static("make_tsumo_win", &Action::make_tsumo_win)
        .def_static("make_ron", &Action::make_ron)
        .def_static("make_chi", &Action::make_chi)
        .def_static("make_pon", &Action::make_pon)
        .def_static("make_daiminkan", &Action::make_daiminkan)
        .def_static("make_kakan", &Action::make_kakan)
        .def_static("make_ankan", &Action::make_ankan)
        .def_static("make_skip", &Action::make_skip)
        .def_static("make_kyuushu", &Action::make_kyuushu)
        .def("to_string", &Action::to_string)
        .def("__repr__", [](const Action& a) { return a.to_string(); })
        .def("__eq__", &Action::operator==);

    py::class_<Event>(m, "Event")
        .def_readonly("type", &Event::type)
        .def_readonly("actor", &Event::actor)
        .def_readonly("target", &Event::target)
        .def_readonly("tile", &Event::tile)
        .def_readonly("meld_type", &Event::meld_type)
        .def_readonly("riichi", &Event::riichi)
        .def_readonly("round_end_reason", &Event::round_end_reason)
        .def("to_string", &Event::to_string)
        .def("__repr__", [](const Event& e) { return e.to_string(); });

    py::class_<StepResult>(m, "StepResult")
        .def_readonly("error", &StepResult::error)
        .def_readonly("round_over", &StepResult::round_over)
        .def_readonly("match_over", &StepResult::match_over)
        .def_property_readonly("rewards", [](const StepResult& r) {
            return std::vector<float>(r.rewards.begin(), r.rewards.end());
        })
        .def_readonly("events", &StepResult::events);

    // --- 状態構造体 ---
    py::class_<PlayerState>(m, "PlayerState")
        .def_readonly("hand", &PlayerState::hand)
        .def_readonly("melds", &PlayerState::melds)
        .def_readonly("discards", &PlayerState::discards)
        .def_readonly("score", &PlayerState::score)
        .def_readonly("is_riichi", &PlayerState::is_riichi)
        .def_readonly("is_double_riichi", &PlayerState::is_double_riichi)
        .def_readonly("ippatsu", &PlayerState::ippatsu)
        .def_readonly("is_menzen", &PlayerState::is_menzen)
        .def_readonly("is_furiten", &PlayerState::is_furiten)
        .def_readonly("is_temporary_furiten", &PlayerState::is_temporary_furiten)
        .def_readonly("is_riichi_furiten", &PlayerState::is_riichi_furiten)
        .def_readonly("rinshan_draw", &PlayerState::rinshan_draw)
        .def_readonly("jikaze", &PlayerState::jikaze);

    py::class_<RewardPolicyConfig>(m, "RewardPolicyConfig")
        .def(py::init<>())
        .def_readwrite("type", &RewardPolicyConfig::type)
        .def_readwrite("point_delta_scale", &RewardPolicyConfig::point_delta_scale)
        .def_readwrite("rank_scale", &RewardPolicyConfig::rank_scale);

    py::class_<MatchState>(m, "MatchState")
        .def_readonly("round_number", &MatchState::round_number)
        .def_property_readonly("scores", [](const MatchState& ms) {
            return std::vector<int32_t>(ms.scores.begin(), ms.scores.end());
        })
        .def_readonly("first_dealer", &MatchState::first_dealer)
        .def_readonly("current_dealer", &MatchState::current_dealer)
        .def_readonly("honba", &MatchState::honba)
        .def_readonly("kyotaku", &MatchState::kyotaku)
        .def_readonly("is_extra_round", &MatchState::is_extra_round)
        .def_readonly("is_match_over", &MatchState::is_match_over)
        .def_property_readonly("final_ranking", [](const MatchState& ms) {
            return std::vector<uint8_t>(ms.final_ranking.begin(), ms.final_ranking.end());
        })
        .def("bakaze", &MatchState::bakaze)
        .def("is_oorasu", &MatchState::is_oorasu);

    py::class_<ResponseContext>(m, "ResponseContext")
        .def_readonly("discarder", &ResponseContext::discarder)
        .def_readonly("discard_tile", &ResponseContext::discard_tile)
        .def_readonly("active", &ResponseContext::active);

    py::class_<RoundState>(m, "RoundState")
        .def_readonly("round_number", &RoundState::round_number)
        .def_readonly("dealer", &RoundState::dealer)
        .def_readonly("current_player", &RoundState::current_player)
        .def_property_readonly("wall", [](const RoundState& rs) {
            return std::vector<uint8_t>(rs.wall.begin(), rs.wall.end());
        })
        .def_readonly("wall_position", &RoundState::wall_position)
        .def_readonly("dora_indicators", &RoundState::dora_indicators)
        .def_readonly("uradora_indicators", &RoundState::uradora_indicators)
        .def_property_readonly("players", [](const RoundState& rs) {
            return std::vector<PlayerState>(rs.players.begin(), rs.players.end());
        })
        .def_readonly("honba", &RoundState::honba)
        .def_readonly("kyotaku", &RoundState::kyotaku)
        .def_readonly("turn_number", &RoundState::turn_number)
        .def_readonly("last_discard", &RoundState::last_discard)
        .def_readonly("last_discarder", &RoundState::last_discarder)
        .def_readonly("end_reason", &RoundState::end_reason)
        .def_readonly("phase", &RoundState::phase)
        .def_readonly("response_context", &RoundState::response_context)
        .def_readonly("total_kan_count", &RoundState::total_kan_count)
        .def("is_round_over", &RoundState::is_round_over)
        .def("remaining_draws", &RoundState::remaining_draws);

    py::class_<EnvironmentState>(m, "EnvironmentState")
        .def(py::init<>())
        .def_readwrite("match_state", &EnvironmentState::match_state)
        .def_readwrite("round_state", &EnvironmentState::round_state)
        .def_readwrite("run_mode", &EnvironmentState::run_mode)
        .def_readwrite("logging_enabled", &EnvironmentState::logging_enabled)
        .def_readwrite("reward_policy_config", &EnvironmentState::reward_policy_config);

    // --- Observation ---
    py::class_<PartialObservation>(m, "PartialObservation")
        .def_readonly("observer", &PartialObservation::observer)
        .def_readonly("hand", &PartialObservation::hand)
        .def_readonly("melds", &PartialObservation::melds)
        .def_readonly("is_riichi", &PartialObservation::is_riichi)
        .def_readonly("is_menzen", &PartialObservation::is_menzen)
        .def_readonly("is_furiten", &PartialObservation::is_furiten)
        .def_readonly("is_temporary_furiten", &PartialObservation::is_temporary_furiten)
        .def_readonly("is_riichi_furiten", &PartialObservation::is_riichi_furiten)
        .def_readonly("discards", &PartialObservation::discards)
        .def_property_readonly("public_melds", [](const PartialObservation& obs) {
            std::vector<std::vector<Meld>> result;
            for (const auto& melds : obs.public_melds) {
                result.push_back(melds);
            }
            return result;
        })
        .def_property_readonly("scores", [](const PartialObservation& obs) {
            return std::vector<int32_t>(obs.scores.begin(), obs.scores.end());
        })
        .def_property_readonly("riichi_declared", [](const PartialObservation& obs) {
            return std::vector<bool>(obs.riichi_declared.begin(), obs.riichi_declared.end());
        })
        .def_readonly("round_number", &PartialObservation::round_number)
        .def_readonly("dealer", &PartialObservation::dealer)
        .def_readonly("bakaze", &PartialObservation::bakaze)
        .def_readonly("jikaze", &PartialObservation::jikaze)
        .def_readonly("honba", &PartialObservation::honba)
        .def_readonly("kyotaku", &PartialObservation::kyotaku)
        .def_readonly("turn_number", &PartialObservation::turn_number)
        .def_readonly("current_player", &PartialObservation::current_player)
        .def_readonly("phase", &PartialObservation::phase)
        .def_readonly("dora_indicators", &PartialObservation::dora_indicators);

    py::class_<FullObservation>(m, "FullObservation")
        .def_property_readonly("hands", [](const FullObservation& obs) {
            std::vector<std::vector<uint8_t>> result;
            for (const auto& hand : obs.hands) {
                result.push_back(hand);
            }
            return result;
        })
        .def_property_readonly("melds", [](const FullObservation& obs) {
            std::vector<std::vector<Meld>> result;
            for (const auto& melds : obs.melds) {
                result.push_back(melds);
            }
            return result;
        })
        .def_property_readonly("discards", [](const FullObservation& obs) {
            std::vector<std::vector<DiscardInfo>> result;
            for (const auto& discards : obs.discards) {
                result.push_back(discards);
            }
            return result;
        })
        .def_property_readonly("scores", [](const FullObservation& obs) {
            return std::vector<int32_t>(obs.scores.begin(), obs.scores.end());
        })
        .def_property_readonly("wall", [](const FullObservation& obs) {
            return std::vector<uint8_t>(obs.wall.begin(), obs.wall.end());
        })
        .def_readonly("wall_position", &FullObservation::wall_position)
        .def_readonly("dora_indicators", &FullObservation::dora_indicators)
        .def_readonly("uradora_indicators", &FullObservation::uradora_indicators)
        .def_readonly("round_number", &FullObservation::round_number)
        .def_readonly("dealer", &FullObservation::dealer)
        .def_readonly("current_player", &FullObservation::current_player)
        .def_readonly("phase", &FullObservation::phase)
        .def_readonly("honba", &FullObservation::honba)
        .def_readonly("kyotaku", &FullObservation::kyotaku)
        .def_readonly("turn_number", &FullObservation::turn_number)
        .def_readonly("end_reason", &FullObservation::end_reason)
        .def_readonly("match_state", &FullObservation::match_state);

    // --- GameEngine ---
    py::class_<GameEngine>(m, "GameEngine")
        .def(py::init<>())
        .def("reset_match",
             py::overload_cast<EnvironmentState&, uint64_t, RunMode>(
                 &GameEngine::reset_match),
             py::arg("env"), py::arg("seed"), py::arg("mode") = RunMode::Fast)
        .def("reset_match",
             py::overload_cast<EnvironmentState&, uint64_t, PlayerId, RunMode>(
                 &GameEngine::reset_match),
             py::arg("env"), py::arg("seed"), py::arg("first_dealer"),
             py::arg("mode") = RunMode::Fast)
        .def("step", &GameEngine::step)
        .def("get_legal_actions", &GameEngine::get_legal_actions)
        .def("advance_round", &GameEngine::advance_round);

    // --- Observation 生成関数 ---
    m.def("make_partial_observation", &make_partial_observation,
          py::arg("env"), py::arg("observer"),
          "部分観測を生成する");
    m.def("make_full_observation", &make_full_observation,
          py::arg("env"),
          "完全観測を生成する");

    // --- hand_utils ---
    m.def("make_type_counts", [](const std::vector<uint8_t>& hand) {
        return hand_utils::make_type_counts(hand);
    }, "手牌の TileId 列から TileType 別カウントを返す");
    m.def("is_agari", &hand_utils::is_agari, "和了形チェック");
    m.def("is_tenpai", &hand_utils::is_tenpai, "テンパイチェック");
    m.def("get_waits", &hand_utils::get_waits, "待ち牌一覧を返す");

    // --- 定数 ---
    m.attr("NUM_TILES") = kNumTiles;
    m.attr("NUM_TILE_TYPES") = kNumTileTypes;
    m.attr("NUM_PLAYERS") = kNumPlayers;
}
