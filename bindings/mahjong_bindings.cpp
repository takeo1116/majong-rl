// pybind11 バインディング: C++ ゲームエンジンを Python から呼び出す
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <algorithm>
#include <stdexcept>

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
#include "rl/shanten.h"
#include "rules/yaku.h"
#include "rules/agari.h"
#include "rules/score_calculator.h"

namespace py = pybind11;

// CQ-0227: binding 用 WinContext 構築 helper
static mahjong::WinContext make_win_context_for_binding(
    const mahjong::RoundState& rs,
    mahjong::PlayerId winner,
    mahjong::TileType agari_tile,
    bool is_tsumo,
    bool is_chankan)
{
    using namespace mahjong;
    const auto& player = rs.players[winner];
    WinContext ctx{};
    ctx.agari_tile = agari_tile;
    ctx.is_tsumo = is_tsumo;
    ctx.is_menzen = player.is_menzen;
    ctx.is_riichi = player.is_riichi;
    ctx.is_ippatsu = player.ippatsu;
    ctx.is_rinshan = player.rinshan_draw;
    ctx.is_chankan = is_chankan;
    ctx.is_haitei = is_tsumo && rs.remaining_draws() == 0;
    ctx.is_houtei = !is_tsumo && rs.remaining_draws() == 0;
    Wind bakaze = (rs.round_number < 4) ? Wind::East : Wind::South;
    ctx.bakaze = static_cast<TileType>(27 + static_cast<int>(bakaze));
    ctx.jikaze = static_cast<TileType>(27 + static_cast<int>(player.jikaze));
    for (TileId t : player.hand) ctx.all_tile_ids.push_back(t);
    for (const auto& meld : player.melds) {
        for (int i = 0; i < meld.tile_count; ++i) {
            ctx.all_tile_ids.push_back(meld.tiles[i]);
        }
    }
    for (TileId ind : rs.dora_indicators) {
        ctx.dora_indicators.push_back(ind / 4);
    }
    if (player.is_riichi) {
        for (TileId ind : rs.uradora_indicators) {
            ctx.uradora_indicators.push_back(ind / 4);
        }
    }
    return ctx;
}
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
        .def_readonly("dora_indicators", &PartialObservation::dora_indicators)
        .def_readonly("remaining_draws", &PartialObservation::remaining_draws);

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
        .def_property_readonly("riichi_declared", [](const FullObservation& obs) {
            return std::vector<bool>(obs.riichi_declared.begin(), obs.riichi_declared.end());
        })
        .def_property_readonly("menzen_flags", [](const FullObservation& obs) {
            return std::vector<bool>(obs.menzen_flags.begin(), obs.menzen_flags.end());
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
        .def_readonly("remaining_draws", &FullObservation::remaining_draws)
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
    m.def("is_tenpai", &hand_utils::is_tenpai, "テンパイチェック (門前専用)");
    m.def("is_tenpai_with_melds", [](const std::vector<int>& counts_vec,
                                      const EnvironmentState& env, int player) {
        std::array<int, kNumTileTypes> counts;
        std::copy(counts_vec.begin(), counts_vec.end(), counts.begin());
        return hand_utils::is_tenpai_with_melds(counts, env.round_state.players[player].melds);
    }, "テンパイチェック (副露考慮)");
    m.def("get_waits", &hand_utils::get_waits, "待ち牌一覧を返す");

    // --- shanten ---
    m.def("compute_shanten", [](const std::vector<int>& counts_vec, int meld_count) {
        if (counts_vec.size() != 34) {
            throw std::invalid_argument("counts must have exactly 34 elements");
        }
        std::array<int, 34> counts;
        std::copy(counts_vec.begin(), counts_vec.end(), counts.begin());
        return compute_shanten(counts, meld_count);
    }, py::arg("counts"), py::arg("meld_count") = 0,
       "シャンテン数を計算する (meld_count > 0 で open hand 対応)");

    m.def("analyze_discards", [](const std::vector<int>& counts_vec,
                                  const std::vector<int>& mask_vec,
                                  int meld_count) {
        if (counts_vec.size() != 34 || mask_vec.size() != 34) {
            throw std::invalid_argument("counts and legal_mask must have exactly 34 elements");
        }
        std::array<int, 34> counts, mask;
        std::copy(counts_vec.begin(), counts_vec.end(), counts.begin());
        std::copy(mask_vec.begin(), mask_vec.end(), mask.begin());
        auto result = analyze_discards(counts, mask, meld_count);
        py::dict d;
        d["shanten_after"] = py::cast(std::vector<int>(result.shanten_after.begin(), result.shanten_after.end()));
        d["acceptance"] = py::cast(std::vector<int>(result.acceptance.begin(), result.acceptance.end()));
        d["ukeire_norm"] = py::cast(std::vector<float>(result.ukeire_norm.begin(), result.ukeire_norm.end()));
        d["shanten_sign"] = py::cast(std::vector<float>(result.shanten_sign.begin(), result.shanten_sign.end()));
        return d;
    }, py::arg("counts"), py::arg("legal_mask"), py::arg("meld_count") = 0,
       "打牌候補の一括分析 (shanten/acceptance/ukeire_norm/shanten_sign)");

    m.def("find_best_discard", [](const std::vector<int>& counts_vec,
                                   const std::vector<int>& mask_vec,
                                   int meld_count) {
        if (counts_vec.size() != 34 || mask_vec.size() != 34) {
            throw std::invalid_argument("counts and legal_mask must have exactly 34 elements");
        }
        std::array<int, 34> counts, mask;
        std::copy(counts_vec.begin(), counts_vec.end(), counts.begin());
        std::copy(mask_vec.begin(), mask_vec.end(), mask.begin());
        auto result = find_best_discard(counts, mask, meld_count);
        py::dict d;
        d["best_shanten"] = result.best_shanten;
        d["best_acceptance"] = result.best_acceptance;
        d["best_tile"] = result.best_tile;
        d["best_mask"] = py::cast(std::vector<int>(result.best_mask.begin(), result.best_mask.end()));
        return d;
    }, py::arg("counts"), py::arg("legal_mask"), py::arg("meld_count") = 0,
       "最善打牌を選択する (シャンテン最小 → 受け入れ最大)");

    m.def("make_discard_mask", [](GameEngine& engine, EnvironmentState& env) {
        auto actions = engine.get_legal_actions(env);
        std::array<int, 34> mask{};
        // 立直打牌があるか確認
        bool has_riichi = false;
        for (const auto& a : actions) {
            if (a.type == ActionType::Discard && a.riichi) {
                has_riichi = true;
                break;
            }
        }
        if (has_riichi) {
            for (const auto& a : actions) {
                if (a.type == ActionType::Discard && a.riichi) {
                    mask[a.tile / 4] = 1;
                }
            }
        } else {
            for (const auto& a : actions) {
                if (a.type == ActionType::Discard) {
                    mask[a.tile / 4] = 1;
                }
            }
        }
        return std::vector<int>(mask.begin(), mask.end());
    }, py::arg("engine"), py::arg("env"),
       "Stage1 用打牌マスク (34次元) を直接生成する");

    m.def("compute_shape_hint", [](const std::vector<int>& counts_vec) {
        if (counts_vec.size() != 34) {
            throw std::invalid_argument("counts must have exactly 34 elements");
        }
        std::array<int, 34> counts;
        std::copy(counts_vec.begin(), counts_vec.end(), counts.begin());
        auto result = compute_shape_hint(counts);
        // chi(21) + outside_wait(24) + inside_wait(21) = 66 要素の連結リスト
        std::vector<float> combined;
        combined.reserve(66);
        combined.insert(combined.end(), result.chi.begin(), result.chi.end());
        combined.insert(combined.end(), result.outside_wait.begin(), result.outside_wait.end());
        combined.insert(combined.end(), result.inside_wait.begin(), result.inside_wait.end());
        return combined;
    }, py::arg("counts"),
       "手牌形状ヒント (chi[21]+outside_wait[24]+inside_wait[21]=66 要素)");

    // --- CQ-0227: round outcome summary ---
    m.def("get_round_outcome", [](const EnvironmentState& env) {
        const auto& rs = env.round_state;
        py::dict outcome;
        outcome["end_reason"] = static_cast<int>(rs.end_reason);

        // tenpai / noten (副露考慮)
        py::list tenpai_list, noten_list;
        for (int p = 0; p < kNumPlayers; ++p) {
            auto counts = hand_utils::make_type_counts(rs.players[p].hand);
            if (hand_utils::is_tenpai_with_melds(counts, rs.players[p].melds)) {
                tenpai_list.append(p);
            } else {
                noten_list.append(p);
            }
        }
        outcome["tenpai_players"] = tenpai_list;
        outcome["noten_players"] = noten_list;

        // win summaries
        py::list wins;
        if (rs.end_reason == RoundEndReason::Tsumo) {
            int winner = rs.current_player;
            const auto& player = rs.players[winner];
            TileType agari_tile = player.hand.back() / 4;
            auto ctx = make_win_context_for_binding(rs, winner, agari_tile, true, false);
            auto counts = hand_utils::make_type_counts(player.hand);
            auto decomps = agari::enumerate_decompositions(counts, player.melds);
            bool is_dealer = (winner == rs.dealer);
            auto sr = score_calculator::calculate_win_score(decomps, ctx, is_dealer, rs.honba);
            if (sr.valid) {
                py::dict w;
                w["winner"] = winner;
                w["is_tsumo"] = true;
                w["is_menzen"] = player.is_menzen;
                w["total_han"] = sr.total_han;
                w["fu"] = sr.fu;
                w["dora_count"] = sr.dora_count;
                w["akadora_count"] = sr.akadora_count;
                w["uradora_count"] = sr.uradora_count;
                py::list yakus;
                for (const auto& y : sr.yakus) {
                    yakus.append(static_cast<int>(y.type));
                }
                w["yaku_ids"] = yakus;
                wins.append(w);
            }
        } else if (rs.end_reason == RoundEndReason::Ron) {
            const auto& ctx_resp = rs.response_context;
            TileId ron_tile = ctx_resp.discard_tile;
            TileType agari_tile = ron_tile / 4;
            bool is_chankan = ctx_resp.is_chankan_response;
            for (int offset = 1; offset <= 3; ++offset) {
                int p = (ctx_resp.discarder + offset) % kNumPlayers;
                if (ctx_resp.has_responded[p] && ctx_resp.responses[p].type == ActionType::Ron) {
                    const auto& player = rs.players[p];
                    auto hand_with_ron = player.hand;
                    hand_with_ron.push_back(ron_tile);
                    auto counts = hand_utils::make_type_counts(hand_with_ron);
                    auto decomps = agari::enumerate_decompositions(counts, player.melds);
                    auto win_ctx = make_win_context_for_binding(rs, p, agari_tile, false, is_chankan);
                    win_ctx.all_tile_ids.push_back(ron_tile);
                    bool is_dealer = (p == rs.dealer);
                    auto sr = score_calculator::calculate_win_score(decomps, win_ctx, is_dealer, rs.honba);
                    if (sr.valid) {
                        py::dict w;
                        w["winner"] = p;
                        w["is_tsumo"] = false;
                        w["is_menzen"] = player.is_menzen;
                        w["total_han"] = sr.total_han;
                        w["fu"] = sr.fu;
                        w["dora_count"] = sr.dora_count;
                        w["akadora_count"] = sr.akadora_count;
                        w["uradora_count"] = sr.uradora_count;
                        py::list yakus;
                        for (const auto& y : sr.yakus) {
                            yakus.append(static_cast<int>(y.type));
                        }
                        w["yaku_ids"] = yakus;
                        wins.append(w);
                    }
                }
            }
            outcome["loser_player"] = static_cast<int>(ctx_resp.discarder);
        }
        outcome["wins"] = wins;

        // winner_players
        py::list winner_players;
        for (size_t i = 0; i < py::len(wins); ++i) {
            winner_players.append(wins[i].attr("__getitem__")("winner"));
        }
        outcome["winner_players"] = winner_players;

        return outcome;
    }, py::arg("env"),
       "局終了時の outcome summary を返す (settle_round 前に呼ぶ)");

    // --- 定数 ---
    m.attr("NUM_TILES") = kNumTiles;
    m.attr("NUM_TILE_TYPES") = kNumTileTypes;
    m.attr("NUM_PLAYERS") = kNumPlayers;
}
