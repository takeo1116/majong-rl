#include "engine/game_engine.h"
#include "core/environment_state.h"
#include "io/display.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <chrono>

using namespace mahjong;

// 方策の種類
enum class PolicyType { First, Random, Discard };

// 方策: 常に最初の合法手
static Action first_policy(const std::vector<Action>& actions, EnvironmentState&) {
    return actions[0];
}

// 方策: ランダム（env.rng 使用で再現可能）
static Action random_policy(const std::vector<Action>& actions, EnvironmentState& env) {
    int idx = env.rng.next_int(static_cast<int>(actions.size()));
    return actions[idx];
}

// 方策: Discard(非立直)/Skip 優先
static Action discard_policy(const std::vector<Action>& actions, EnvironmentState&) {
    for (const auto& a : actions) {
        if (a.type == ActionType::Discard && !a.riichi) return a;
        if (a.type == ActionType::Skip) return a;
    }
    return actions[0];
}

static void print_usage() {
    std::cout << "Usage: mahjong_cli [options]\n"
              << "  --seed <N>              RNG シード（デフォルト: 時刻ベース）\n"
              << "  --dealer <0-3>          起家（デフォルト: 0）\n"
              << "  --mode <debug|fast>     実行モード（デフォルト: debug）\n"
              << "  --quiet                 手ごとのログを抑制（局サマリのみ）\n"
              << "  --policy <first|random|discard>  方策（デフォルト: first）\n"
              << "  --help                  ヘルプ表示\n";
}

int main(int argc, char* argv[]) {
    uint64_t seed = 0;
    bool seed_specified = false;
    PlayerId first_dealer = 0;
    RunMode mode = RunMode::Debug;
    bool quiet = false;
    PolicyType policy_type = PolicyType::First;

    // 引数解析
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = std::strtoull(argv[++i], nullptr, 10);
            seed_specified = true;
        } else if (std::strcmp(argv[i], "--dealer") == 0 && i + 1 < argc) {
            first_dealer = static_cast<PlayerId>(std::atoi(argv[++i]));
            if (first_dealer >= kNumPlayers) {
                std::cerr << "エラー: dealer は 0-3 の範囲で指定してください\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            ++i;
            if (std::strcmp(argv[i], "debug") == 0) {
                mode = RunMode::Debug;
            } else if (std::strcmp(argv[i], "fast") == 0) {
                mode = RunMode::Fast;
            } else {
                std::cerr << "エラー: mode は debug または fast を指定してください\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--quiet") == 0) {
            quiet = true;
        } else if (std::strcmp(argv[i], "--policy") == 0 && i + 1 < argc) {
            ++i;
            if (std::strcmp(argv[i], "first") == 0) {
                policy_type = PolicyType::First;
            } else if (std::strcmp(argv[i], "random") == 0) {
                policy_type = PolicyType::Random;
            } else if (std::strcmp(argv[i], "discard") == 0) {
                policy_type = PolicyType::Discard;
            } else {
                std::cerr << "エラー: policy は first, random, discard のいずれかを指定してください\n";
                return 1;
            }
        } else {
            std::cerr << "不明なオプション: " << argv[i] << "\n";
            print_usage();
            return 1;
        }
    }

    // seed 未指定時は時刻ベース
    if (!seed_specified) {
        seed = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }

    // 方策選択
    auto select_action = [&](const std::vector<Action>& actions, EnvironmentState& env) -> Action {
        switch (policy_type) {
            case PolicyType::First:   return first_policy(actions, env);
            case PolicyType::Random:  return random_policy(actions, env);
            case PolicyType::Discard: return discard_policy(actions, env);
        }
        return first_policy(actions, env);
    };

    // 開始表示
    std::cout << "半荘開始: seed=" << seed
              << " 起家=P" << static_cast<int>(first_dealer)
              << " モード=" << to_string(mode)
              << " 方策=";
    switch (policy_type) {
        case PolicyType::First:   std::cout << "first";   break;
        case PolicyType::Random:  std::cout << "random";  break;
        case PolicyType::Discard: std::cout << "discard"; break;
    }
    std::cout << "\n\n";

    GameEngine engine;
    EnvironmentState env;
    engine.reset_match(env, seed, first_dealer, mode);

    int total_steps = 0;
    int round_count = 0;

    while (!env.match_state.is_match_over) {
        ++round_count;

        // 局ヘッダ
        std::cout << display::round_header(
            env.round_state.round_number, env.round_state.honba,
            env.round_state.kyotaku, env.round_state.dealer) << "\n";
        std::cout << "スコア: " << display::scores_to_string(env.match_state.scores) << "\n";

        // 手牌表示（非 quiet 時）
        if (!quiet) {
            for (PlayerId p = 0; p < kNumPlayers; ++p) {
                std::cout << "  P" << static_cast<int>(p) << " 手牌: "
                          << display::hand_to_string(env.round_state.players[p].hand) << "\n";
            }
            std::cout << "\n";
        }

        // 局を進行
        while (!env.match_state.is_match_over) {
            auto actions = engine.get_legal_actions(env);
            if (actions.empty()) break;

            Action chosen = select_action(actions, env);

            if (!quiet) {
                std::cout << "  " << display::action_display(chosen) << "\n";
            }

            auto result = engine.step(env, chosen);
            if (result.error != ErrorCode::Ok) {
                std::cerr << "エラー: " << to_string(result.error) << " (step " << total_steps << ")\n";
                return 1;
            }

            // イベント表示（非 quiet 時、主要イベントのみ）
            if (!quiet) {
                for (const auto& evt : result.events) {
                    // 配牌・打牌イベントは冗長なのでスキップ（アクションで表示済み）
                    if (evt.type == EventType::Deal || evt.type == EventType::Discard) continue;
                    std::cout << "    " << display::event_display(evt) << "\n";
                }
            }

            ++total_steps;

            if (result.round_over) {
                // 局終了サマリ
                std::cout << display::round_end_summary(env) << "\n\n";
                if (!result.match_over) {
                    engine.advance_round(env);
                }
                break;
            }
        }
    }

    // 半荘終了サマリ
    std::cout << display::match_end_summary(env) << "\n";
    std::cout << "局数: " << round_count << " ステップ数: " << total_steps << "\n";

    return 0;
}
