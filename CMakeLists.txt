cmake_minimum_required(VERSION 3.20)
project(mahjong-rl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# コンパイルオプション
add_compile_options(-Wall -Wextra -Wpedantic)

# ライブラリ
add_library(mahjong_core STATIC
    src/core/tile.cpp
    src/core/types.cpp
    src/core/meld.cpp
    src/core/action.cpp
    src/core/event.cpp
    src/core/player_state.cpp
    src/core/round_state.cpp
    src/core/match_state.cpp
    src/core/environment_state.cpp
    src/engine/hand_utils.cpp
    src/engine/game_engine.cpp
    src/rules/agari.cpp
    src/rules/yaku.cpp
    src/rules/fu_calculator.cpp
    src/rules/score_calculator.cpp
    src/engine/state_validator.cpp
    src/rl/observation.cpp
    src/rl/reward_policy.cpp
    src/rl/search_utils.cpp
    src/io/game_record.cpp
    src/io/display.cpp
)
target_include_directories(mahjong_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(mahjong_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# サンプル実行ファイル
add_executable(mahjong_example examples/main.cpp)
target_link_libraries(mahjong_example PRIVATE mahjong_core)

# CLI デバッグ実行ツール
add_executable(mahjong_cli tools/mahjong_cli.cpp)
target_link_libraries(mahjong_cli PRIVATE mahjong_core)

# テスト
enable_testing()
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(mahjong_tests
    tests/unit/tile_test.cpp
    tests/unit/types_test.cpp
    tests/unit/meld_test.cpp
    tests/unit/action_test.cpp
    tests/unit/event_test.cpp
    tests/unit/state_test.cpp
    tests/unit/environment_test.cpp
    tests/unit/phase_test.cpp
    tests/unit/hand_utils_test.cpp
    tests/unit/game_engine_test.cpp
    tests/unit/round_reset_test.cpp
    tests/unit/illegal_action_test.cpp
    tests/unit/agari_test.cpp
    tests/unit/yaku_test.cpp
    tests/unit/fu_calculator_test.cpp
    tests/unit/score_calculator_test.cpp
    tests/unit/ippatsu_test.cpp
    tests/unit/chankan_test.cpp
    tests/unit/furiten_test.cpp
    tests/unit/settlement_test.cpp
    tests/unit/furiten_naki_test.cpp
    tests/unit/multi_ron_test.cpp
    tests/unit/oorasu_test.cpp
    tests/unit/state_validator_test.cpp
    tests/unit/observation_test.cpp
    tests/unit/reward_policy_test.cpp
    tests/unit/search_utils_test.cpp
    tests/unit/game_record_test.cpp
    tests/unit/debug_fast_test.cpp
    tests/unit/reproducibility_test.cpp
    tests/integration/round_flow_test.cpp
    tests/replay/replay_test.cpp
)
target_link_libraries(mahjong_tests PRIVATE mahjong_core GTest::gtest_main)
target_include_directories(mahjong_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(GoogleTest)
gtest_discover_tests(mahjong_tests)

# pybind11 Python バインディング（Python が見つかった場合のみ）
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(Python3_FOUND)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(_mahjong_core bindings/mahjong_bindings.cpp)
    target_link_libraries(_mahjong_core PRIVATE mahjong_core)
    target_include_directories(_mahjong_core PRIVATE ${CMAKE_SOURCE_DIR}/src)

    install(TARGETS _mahjong_core DESTINATION mahjong_rl)
endif()
