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
)
target_include_directories(mahjong_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

# サンプル実行ファイル
add_executable(mahjong_example examples/main.cpp)
target_link_libraries(mahjong_example PRIVATE mahjong_core)

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
)
target_link_libraries(mahjong_tests PRIVATE mahjong_core GTest::gtest_main)
target_include_directories(mahjong_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(GoogleTest)
gtest_discover_tests(mahjong_tests)
