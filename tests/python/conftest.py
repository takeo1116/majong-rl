"""共有テストフィクスチャ"""
import os
import multiprocessing as mp

import pytest
import mahjong_rl


def _can_spawn_subprocess() -> bool:
    """subprocess (spawn) が利用可能か簡易判定する"""
    try:
        ctx = mp.get_context("spawn")
        # 実際に Process を起動せず、context 取得が成功すれば OK とする
        # sandbox 等で fork/spawn が禁止されている場合は例外になる
        return True
    except Exception:
        return False


# requires_multiprocess マーカーの自動 skip (CQ-0104)
def pytest_collection_modifyitems(config, items):
    """multiprocess 依存テストを条件未満環境で skip する"""
    if _can_spawn_subprocess():
        return
    skip_mp = pytest.mark.skip(
        reason="multiprocess 未対応環境（spawn コンテキスト取得失敗）")
    for item in items:
        if "requires_multiprocess" in item.keywords:
            item.add_marker(skip_mp)


@pytest.fixture
def engine():
    """ゲームエンジンインスタンス"""
    return mahjong_rl.GameEngine()


@pytest.fixture
def env():
    """環境状態インスタンス"""
    return mahjong_rl.EnvironmentState()


@pytest.fixture
def initialized_env(engine, env):
    """seed=42 で初期化済みの環境"""
    engine.reset_match(env, 42)
    return env
