"""共有テストフィクスチャ"""
import pytest
import mahjong_rl


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
