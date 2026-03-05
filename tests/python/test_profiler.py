"""テスト: profiler.py — 簡易プロファイラ (CQ-0098)"""
import json
import pytest
from pathlib import Path

pytestmark = pytest.mark.smoke

from mahjong_rl.profiler import Profiler


class TestProfiler:
    """Profiler 単体テスト"""

    def test_profiler_start_stop(self):
        """start/stop で計測が記録される"""
        p = Profiler(enabled=True)
        p.start("test_op")
        p.stop("test_op")

        d = p.to_dict()
        assert d["enabled"] is True
        assert "test_op" in d["entries"]
        assert d["entries"]["test_op"]["count"] == 1
        assert d["entries"]["test_op"]["total_sec"] >= 0

    def test_profiler_disabled_noop(self):
        """enabled=False で to_dict が空"""
        p = Profiler(enabled=False)
        p.start("test_op")
        p.stop("test_op")

        d = p.to_dict()
        assert d["enabled"] is False
        assert d["entries"] == {}

    def test_profiler_multiple_calls(self):
        """同じ name の複数呼び出しで count/total が蓄積"""
        p = Profiler(enabled=True)
        for _ in range(3):
            p.start("repeated")
            p.stop("repeated")

        d = p.to_dict()
        assert d["entries"]["repeated"]["count"] == 3
        assert d["entries"]["repeated"]["total_sec"] >= 0
        assert d["entries"]["repeated"]["mean_sec"] >= 0

    def test_profiler_save(self, tmp_path: Path):
        """save で profile.json が生成される"""
        p = Profiler(enabled=True)
        p.start("save_test")
        p.stop("save_test")
        p.save(tmp_path / "profile.json")

        assert (tmp_path / "profile.json").exists()
        with open(tmp_path / "profile.json") as f:
            data = json.load(f)
        assert data["enabled"] is True
        assert "save_test" in data["entries"]

    def test_profiler_save_disabled(self, tmp_path: Path):
        """enabled=False で save してもファイルが生成されない"""
        p = Profiler(enabled=False)
        p.save(tmp_path / "profile.json")
        assert not (tmp_path / "profile.json").exists()

    def test_stop_without_start(self):
        """start なしの stop は無視される"""
        p = Profiler(enabled=True)
        p.stop("no_start")
        d = p.to_dict()
        assert "no_start" not in d["entries"]
