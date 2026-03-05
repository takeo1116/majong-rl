"""簡易プロファイラ: phase 内の主要処理の経過時間を計測する (CQ-0098)"""
from __future__ import annotations

import json
import time
from pathlib import Path


class Profiler:
    """計測ポイントの経過時間を記録する簡易プロファイラ

    enabled=False の場合は全メソッドが no-op になる。
    """

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._entries: dict[str, dict] = {}
        self._running: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self, name: str) -> None:
        """計測を開始する"""
        if not self._enabled:
            return
        self._running[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        """計測を停止し、経過時間を記録する"""
        if not self._enabled:
            return
        t_end = time.perf_counter()
        t_start = self._running.pop(name, None)
        if t_start is None:
            return
        elapsed = t_end - t_start
        if name not in self._entries:
            self._entries[name] = {"count": 0, "total_sec": 0.0}
        entry = self._entries[name]
        entry["count"] += 1
        entry["total_sec"] = round(entry["total_sec"] + elapsed, 6)

    def to_dict(self) -> dict:
        """計測結果を dict として返す"""
        if not self._enabled:
            return {"enabled": False, "entries": {}}
        entries = {}
        for name, entry in self._entries.items():
            count = entry["count"]
            total = entry["total_sec"]
            entries[name] = {
                "count": count,
                "total_sec": round(total, 6),
                "mean_sec": round(total / count, 6) if count > 0 else 0.0,
            }
        return {"enabled": True, "entries": entries}

    def save(self, path: Path) -> None:
        """計測結果を JSON ファイルに保存する（enabled 時のみ）"""
        if not self._enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
