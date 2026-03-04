"""テスト: batch_report.py — バッチ集約レポート (CQ-0078, CQ-0080)"""
import csv
import json
import pytest
from pathlib import Path

pytestmark = pytest.mark.smoke

from mahjong_rl.batch_report import generate_batch_report, _compute_aggregate


class TestGenerateBatchReport:
    """集約レポート生成テスト"""

    def _make_success_result(self, seed: int, avg_rank: float = 2.5,
                              run_dir: str = "") -> dict:
        return {
            "seed": seed,
            "success": True,
            "result": {
                "run_dir": run_dir or f"/tmp/run_{seed}",
                "global_seed": seed,
                "eval_metrics": {
                    "avg_rank": avg_rank,
                    "avg_score": 100.0,
                    "win_rate": 0.3,
                    "deal_in_rate": 0.1,
                },
            },
        }

    def _make_failure_result(self, seed: int) -> dict:
        return {
            "seed": seed,
            "success": False,
            "error": f"テスト用エラー seed={seed}",
        }

    def test_all_success(self, tmp_path: Path):
        """全成功時のレポート生成"""
        results = [
            self._make_success_result(42, avg_rank=2.0),
            self._make_success_result(43, avg_rank=3.0),
        ]
        generate_batch_report(tmp_path, results)

        # batch_summary.json
        summary_path = tmp_path / "batch_summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["num_seeds"] == 2
        assert summary["seeds"] == [42, 43]
        assert summary["success_count"] == 2
        assert summary["failure_count"] == 0
        assert summary["success_rate"] == 1.0
        assert len(summary["runs"]) == 2
        assert summary["aggregate"]["avg_rank"]["mean"] == 2.5

        # batch_table.csv
        csv_path = tmp_path / "batch_table.csv"
        assert csv_path.exists()

    def test_with_failure(self, tmp_path: Path):
        """failure 混在時のレポート生成"""
        results = [
            self._make_success_result(42),
            self._make_failure_result(43),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert summary["success_count"] == 1
        assert summary["failure_count"] == 1
        assert summary["success_rate"] == 0.5
        # failure run にはエラー情報がある
        fail_run = summary["runs"][1]
        assert fail_run["success"] is False
        assert "エラー" in fail_run["error"]

    def test_csv_output(self, tmp_path: Path):
        """CSV 形式の検証"""
        results = [
            self._make_success_result(42, avg_rank=2.0),
            self._make_success_result(43, avg_rank=3.0),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_table.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["seed"] == "42"
        assert rows[0]["success"] == "True"
        assert float(rows[0]["avg_rank"]) == 2.0

    def test_empty_results(self, tmp_path: Path):
        """空の結果でもレポート生成できる"""
        generate_batch_report(tmp_path, [])

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert summary["num_seeds"] == 0
        assert summary["aggregate"] == {}


class TestComputeAggregate:
    """集約統計計算テスト"""

    def test_single_value(self):
        """1 件の場合は std=0"""
        result = _compute_aggregate([{"avg_rank": 2.5}])
        assert result["avg_rank"]["mean"] == 2.5
        assert result["avg_rank"]["std"] == 0.0
        assert result["avg_rank"]["count"] == 1

    def test_multiple_values(self):
        """複数件の平均・標準偏差"""
        metrics = [
            {"avg_rank": 2.0, "win_rate": 0.3},
            {"avg_rank": 4.0, "win_rate": 0.5},
        ]
        result = _compute_aggregate(metrics)
        assert result["avg_rank"]["mean"] == 3.0
        assert result["avg_rank"]["min"] == 2.0
        assert result["avg_rank"]["max"] == 4.0
        assert result["avg_rank"]["std"] > 0

    def test_empty_list(self):
        """空リストでは空 dict"""
        assert _compute_aggregate([]) == {}

    def test_missing_keys_skipped(self):
        """一部の指標が欠けていてもエラーにならない"""
        metrics = [
            {"avg_rank": 2.0},
            {"avg_rank": 3.0, "win_rate": 0.4},
        ]
        result = _compute_aggregate(metrics)
        assert "avg_rank" in result
        assert result["avg_rank"]["count"] == 2
        assert result["win_rate"]["count"] == 1


class TestWorkerSettingsAndFailureReason:
    """worker 設定と失敗理由の記録テスト (CQ-0081)"""

    def test_worker_settings_in_runs(self, tmp_path: Path):
        """成功 run に worker_settings が記録される"""
        results = [{
            "seed": 42,
            "success": True,
            "result": {
                "run_dir": str(tmp_path / "fake_run"),
                "global_seed": 42,
                "selfplay_stats": {"num_workers": 4, "total_steps": 100},
                "eval_metrics": {
                    "num_workers": 2,
                    "avg_rank": 2.5,
                    "avg_score": 100.0,
                    "win_rate": 0.3,
                    "deal_in_rate": 0.1,
                },
            },
        }]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        assert "worker_settings" in run_entry
        assert run_entry["worker_settings"]["selfplay_num_workers"] == 4
        assert run_entry["worker_settings"]["evaluation_num_workers"] == 2

    def test_failure_reason_not_unknown(self, tmp_path: Path):
        """失敗理由が unknown にならない"""
        results = [{
            "seed": 42,
            "success": False,
            "error": "selfplay: RuntimeError: テスト用エラー",
        }]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        fail_entry = summary["runs"][0]
        assert fail_entry["error"] != "unknown"
        assert "テスト用エラー" in fail_entry["error"]

    def test_device_info_from_summary(self, tmp_path: Path):
        """summary.json から device_info が読み込まれる"""
        # run_dir に summary.json を作る
        run_dir = tmp_path / "fake_run"
        run_dir.mkdir()
        summary_data = {
            "device_info": {"training": {"requested": "auto", "resolved": "cuda"}},
            "env_info": {"torch_version": "2.0.0"},
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary_data, f)

        results = [{
            "seed": 42,
            "success": True,
            "result": {
                "run_dir": str(run_dir),
                "global_seed": 42,
                "selfplay_stats": {},
                "eval_metrics": {"avg_rank": 2.5, "avg_score": 100.0,
                                 "win_rate": 0.3, "deal_in_rate": 0.1},
            },
        }]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        assert "device_info" in run_entry
        assert run_entry["device_info"]["training"]["resolved"] == "cuda"
        assert "env_info" in run_entry
        assert run_entry["env_info"]["torch_version"] == "2.0.0"
