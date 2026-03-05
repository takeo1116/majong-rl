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

    def test_outlier_info_in_aggregate(self, tmp_path: Path):
        """outlier_min/outlier_max に seed と run_dir がある (CQ-0094)"""
        results = [
            self._make_success_result(42, avg_rank=2.0),
            self._make_success_result(43, avg_rank=3.0),
            self._make_success_result(44, avg_rank=4.0),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        ar = summary["aggregate"]["avg_rank"]
        assert ar["outlier_min"]["seed"] == 42
        assert ar["outlier_min"]["value"] == 2.0
        assert ar["outlier_max"]["seed"] == 44
        assert ar["outlier_max"]["value"] == 4.0
        assert "run_dir" in ar["outlier_min"]
        assert "run_dir" in ar["outlier_max"]

    def test_outlier_single_seed(self, tmp_path: Path):
        """1件の場合 outlier_min == outlier_max (CQ-0094)"""
        results = [self._make_success_result(42, avg_rank=2.5)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        ar = summary["aggregate"]["avg_rank"]
        assert ar["outlier_min"]["seed"] == 42
        assert ar["outlier_max"]["seed"] == 42
        assert ar["outlier_min"]["value"] == ar["outlier_max"]["value"]

    def test_outlier_with_missing_eval_metrics(self, tmp_path: Path):
        """eval_metrics 欠損 run 混在時も outlier の seed 対応が正しい (CQ-0099)"""
        results = [
            self._make_success_result(42, avg_rank=2.0),
            # seed=43 は成功だが eval_metrics なし
            {
                "seed": 43,
                "success": True,
                "result": {
                    "run_dir": "/tmp/run_43",
                    "global_seed": 43,
                },
            },
            self._make_success_result(44, avg_rank=4.0),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        ar = summary["aggregate"]["avg_rank"]
        # seed=43 をスキップしても outlier_max は seed=44 を正しく指す
        assert ar["outlier_min"]["seed"] == 42
        assert ar["outlier_min"]["value"] == 2.0
        assert ar["outlier_max"]["seed"] == 44
        assert ar["outlier_max"]["value"] == 4.0


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

    def test_se_and_ci_with_multiple_values(self):
        """複数件で SE > 0、CI が mean を挟む (CQ-0094)"""
        metrics = [
            {"avg_rank": 2.0, "win_rate": 0.3},
            {"avg_rank": 4.0, "win_rate": 0.5},
            {"avg_rank": 3.0, "win_rate": 0.4},
        ]
        result = _compute_aggregate(metrics)
        ar = result["avg_rank"]
        assert ar["se"] > 0
        assert ar["ci_95_lower"] < ar["mean"]
        assert ar["ci_95_upper"] > ar["mean"]
        # CI が mean を中心に対称
        assert abs((ar["mean"] - ar["ci_95_lower"])
                    - (ar["ci_95_upper"] - ar["mean"])) < 1e-6

    def test_se_single_value(self):
        """1件で SE=0、CI lower == CI upper == mean (CQ-0094)"""
        result = _compute_aggregate([{"avg_rank": 2.5}])
        ar = result["avg_rank"]
        assert ar["se"] == 0.0
        assert ar["ci_95_lower"] == ar["mean"]
        assert ar["ci_95_upper"] == ar["mean"]


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


class TestRotationEvalBatchAggregation:
    """rotation eval の batch 集約テスト (CQ-0092)"""

    def _make_result(self, seed: int, eval_mode: str = "single",
                     avg_rank: float = 2.5) -> dict:
        return {
            "seed": seed,
            "success": True,
            "result": {
                "run_dir": f"/tmp/run_{seed}",
                "global_seed": seed,
                "eval_metrics": {
                    "eval_mode": eval_mode,
                    "avg_rank": avg_rank,
                    "avg_score": 100.0,
                    "win_rate": 0.3,
                    "deal_in_rate": 0.1,
                },
            },
        }

    def test_rotation_eval_mode_in_summary(self, tmp_path: Path):
        """rotation eval の eval_mode が batch_summary に記録される"""
        results = [
            self._make_result(42, eval_mode="rotation", avg_rank=2.0),
            self._make_result(43, eval_mode="rotation", avg_rank=3.0),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)

        # per-seed entry に eval_mode がある
        assert summary["runs"][0]["eval_mode"] == "rotation"
        assert summary["runs"][1]["eval_mode"] == "rotation"

        # aggregate に eval_mode がある
        assert summary["aggregate"]["eval_mode"] == "rotation"

    def test_single_eval_mode_in_summary(self, tmp_path: Path):
        """single eval の eval_mode が batch_summary に記録される"""
        results = [
            self._make_result(42, eval_mode="single"),
            self._make_result(43, eval_mode="single"),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)

        assert summary["aggregate"]["eval_mode"] == "single"

    def test_mixed_eval_mode(self, tmp_path: Path):
        """single と rotation 混在時は mixed"""
        results = [
            self._make_result(42, eval_mode="single"),
            self._make_result(43, eval_mode="rotation"),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)

        assert summary["aggregate"]["eval_mode"] == "mixed"

    def test_csv_has_eval_mode_column(self, tmp_path: Path):
        """CSV に eval_mode 列がある"""
        results = [
            self._make_result(42, eval_mode="rotation"),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_table.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["eval_mode"] == "rotation"
