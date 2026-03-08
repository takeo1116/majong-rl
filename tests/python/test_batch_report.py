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


class TestImitationLossModeBatch:
    """imitation_loss_mode の batch レポート追跡テスト (CQ-0132)"""

    def _make_result_with_imitation(self, tmp_path: Path, seed: int,
                                     loss_mode: str = "strict_top1"):
        run_dir = tmp_path / f"fake_run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # summary.json を作成（batch_report はここから imitation_metrics を読む）
        summary_data = {
            "phase_stats": {
                "imitation": {
                    "teacher_top1_match_rate": 0.5,
                    "teacher_best_set_hit_rate": 0.7,
                    "imitation_loss_mode": loss_mode,
                },
            },
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary_data, f)
        return {
            "seed": seed,
            "success": True,
            "result": {
                "run_dir": str(run_dir),
                "global_seed": seed,
                "selfplay_stats": {},
                "eval_metrics": {"avg_rank": 2.5, "avg_score": 100.0,
                                 "win_rate": 0.3, "deal_in_rate": 0.1},
            },
        }

    def test_loss_mode_in_batch_summary(self, tmp_path: Path):
        """batch_summary.json の per-run に imitation_loss_mode が含まれる"""
        results = [self._make_result_with_imitation(tmp_path, 42, "tie_aware_best_set")]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        assert "imitation_metrics" in run_entry
        assert run_entry["imitation_metrics"]["imitation_loss_mode"] == "tie_aware_best_set"

    def test_loss_mode_in_csv(self, tmp_path: Path):
        """batch_table.csv に imitation_loss_mode 列がある"""
        results = [self._make_result_with_imitation(tmp_path, 42, "strict_top1")]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_table.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["imitation_loss_mode"] == "strict_top1"

    def test_mixed_mode_aggregate(self, tmp_path: Path):
        """strict/tie-aware 混在 batch で mode 別集約が出る (CQ-0133)"""
        results = [
            self._make_result_with_imitation(tmp_path, 42, "strict_top1"),
            self._make_result_with_imitation(tmp_path, 43, "strict_top1"),
            self._make_result_with_imitation(tmp_path, 44, "tie_aware_best_set"),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)

        agg = summary["aggregate"]
        # 既存の全体集約は維持
        assert "imitation" in agg
        assert agg["imitation"]["teacher_top1_match_rate"]["count"] == 3

        # mode 別集約
        assert "imitation_by_loss_mode" in agg
        by_mode = agg["imitation_by_loss_mode"]
        assert "strict_top1" in by_mode
        assert by_mode["strict_top1"]["teacher_top1_match_rate"]["count"] == 2
        assert "tie_aware_best_set" in by_mode
        assert by_mode["tie_aware_best_set"]["teacher_top1_match_rate"]["count"] == 1

    def test_single_mode_aggregate(self, tmp_path: Path):
        """単一 mode batch でも mode 別集約が出る (CQ-0133)"""
        results = [
            self._make_result_with_imitation(tmp_path, 42, "tie_aware_best_set"),
            self._make_result_with_imitation(tmp_path, 43, "tie_aware_best_set"),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)

        by_mode = summary["aggregate"]["imitation_by_loss_mode"]
        assert list(by_mode.keys()) == ["tie_aware_best_set"]
        assert by_mode["tie_aware_best_set"]["teacher_top1_match_rate"]["count"] == 2

    def _make_result_with_imitation_none_mode(self, tmp_path: Path, seed: int):
        """imitation_loss_mode が None の旧成果物を模擬"""
        run_dir = tmp_path / f"fake_run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_data = {
            "phase_stats": {
                "imitation": {
                    "teacher_top1_match_rate": 0.4,
                    "teacher_best_set_hit_rate": 0.6,
                    "imitation_loss_mode": None,
                },
            },
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary_data, f)
        return {
            "seed": seed,
            "success": True,
            "result": {
                "run_dir": str(run_dir),
                "global_seed": seed,
                "selfplay_stats": {},
                "eval_metrics": {"avg_rank": 2.5, "avg_score": 100.0,
                                 "win_rate": 0.3, "deal_in_rate": 0.1},
            },
        }

    def test_none_mode_mixed_no_crash(self, tmp_path: Path):
        """None + strict_top1 混在でクラッシュしない (CQ-0134)"""
        results = [
            self._make_result_with_imitation_none_mode(tmp_path, 42),
            self._make_result_with_imitation(tmp_path, 43, "strict_top1"),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)

        agg = summary["aggregate"]
        assert "imitation" in agg
        assert agg["imitation"]["teacher_top1_match_rate"]["count"] == 2

        by_mode = agg["imitation_by_loss_mode"]
        assert "unknown" in by_mode
        assert by_mode["unknown"]["teacher_top1_match_rate"]["count"] == 1
        assert "strict_top1" in by_mode
        assert by_mode["strict_top1"]["teacher_top1_match_rate"]["count"] == 1

    def test_none_and_tie_aware_mixed(self, tmp_path: Path):
        """None + tie_aware 混在で mode 別 count が正しい (CQ-0134)"""
        results = [
            self._make_result_with_imitation_none_mode(tmp_path, 42),
            self._make_result_with_imitation_none_mode(tmp_path, 43),
            self._make_result_with_imitation(tmp_path, 44, "tie_aware_best_set"),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)

        by_mode = summary["aggregate"]["imitation_by_loss_mode"]
        assert by_mode["unknown"]["teacher_top1_match_rate"]["count"] == 2
        assert by_mode["tie_aware_best_set"]["teacher_top1_match_rate"]["count"] == 1


class TestLearnerDiagBatch:
    """learner 診断統計の batch レポート統合テスト (CQ-0137)"""

    def _make_result_with_ppo_diag(self, tmp_path: Path, seed: int,
                                    clip_fraction: float = 0.1,
                                    advantage_mean: float = 0.0):
        run_dir = tmp_path / f"fake_run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ppo_diag = {
            "advantage_mean": advantage_mean,
            "advantage_std": 1.0,
            "advantage_p50": 0.0, "advantage_p90": 1.0, "advantage_p99": 2.0,
            "advantage_positive_ratio": 0.5, "advantage_negative_ratio": 0.5,
            "return_mean": 0.1, "return_std": 0.5,
            "return_p50": 0.1, "return_p90": 0.5, "return_p99": 1.0,
            "old_value_mean": 0.05, "old_value_std": 0.3,
            "new_value_mean": 0.08, "new_value_std": 0.25,
            "value_error_mean": -0.05, "value_error_std": 0.4,
            "value_error_p50": -0.02, "value_error_p90": 0.3, "value_error_p99": 0.8,
            "ratio_mean": 1.01, "ratio_std": 0.1,
            "ratio_p50": 1.0, "ratio_p90": 1.1, "ratio_p99": 1.3,
            "clip_fraction": clip_fraction,
        }
        summary_data = {
            "phase_stats": {
                "learner": {
                    "total_steps": 100,
                    "num_updates": 10,
                    "policy_loss": 0.5,
                    "value_loss": 0.3,
                    "mode": "ppo",
                    "ppo_diag": ppo_diag,
                },
            },
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary_data, f)
        return {
            "seed": seed,
            "success": True,
            "result": {
                "run_dir": str(run_dir),
                "global_seed": seed,
                "selfplay_stats": {},
                "eval_metrics": {"avg_rank": 2.5, "avg_score": 100.0,
                                 "win_rate": 0.3, "deal_in_rate": 0.1},
            },
        }

    def test_learner_diag_in_runs(self, tmp_path: Path):
        """runs[*].learner_diag が batch_summary に含まれる"""
        results = [self._make_result_with_ppo_diag(tmp_path, 42, clip_fraction=0.15)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        assert "learner_diag" in run_entry
        assert run_entry["learner_diag"]["clip_fraction"] == 0.15

    def test_learner_diag_aggregate(self, tmp_path: Path):
        """aggregate.learner_diag に mean/std が集約される"""
        results = [
            self._make_result_with_ppo_diag(tmp_path, 42, clip_fraction=0.1),
            self._make_result_with_ppo_diag(tmp_path, 43, clip_fraction=0.2),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        agg = summary["aggregate"]
        assert "learner_diag" in agg
        assert agg["learner_diag"]["clip_fraction"]["count"] == 2
        assert agg["learner_diag"]["clip_fraction"]["mean"] == pytest.approx(0.15, abs=1e-6)

    def test_no_learner_diag_no_crash(self, tmp_path: Path):
        """ppo_diag なし（imitation のみ）でクラッシュしない"""
        run_dir = tmp_path / "fake_run_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_data = {
            "phase_stats": {
                "imitation": {
                    "teacher_top1_match_rate": 0.5,
                    "teacher_best_set_hit_rate": 0.7,
                },
            },
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
        assert "learner_diag" not in summary["runs"][0]
        assert "learner_diag" not in summary["aggregate"]

    def test_mixed_with_and_without_diag(self, tmp_path: Path):
        """ppo_diag あり/なし混在でもクラッシュせず、あり分だけ集約される"""
        # seed 42: ppo_diag あり
        results = [self._make_result_with_ppo_diag(tmp_path, 42, clip_fraction=0.1)]
        # seed 43: ppo_diag なし
        run_dir_43 = tmp_path / "fake_run_43"
        run_dir_43.mkdir(parents=True, exist_ok=True)
        with open(run_dir_43 / "summary.json", "w") as f:
            json.dump({"phase_stats": {}}, f)
        results.append({
            "seed": 43,
            "success": True,
            "result": {
                "run_dir": str(run_dir_43),
                "global_seed": 43,
                "selfplay_stats": {},
                "eval_metrics": {"avg_rank": 2.5, "avg_score": 100.0,
                                 "win_rate": 0.3, "deal_in_rate": 0.1},
            },
        })
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert "learner_diag" in summary["runs"][0]
        assert "learner_diag" not in summary["runs"][1]
        # aggregate は1件のみ
        assert summary["aggregate"]["learner_diag"]["clip_fraction"]["count"] == 1
