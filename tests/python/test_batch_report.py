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

    def _make_result_with_shanten_diag(self, tmp_path: Path, seed: int):
        """CQ-0147: shanten_diag 付き ppo_diag を含む結果を生成"""
        run_dir = tmp_path / f"fake_run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ppo_diag = {
            "advantage_mean": 0.0, "advantage_std": 1.0,
            "clip_fraction": 0.1,
            "ratio_mean": 1.0, "ratio_std": 0.05,
            "old_value_mean": 0.0, "new_value_mean": 0.0,
            "value_error_mean": 0.0, "value_error_std": 0.1,
            "shanten_diag": {
                "status": "complete",
                "total_samples": 100,
                "available_samples": 100,
                "unavailable_samples": 0,
                "improve": {
                    "count": 30,
                    "advantage": {"mean": 0.5, "std": 0.3, "p50": 0.4,
                                  "p90": 0.8, "p99": 1.0,
                                  "positive_ratio": 0.8, "negative_ratio": 0.2},
                    "return": {"mean": 0.3, "std": 0.2, "p50": 0.3,
                               "p90": 0.5, "p99": 0.7},
                    "value_error": {"mean": -0.1, "std": 0.1, "p50": -0.1,
                                    "p90": 0.0, "p99": 0.1},
                },
                "same": {"count": 40},
                "worsen": {"count": 30},
            },
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

    def test_shanten_diag_in_batch_runs(self, tmp_path: Path):
        """CQ-0147: runs[*].learner_diag.shanten_diag が batch_summary に含まれる"""
        results = [self._make_result_with_shanten_diag(tmp_path, 42)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        assert "learner_diag" in run_entry
        assert "shanten_diag" in run_entry["learner_diag"]
        sd = run_entry["learner_diag"]["shanten_diag"]
        assert sd["status"] == "complete"
        assert sd["improve"]["count"] == 30

    def test_shanten_diag_mixed_no_crash(self, tmp_path: Path):
        """CQ-0147: shanten_diag あり/なし混在でも batch がクラッシュしない"""
        results = [self._make_result_with_shanten_diag(tmp_path, 42)]
        # seed 43: shanten_diag なし
        results.append(self._make_result_with_ppo_diag(tmp_path, 43))
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        # seed 42 には shanten_diag あり
        assert "shanten_diag" in summary["runs"][0]["learner_diag"]
        # seed 43 には shanten_diag なし（通常の ppo_diag のみ）
        assert "learner_diag" in summary["runs"][1]
        # aggregate は落ちない
        assert "learner_diag" in summary["aggregate"]


class TestRewardCompositionBatch:
    """reward composition の batch レポート統合テスト (CQ-0141)"""

    def _make_result_with_rc(self, tmp_path: Path, seed: int,
                              shanten_delta_mean: float = 0.001,
                              include_shaping: bool = False):
        run_dir = tmp_path / f"fake_run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        sp_data: dict = {
            "total_steps": 500,
            "total_matches": 2,
            "reward_composition": {
                "shanten_delta_enabled": True,
                "point_delta": {
                    "count": 500, "sum": 0.05, "mean": 0.0001,
                    "std": 0.001, "nonzero_count": 50,
                    "p50": 0.0, "p90": 0.0002, "p99": 0.01,
                },
                "shanten_delta": {
                    "count": 500, "sum": shanten_delta_mean * 500,
                    "mean": shanten_delta_mean,
                    "std": 0.002, "nonzero_count": 100,
                    "p50": 0.0, "p90": shanten_delta_mean * 2, "p99": shanten_delta_mean * 5,
                },
                "total": {
                    "count": 500, "sum": 0.05 + shanten_delta_mean * 500,
                    "mean": 0.0001 + shanten_delta_mean,
                    "std": 0.002, "nonzero_count": 120,
                    "p50": 0.0, "p90": 0.0003, "p99": 0.012,
                },
            },
        }
        if include_shaping:
            sp_data["reward_shaping"] = {
                "shanten_delta": {
                    "enabled": True,
                    "scale": 0.01,
                    "mode": "both",
                    "schedule_type": "constant",
                },
            }
        summary_data = {"phase_stats": {"selfplay": sp_data}}
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

    def test_reward_composition_in_runs(self, tmp_path: Path):
        """runs[*].reward_composition が batch_summary に含まれる"""
        results = [self._make_result_with_rc(tmp_path, 42)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        assert "reward_composition" in run_entry
        assert run_entry["reward_composition"]["shanten_delta_enabled"] is True

    def test_mixed_with_without_no_crash(self, tmp_path: Path):
        """reward_composition あり/なし混在でもクラッシュせず集約される"""
        results = [self._make_result_with_rc(tmp_path, 42, shanten_delta_mean=0.002)]
        # seed 43: reward_composition なし
        run_dir_43 = tmp_path / "fake_run_43"
        run_dir_43.mkdir(parents=True, exist_ok=True)
        with open(run_dir_43 / "summary.json", "w") as f:
            json.dump({"phase_stats": {"selfplay": {"total_steps": 100}}}, f)
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
        assert "reward_composition" in summary["runs"][0]
        assert "reward_composition" not in summary["runs"][1]
        # aggregate にあり分の集約がある
        agg_rc = summary["aggregate"].get("reward_composition", {})
        assert "shanten_delta" in agg_rc

    def test_quantile_in_aggregate(self, tmp_path: Path):
        """aggregate.reward_composition に quantile 系の集約がある (CQ-0142)"""
        results = [
            self._make_result_with_rc(tmp_path, 42, shanten_delta_mean=0.001),
            self._make_result_with_rc(tmp_path, 43, shanten_delta_mean=0.003),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        agg_rc = summary["aggregate"]["reward_composition"]
        # p50/p90/p99 が集約されている
        for comp in ("point_delta", "shanten_delta", "total"):
            assert comp in agg_rc
            for qk in ("p50", "p90", "p99"):
                assert qk in agg_rc[comp], f"aggregate.reward_composition.{comp}.{qk} missing"
                assert agg_rc[comp][qk]["count"] == 2

    def test_reward_shaping_in_runs(self, tmp_path: Path):
        """runs[*].reward_shaping が batch_summary に含まれる (CQ-0143)"""
        results = [self._make_result_with_rc(tmp_path, 42, include_shaping=True)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        assert "reward_shaping" in run_entry
        sd = run_entry["reward_shaping"]["shanten_delta"]
        assert sd["enabled"] is True
        assert sd["scale"] == 0.01
        assert sd["mode"] == "both"
        assert sd["schedule_type"] == "constant"

    def test_reward_shaping_mixed_no_crash(self, tmp_path: Path):
        """reward_shaping あり/なし混在でもクラッシュしない (CQ-0143)"""
        results = [self._make_result_with_rc(tmp_path, 42, include_shaping=True)]
        # seed 43: reward_shaping なし
        run_dir_43 = tmp_path / "fake_run_43"
        run_dir_43.mkdir(parents=True, exist_ok=True)
        with open(run_dir_43 / "summary.json", "w") as f:
            json.dump({"phase_stats": {"selfplay": {"total_steps": 100}}}, f)
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
        assert "reward_shaping" in summary["runs"][0]
        assert "reward_shaping" not in summary["runs"][1]


class TestJointImitationBatch:
    """joint imitation の batch レポート統合テスト (CQ-0150, CQ-0152)"""

    def _make_result_with_joint_imi(self, tmp_path: Path, seed: int,
                                      value_loss: float = 0.05,
                                      enabled: bool = True):
        run_dir = tmp_path / f"fake_run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_data = {
            "phase_stats": {
                "imitation": {
                    "teacher_top1_match_rate": 0.5,
                    "teacher_best_set_hit_rate": 0.7,
                    "imitation_loss_mode": "strict_top1",
                    "value_loss": value_loss,
                    "imitation_value_warmstart": {
                        "enabled": enabled,
                        "coef": 0.5,
                    },
                },
            },
            "model_features": {
                "value_features": {
                    "current_shanten": {"enabled": True},
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

    def test_imitation_value_loss_in_runs(self, tmp_path: Path):
        """runs[*].imitation_metrics に value_loss と imitation_value_warmstart が含まれる"""
        results = [self._make_result_with_joint_imi(tmp_path, 42, value_loss=0.05)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        assert "imitation_metrics" in run_entry
        im = run_entry["imitation_metrics"]
        assert im["value_loss"] == 0.05
        assert im["imitation_value_warmstart"]["enabled"] is True
        assert im["imitation_value_warmstart"]["coef"] == 0.5
        # model_features も転記される
        assert "model_features" in run_entry
        assert run_entry["model_features"]["value_features"]["current_shanten"]["enabled"] is True
        # aggregate にも value_loss が集約される
        agg = summary["aggregate"]
        assert "imitation" in agg
        assert "value_loss" in agg["imitation"]
        assert agg["imitation"]["value_loss"]["count"] == 1

    def test_imitation_mixed_no_crash(self, tmp_path: Path):
        """joint imitation on/off 混在でも集約が落ちない"""
        results = [
            self._make_result_with_joint_imi(tmp_path, 42, value_loss=0.05, enabled=True),
        ]
        # seed 43: joint imitation off (value_loss なし)
        run_dir_43 = tmp_path / "fake_run_43"
        run_dir_43.mkdir(parents=True, exist_ok=True)
        summary_data = {
            "phase_stats": {
                "imitation": {
                    "teacher_top1_match_rate": 0.4,
                    "teacher_best_set_hit_rate": 0.6,
                    "imitation_loss_mode": "strict_top1",
                },
            },
        }
        with open(run_dir_43 / "summary.json", "w") as f:
            json.dump(summary_data, f)
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
        # クラッシュしない
        assert summary["success_count"] == 2
        # imitation 集約は 2 件
        agg = summary["aggregate"]
        assert "imitation" in agg
        assert agg["imitation"]["teacher_top1_match_rate"]["count"] == 2
        # value_loss は 1 件のみ（seed 43 にはなし）
        assert agg["imitation"]["value_loss"]["count"] == 1


class TestCurrentShantenBatch:
    """current_shanten model_features の batch レポート統合テスト (CQ-0154)"""

    def _make_result_with_model_features(self, tmp_path: Path, seed: int,
                                          cs_enabled: bool = True):
        run_dir = tmp_path / f"fake_run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_data = {
            "phase_stats": {
                "imitation": {
                    "teacher_top1_match_rate": 0.5,
                    "teacher_best_set_hit_rate": 0.7,
                    "imitation_loss_mode": "strict_top1",
                    "value_loss": 0.05,
                    "imitation_value_warmstart": {"enabled": True, "coef": 0.5},
                },
            },
            "model_features": {
                "value_features": {
                    "current_shanten": {"enabled": cs_enabled},
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

    def test_model_features_in_runs(self, tmp_path: Path):
        """runs[*].model_features に current_shanten.enabled が含まれる"""
        results = [self._make_result_with_model_features(tmp_path, 42, cs_enabled=True)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        assert "model_features" in run_entry
        vf = run_entry["model_features"]["value_features"]
        assert vf["current_shanten"]["enabled"] is True

    def test_model_features_mixed_no_crash(self, tmp_path: Path):
        """current_shanten on/off 混在でも batch 集約が落ちない"""
        results = [
            self._make_result_with_model_features(tmp_path, 42, cs_enabled=True),
        ]
        # seed 43: model_features なし
        run_dir_43 = tmp_path / "fake_run_43"
        run_dir_43.mkdir(parents=True, exist_ok=True)
        summary_data = {
            "phase_stats": {
                "imitation": {
                    "teacher_top1_match_rate": 0.4,
                    "teacher_best_set_hit_rate": 0.6,
                    "imitation_loss_mode": "strict_top1",
                },
            },
        }
        with open(run_dir_43 / "summary.json", "w") as f:
            json.dump(summary_data, f)
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
        # クラッシュしない
        assert summary["success_count"] == 2
        # seed 42 には model_features あり、seed 43 にはなし
        assert "model_features" in summary["runs"][0]
        assert "model_features" not in summary["runs"][1]


class TestTurnDiagBatch:
    """CQ-0156: turn_diag の batch レポートテスト"""

    def _make_result_with_turn_diag(self, tmp_path: Path, seed: int,
                                     include_turn_diag: bool = True):
        run_dir = tmp_path / f"fake_run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ppo_diag = {
            "advantage_mean": 0.01,
            "clip_fraction": 0.05,
            "ratio_mean": 1.0,
        }
        if include_turn_diag:
            ppo_diag["turn_diag"] = {
                "total_samples": 90,
                "early": {"count": 30, "advantage": {"mean": 0.02}},
                "mid": {"count": 30, "advantage": {"mean": -0.01}},
                "late": {"count": 30, "advantage": {"mean": 0.0}},
            }
        summary_data = {
            "phase_stats": {"learner": {"ppo_diag": ppo_diag}},
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

    def test_turn_diag_in_runs(self, tmp_path: Path):
        """runs[*].learner_diag.turn_diag が参照可能"""
        results = [self._make_result_with_turn_diag(tmp_path, 42)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        assert "learner_diag" in run_entry
        assert "turn_diag" in run_entry["learner_diag"]
        td = run_entry["learner_diag"]["turn_diag"]
        assert td["early"]["count"] == 30

    def test_turn_diag_mixed_no_crash(self, tmp_path: Path):
        """turn_diag あり/なし混在でクラッシュしない"""
        results = [
            self._make_result_with_turn_diag(tmp_path, 42, include_turn_diag=True),
            self._make_result_with_turn_diag(tmp_path, 43, include_turn_diag=False),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert summary["success_count"] == 2
        # seed 42 has turn_diag, seed 43 doesn't
        assert "turn_diag" in summary["runs"][0]["learner_diag"]
        assert "turn_diag" not in summary["runs"][1]["learner_diag"]


class TestTowerBatch:
    """CQ-0158: tower 構造の batch レポートテスト"""

    def _make_result_with_tower(self, tmp_path: Path, seed: int,
                                 pt_enabled: bool = False, vt_enabled: bool = False):
        run_dir = tmp_path / f"fake_run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_data = {
            "phase_stats": {},
            "model_features": {
                "value_features": {"current_shanten": {"enabled": True}},
                "policy_tower": {"enabled": pt_enabled, "hidden_dim": 16 if pt_enabled else None},
                "value_tower": {"enabled": vt_enabled, "hidden_dim": 16 if vt_enabled else None},
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

    def test_tower_in_runs(self, tmp_path: Path):
        """runs[*].model_features から tower 条件が読める"""
        results = [self._make_result_with_tower(tmp_path, 42, pt_enabled=True, vt_enabled=True)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        assert "model_features" in run_entry
        mf = run_entry["model_features"]
        assert mf["policy_tower"]["enabled"] is True
        assert mf["policy_tower"]["hidden_dim"] == 16
        assert mf["value_tower"]["enabled"] is True
        assert mf["value_tower"]["hidden_dim"] == 16

    def test_tower_mixed_no_crash(self, tmp_path: Path):
        """tower あり/なし混在で batch 集約が落ちない"""
        results = [
            self._make_result_with_tower(tmp_path, 42, pt_enabled=True, vt_enabled=True),
        ]
        # seed 43: model_features なし (旧形式)
        run_dir_43 = tmp_path / "fake_run_43"
        run_dir_43.mkdir(parents=True, exist_ok=True)
        summary_data = {"phase_stats": {}}
        with open(run_dir_43 / "summary.json", "w") as f:
            json.dump(summary_data, f)
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
        assert summary["success_count"] == 2
        assert "model_features" in summary["runs"][0]
        assert "model_features" not in summary["runs"][1]


class TestRewardComponentDiagBatch:
    """CQ-0160/CQ-0161: reward 成分 shanten_diag の batch レポートテスト"""

    def _make_result_with_reward_diag(self, tmp_path: Path, seed: int,
                                       include_reward_keys: bool = True):
        run_dir = tmp_path / f"fake_run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        improve = {"count": 10, "advantage": {"mean": 0.02}}
        if include_reward_keys:
            improve["reward"] = {"mean": 0.01, "std": 0.005, "p50": 0.01, "p90": 0.02, "p99": 0.03}
            improve["point_delta_reward"] = {"mean": 0.008, "std": 0.003, "p50": 0.008, "p90": 0.015, "p99": 0.02}
            improve["shanten_delta_reward"] = {"mean": 0.002, "std": 0.001, "p50": 0.002, "p90": 0.003, "p99": 0.005}
            improve["delta_t"] = {"mean": -0.01, "std": 0.02, "p50": -0.005, "p90": 0.01, "p99": 0.03}
        ppo_diag = {
            "advantage_mean": 0.01,
            "clip_fraction": 0.05,
            "ratio_mean": 1.0,
            "shanten_diag": {
                "status": "complete",
                "total_samples": 30,
                "available_samples": 30,
                "unavailable_samples": 0,
                "improve": improve,
                "same": {"count": 10, "advantage": {"mean": 0.0}},
                "worsen": {"count": 10, "advantage": {"mean": -0.01}},
            },
        }
        summary_data = {
            "phase_stats": {"learner": {"ppo_diag": ppo_diag}},
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

    def test_reward_diag_in_runs(self, tmp_path: Path):
        """runs[*].learner_diag.shanten_diag から reward/delta_t が参照可能"""
        results = [self._make_result_with_reward_diag(tmp_path, 42)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        sd = run_entry["learner_diag"]["shanten_diag"]
        improve = sd["improve"]
        for key in ("reward", "point_delta_reward", "shanten_delta_reward", "delta_t"):
            assert key in improve, f"improve に {key} がない"

    def test_reward_diag_mixed_no_crash(self, tmp_path: Path):
        """reward キーあり/なし混在でクラッシュしない"""
        results = [
            self._make_result_with_reward_diag(tmp_path, 42, include_reward_keys=True),
            self._make_result_with_reward_diag(tmp_path, 43, include_reward_keys=False),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert summary["success_count"] == 2
        # seed 42 has reward keys
        sd_42 = summary["runs"][0]["learner_diag"]["shanten_diag"]
        assert "reward" in sd_42["improve"]
        # seed 43 doesn't
        sd_43 = summary["runs"][1]["learner_diag"]["shanten_diag"]
        assert "reward" not in sd_43["improve"]


class TestPostRiichiDiagBatch:
    """CQ-0163/CQ-0165: 立直後打牌統計の batch レポートテスト"""

    def _make_result_with_post_riichi(self, tmp_path: Path, seed: int,
                                       include_post_riichi: bool = True):
        run_dir = tmp_path / f"fake_run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        improve = {"count": 10, "advantage": {"mean": 0.02}}
        same = {"count": 15, "advantage": {"mean": 0.01}}
        if include_post_riichi:
            improve["post_riichi_discard_count"] = 2
            improve["post_riichi_discard_ratio"] = 0.2
            same["post_riichi_discard_count"] = 8
            same["post_riichi_discard_ratio"] = 0.533
        ppo_diag = {
            "advantage_mean": 0.01,
            "clip_fraction": 0.05,
            "ratio_mean": 1.0,
            "shanten_diag": {
                "status": "complete",
                "total_samples": 30,
                "available_samples": 30,
                "unavailable_samples": 0,
                "total_post_riichi_discards": 12 if include_post_riichi else None,
                "available_post_riichi_discards": 12 if include_post_riichi else None,
                "improve": improve,
                "same": same,
                "worsen": {"count": 5, "advantage": {"mean": -0.01}},
            },
        }
        summary_data = {
            "phase_stats": {"learner": {"ppo_diag": ppo_diag}},
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

    def test_post_riichi_in_runs(self, tmp_path: Path):
        """runs[*].learner_diag.shanten_diag に post_riichi が参照可能"""
        results = [self._make_result_with_post_riichi(tmp_path, 42)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        sd = run_entry["learner_diag"]["shanten_diag"]
        assert sd["total_post_riichi_discards"] == 12
        assert sd["same"]["post_riichi_discard_count"] == 8
        assert sd["same"]["post_riichi_discard_ratio"] == pytest.approx(0.533, abs=0.01)

    def test_post_riichi_mixed_no_crash(self, tmp_path: Path):
        """post_riichi あり/なし混在でクラッシュしない"""
        results = [
            self._make_result_with_post_riichi(tmp_path, 42, include_post_riichi=True),
            self._make_result_with_post_riichi(tmp_path, 43, include_post_riichi=False),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert summary["success_count"] == 2


class TestLearnerAuxStatsBatch:
    """CQ-0166/CQ-0167: learner 補助統計の summary / batch 転送テスト"""

    def _make_result_with_exclusion(self, tmp_path: Path, seed: int,
                                     include_exclusion: bool = True):
        run_dir = tmp_path / f"fake_run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        learner_stats = {
            "total_steps": 100,
            "num_updates": 5,
            "policy_loss": 0.01,
            "value_loss": 0.02,
            "mode": "ppo",
            "ppo_diag": {
                "advantage_mean": 0.01,
                "clip_fraction": 0.05,
                "ratio_mean": 1.0,
            },
        }
        if include_exclusion:
            learner_stats["post_riichi_exclusion"] = {
                "total_before_exclusion": 120,
                "excluded_post_riichi_discards": 20,
                "used_samples": 100,
            }
        summary_data = {
            "phase_stats": {"learner": learner_stats},
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

    def test_exclusion_in_batch_runs(self, tmp_path: Path):
        """runs[*].post_riichi_exclusion が batch_summary に載る"""
        results = [self._make_result_with_exclusion(tmp_path, 42)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run_entry = summary["runs"][0]
        assert "post_riichi_exclusion" in run_entry
        exc = run_entry["post_riichi_exclusion"]
        assert exc["excluded_post_riichi_discards"] == 20
        assert exc["used_samples"] == 100

    def test_exclusion_missing_no_crash(self, tmp_path: Path):
        """exclusion なしの run が混在してもクラッシュしない"""
        results = [
            self._make_result_with_exclusion(tmp_path, 42, include_exclusion=True),
            self._make_result_with_exclusion(tmp_path, 43, include_exclusion=False),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert summary["success_count"] == 2
        # seed 42 has exclusion
        assert "post_riichi_exclusion" in summary["runs"][0]
        # seed 43 doesn't
        assert "post_riichi_exclusion" not in summary["runs"][1]

    def test_no_exclusion_at_all(self, tmp_path: Path):
        """全 run で exclusion なしでもクラッシュしない"""
        results = [
            self._make_result_with_exclusion(tmp_path, 42, include_exclusion=False),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert summary["success_count"] == 1
        assert "post_riichi_exclusion" not in summary["runs"][0]


class TestEncoderFeaturesBatch:
    """encoder_features の batch_summary 転送テスト (CQ-0171)"""

    @staticmethod
    def _make_result_with_encoder(tmp_path: Path, seed: int,
                                   encoder_features: dict) -> dict:
        run_dir = tmp_path / f"run_{seed}"
        run_dir.mkdir()
        summary = {
            "encoder_features": encoder_features,
            "phase_stats": {},
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f)
        return {
            "seed": seed,
            "success": True,
            "result": {"avg_rank": 2.5, "run_dir": str(run_dir)},
        }

    def test_encoder_features_transferred(self, tmp_path: Path):
        """encoder_features が batch_summary に転送される"""
        ef = {
            "name": "FlatFeatureEncoder",
            "observation_mode": "full",
            "shanten_hint": True,
            "discard_ukeire_hint": True,
            "current_shanten": False,
            "shape_hint": True,
            "input_dim": 555,
        }
        results = [self._make_result_with_encoder(tmp_path, 42, ef)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run = summary["runs"][0]
        assert "encoder_features" in run
        assert run["encoder_features"]["discard_ukeire_hint"] is True
        assert run["encoder_features"]["current_shanten"] is False
        assert run["encoder_features"]["shape_hint"] is True

    def test_turn_context_transferred(self, tmp_path: Path):
        """turn_context が batch_summary に転送される (CQ-0177)"""
        ef = {
            "name": "FlatFeatureEncoder",
            "observation_mode": "full",
            "turn_context": True,
            "input_dim": 459,
        }
        results = [self._make_result_with_encoder(tmp_path, 42, ef)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        run = summary["runs"][0]
        assert run["encoder_features"]["turn_context"] is True

    def test_no_encoder_features_ok(self, tmp_path: Path):
        """encoder_features がない summary でもクラッシュしない"""
        run_dir = tmp_path / "run_42"
        run_dir.mkdir()
        with open(run_dir / "summary.json", "w") as f:
            json.dump({"phase_stats": {}}, f)
        results = [{
            "seed": 42,
            "success": True,
            "result": {"avg_rank": 2.5, "run_dir": str(run_dir)},
        }]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert "encoder_features" not in summary["runs"][0]


class TestEvalBeforeAndPhaseTimingBatch:
    """eval_before / phase_timing の batch_summary 転送・集約テスト (CQ-0174)"""

    @staticmethod
    def _make_result(tmp_path, seed, eval_before=None, phase_timing=None):
        run_dir = tmp_path / f"run_{seed}"
        run_dir.mkdir(exist_ok=True)
        summary = {"phase_stats": {}}
        if eval_before is not None:
            summary["phase_stats"]["eval_before"] = eval_before
        if phase_timing is not None:
            summary["phase_timing"] = phase_timing
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f)
        return {
            "seed": seed,
            "success": True,
            "result": {"avg_rank": 2.5, "run_dir": str(run_dir)},
        }

    def test_eval_before_transferred_and_aggregated(self, tmp_path: Path):
        """eval_before が runs に転送され aggregate に集約される"""
        results = [
            self._make_result(tmp_path, 42, eval_before={
                "avg_rank": 3.0, "avg_score": -10000,
                "win_rate": 0.1, "deal_in_rate": 0.5,
            }),
            self._make_result(tmp_path, 43, eval_before={
                "avg_rank": 2.5, "avg_score": -5000,
                "win_rate": 0.2, "deal_in_rate": 0.4,
            }),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)

        # runs に転送
        for run in summary["runs"]:
            assert "eval_before" in run
        assert summary["runs"][0]["eval_before"]["avg_rank"] == 3.0

        # aggregate に集約
        agg_eb = summary["aggregate"].get("eval_before")
        assert agg_eb is not None, "aggregate.eval_before がない"
        assert "avg_rank" in agg_eb
        assert agg_eb["avg_rank"]["count"] == 2
        assert agg_eb["avg_rank"]["mean"] == pytest.approx(2.75)

    def test_phase_timing_transferred_and_aggregated(self, tmp_path: Path):
        """phase_timing が runs に転送され aggregate に集約される"""
        results = [
            self._make_result(tmp_path, 42, phase_timing={
                "selfplay": {"start_ts": "t", "end_ts": "t", "duration_sec": 10.0},
                "learner": {"start_ts": "t", "end_ts": "t", "duration_sec": 2.0},
                "eval": {"start_ts": "t", "end_ts": "t", "duration_sec": 30.0},
            }),
            self._make_result(tmp_path, 43, phase_timing={
                "selfplay": {"start_ts": "t", "end_ts": "t", "duration_sec": 12.0},
                "learner": {"start_ts": "t", "end_ts": "t", "duration_sec": 3.0},
                "eval": {"start_ts": "t", "end_ts": "t", "duration_sec": 28.0},
            }),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)

        # runs に転送
        assert "phase_timing" in summary["runs"][0]

        # aggregate に集約
        agg_pt = summary["aggregate"].get("phase_timing")
        assert agg_pt is not None, "aggregate.phase_timing がない"
        assert "selfplay" in agg_pt
        assert agg_pt["selfplay"]["mean"] == pytest.approx(11.0)
        assert agg_pt["selfplay"]["count"] == 2
        assert "learner" in agg_pt
        assert "eval" in agg_pt
        # total
        assert "total" in agg_pt
        assert agg_pt["total"]["mean"] == pytest.approx(42.5)

    def test_no_eval_before_ok(self, tmp_path: Path):
        """eval_before がない run でもクラッシュしない"""
        results = [self._make_result(tmp_path, 42)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert "eval_before" not in summary["runs"][0]
        assert "eval_before" not in summary["aggregate"]

    def test_no_phase_timing_ok(self, tmp_path: Path):
        """phase_timing がない run でもクラッシュしない"""
        results = [self._make_result(tmp_path, 42)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert "phase_timing" not in summary["runs"][0]
        assert "phase_timing" not in summary["aggregate"]


class TestCycleMetricsBatch:
    """cycle 別メトリクスの batch_summary 転送・集約テスト (CQ-0180)"""

    @staticmethod
    def _make_result_with_cycles(tmp_path, seed, cycles):
        run_dir = tmp_path / f"run_{seed}"
        run_dir.mkdir(exist_ok=True)
        summary = {
            "phase_stats": {"cycles": cycles},
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f)
        return {
            "seed": seed,
            "success": True,
            "result": {"avg_rank": 2.5, "run_dir": str(run_dir)},
        }

    def test_cycles_transferred(self, tmp_path: Path):
        """cycles が runs に転送される"""
        cycles = [
            {"cycle_index": 0, "eval": {"avg_rank": 3.0},
             "train_metrics": {"policy_loss": 0.1}},
            {"cycle_index": 1, "eval": {"avg_rank": 2.5},
             "train_metrics": {"policy_loss": 0.05}},
        ]
        results = [self._make_result_with_cycles(tmp_path, 42, cycles)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert "cycles" in summary["runs"][0]
        assert len(summary["runs"][0]["cycles"]) == 2

    def test_cycles_aggregate(self, tmp_path: Path):
        """cycle 別 aggregate が生成される"""
        cycles_a = [
            {"cycle_index": 0,
             "eval": {"avg_rank": 3.0, "avg_score": -10000},
             "eval_before": {"avg_rank": 3.5},
             "eval_diff": {"avg_rank": {"delta": -0.5}},
             "learner_diag": {"clip_fraction": 0.05, "ratio_std": 0.1}},
            {"cycle_index": 1,
             "eval": {"avg_rank": 2.5, "avg_score": -5000},
             "eval_diff": {"avg_rank": {"delta": -0.3}},
             "learner_diag": {"clip_fraction": 0.03, "ratio_std": 0.08}},
        ]
        cycles_b = [
            {"cycle_index": 0,
             "eval": {"avg_rank": 2.8, "avg_score": -8000},
             "eval_diff": {"avg_rank": {"delta": -0.4}},
             "learner_diag": {"clip_fraction": 0.04, "ratio_std": 0.12}},
            {"cycle_index": 1,
             "eval": {"avg_rank": 2.3, "avg_score": -3000},
             "eval_diff": {"avg_rank": {"delta": -0.2}},
             "learner_diag": {"clip_fraction": 0.02, "ratio_std": 0.06}},
        ]
        results = [
            self._make_result_with_cycles(tmp_path, 42, cycles_a),
            self._make_result_with_cycles(tmp_path, 43, cycles_b),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        agg_cycles = summary["aggregate"].get("cycles")
        assert agg_cycles is not None
        assert len(agg_cycles) == 2
        # cycle 0
        assert agg_cycles[0]["cycle_index"] == 0
        assert agg_cycles[0]["count"] == 2
        assert "eval" in agg_cycles[0]
        assert agg_cycles[0]["eval"]["avg_rank"]["count"] == 2
        assert agg_cycles[0]["eval"]["avg_rank"]["mean"] == pytest.approx(2.9)
        # eval_diff
        assert "eval_diff_avg_rank" in agg_cycles[0]
        assert agg_cycles[0]["eval_diff_avg_rank"]["mean"] == pytest.approx(-0.45)
        # learner_diag
        assert "learner_diag" in agg_cycles[0]
        assert agg_cycles[0]["learner_diag"]["clip_fraction"]["mean"] == pytest.approx(0.045)

    def test_no_cycles_ok(self, tmp_path: Path):
        """cycles がない run でもクラッシュしない"""
        run_dir = tmp_path / "run_42"
        run_dir.mkdir()
        with open(run_dir / "summary.json", "w") as f:
            json.dump({"phase_stats": {}}, f)
        results = [{
            "seed": 42,
            "success": True,
            "result": {"avg_rank": 2.5, "run_dir": str(run_dir)},
        }]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert "cycles" not in summary["runs"][0]
        assert "cycles" not in summary["aggregate"]


class TestPolicyAnchorBatch:
    """policy_anchor の batch 転送テスト (CQ-0183)"""

    @staticmethod
    def _make_result_with_anchor(tmp_path, seed, policy_anchor):
        run_dir = tmp_path / f"run_{seed}"
        run_dir.mkdir(exist_ok=True)
        summary = {
            "phase_stats": {
                "learner": {
                    "ppo_diag": {
                        "clip_fraction": 0.05,
                        "policy_anchor": policy_anchor,
                    },
                },
            },
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f)
        return {
            "seed": seed,
            "success": True,
            "result": {"avg_rank": 2.5, "run_dir": str(run_dir)},
        }

    def test_anchor_in_learner_diag(self, tmp_path: Path):
        """policy_anchor が learner_diag に転送される"""
        pa = {"enabled": True, "type": "kl", "coef": 0.1,
              "anchor_loss_mean": 0.05, "anchor_kl_mean": 0.05}
        results = [self._make_result_with_anchor(tmp_path, 42, pa)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        diag = summary["runs"][0].get("learner_diag", {})
        assert "policy_anchor" in diag
        assert diag["policy_anchor"]["enabled"] is True
        assert diag["policy_anchor"]["anchor_loss_mean"] == 0.05

    def test_mixed_anchor_no_crash(self, tmp_path: Path):
        """anchor あり/なしが混在してもクラッシュしない"""
        pa_on = {"enabled": True, "type": "kl", "coef": 0.1,
                 "anchor_loss_mean": 0.05}
        pa_off = {"enabled": False}
        results = [
            self._make_result_with_anchor(tmp_path, 42, pa_on),
            self._make_result_with_anchor(tmp_path, 43, pa_off),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert summary["success_count"] == 2


class TestCycleAggregateExtended:
    """cycle 集約拡張テスト (CQ-0189)"""

    @staticmethod
    def _make_result(tmp_path, seed, cycles):
        run_dir = tmp_path / f"run_{seed}"
        run_dir.mkdir(exist_ok=True)
        with open(run_dir / "summary.json", "w") as f:
            json.dump({"phase_stats": {"cycles": cycles}}, f)
        return {
            "seed": seed, "success": True,
            "result": {"avg_rank": 2.5, "run_dir": str(run_dir)},
        }

    def test_actor_type_counts_aggregated(self, tmp_path: Path):
        """actor_type_counts が cycle 集約に含まれる"""
        cycles_a = [{"cycle_index": 0,
                      "actor_type_counts": {"policy": 100, "baseline": 30}}]
        cycles_b = [{"cycle_index": 0,
                      "actor_type_counts": {"policy": 80, "baseline": 50}}]
        results = [
            self._make_result(tmp_path, 42, cycles_a),
            self._make_result(tmp_path, 43, cycles_b),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        agg = summary["aggregate"]["cycles"][0]
        assert "actor_type_counts" in agg
        assert agg["actor_type_counts"]["policy"]["mean"] == pytest.approx(90.0)
        assert agg["actor_type_counts"]["baseline"]["mean"] == pytest.approx(40.0)

    def test_learner_stages_aggregated(self, tmp_path: Path):
        """learner_stages が cycle 集約に含まれる"""
        cycles_a = [{"cycle_index": 0, "learner_stages": {
            "baseline_imitation": {"executed": True, "used_samples": 30, "policy_loss": 0.1},
            "policy_ppo": {"executed": True, "used_samples": 70, "policy_loss": 0.05},
        }}]
        cycles_b = [{"cycle_index": 0, "learner_stages": {
            "baseline_imitation": {"executed": True, "used_samples": 50, "policy_loss": 0.08},
            "policy_ppo": {"executed": True, "used_samples": 80, "policy_loss": 0.04},
        }}]
        results = [
            self._make_result(tmp_path, 42, cycles_a),
            self._make_result(tmp_path, 43, cycles_b),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        ls = summary["aggregate"]["cycles"][0].get("learner_stages", {})
        assert "baseline_imitation" in ls
        assert ls["baseline_imitation"]["executed_count"] == 2
        assert ls["baseline_imitation"]["used_samples"]["mean"] == pytest.approx(40.0)
        assert "policy_ppo" in ls
        assert ls["policy_ppo"]["used_samples"]["mean"] == pytest.approx(75.0)

    def test_mixed_old_new_no_crash(self, tmp_path: Path):
        """旧run（新キーなし）混在でクラッシュしない"""
        cycles_new = [{"cycle_index": 0,
                        "actor_type_counts": {"policy": 100},
                        "learner_stages": {"policy_ppo": {"executed": True, "used_samples": 70}}}]
        cycles_old = [{"cycle_index": 0, "eval": {"avg_rank": 2.5}}]
        results = [
            self._make_result(tmp_path, 42, cycles_new),
            self._make_result(tmp_path, 43, cycles_old),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        assert len(summary["runs"]) == 2
        assert "cycles" in summary["aggregate"]


class TestTeacherAgreementBatch:
    """teacher_agreement の batch 転送・集約テスト (CQ-0200)"""

    @staticmethod
    def _make_result(tmp_path, seed, cycles):
        run_dir = tmp_path / f"run_{seed}"
        run_dir.mkdir(exist_ok=True)
        with open(run_dir / "summary.json", "w") as f:
            json.dump({"phase_stats": {"cycles": cycles}}, f)
        return {
            "seed": seed, "success": True,
            "result": {"avg_rank": 2.5, "run_dir": str(run_dir)},
        }

    def test_teacher_agreement_aggregated(self, tmp_path: Path):
        """teacher_agreement が cycle 集約に含まれる"""
        ta = {
            "enabled": True,
            "num_baseline_samples": 50,
            "num_best_set_samples": 40,
            "action_match_rate_before": 0.3,
            "action_match_rate_after": 0.5,
            "best_set_hit_rate_before": 0.4,
            "best_set_hit_rate_after": 0.6,
        }
        cycles_a = [{"cycle_index": 0,
                      "learner_diag": {"clip_fraction": 0.05, "teacher_agreement": ta}}]
        ta2 = dict(ta)
        ta2["action_match_rate_after"] = 0.7
        cycles_b = [{"cycle_index": 0,
                      "learner_diag": {"clip_fraction": 0.03, "teacher_agreement": ta2}}]
        results = [
            self._make_result(tmp_path, 42, cycles_a),
            self._make_result(tmp_path, 43, cycles_b),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        agg_ta = summary["aggregate"]["cycles"][0]["learner_diag"].get("teacher_agreement")
        assert agg_ta is not None
        assert agg_ta["action_match_rate_after"]["count"] == 2
        assert agg_ta["action_match_rate_after"]["mean"] == pytest.approx(0.6)

    def test_missing_teacher_agreement_no_crash(self, tmp_path: Path):
        """teacher_agreement がない run が混在しても壊れない"""
        cycles_new = [{"cycle_index": 0,
                        "learner_diag": {"clip_fraction": 0.05, "teacher_agreement": {
                            "enabled": True, "num_baseline_samples": 50,
                            "action_match_rate_before": 0.3, "action_match_rate_after": 0.5,
                        }}}]
        cycles_old = [{"cycle_index": 0,
                        "learner_diag": {"clip_fraction": 0.04}}]
        results = [
            self._make_result(tmp_path, 42, cycles_new),
            self._make_result(tmp_path, 43, cycles_old),
        ]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        # クラッシュしない
        assert summary["success_count"] == 2


class TestMultiChunkImitationBatch:
    """multi-chunk imitation の batch 転送テスト (CQ-0206)"""

    @staticmethod
    def _make_result(tmp_path, seed, imi_stats):
        run_dir = tmp_path / f"run_{seed}"
        run_dir.mkdir(exist_ok=True)
        summary = {"phase_stats": {"imitation": imi_stats}}
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f)
        return {
            "seed": seed, "success": True,
            "result": {"avg_rank": 2.5, "run_dir": str(run_dir)},
        }

    def test_chunks_transferred(self, tmp_path: Path):
        """chunks が imitation_metrics に転送される"""
        imi = {
            "teacher_top1_match_rate": 0.5,
            "teacher_best_set_hit_rate": 0.6,
            "multi_chunk_imitation": {
                "enabled": True, "num_chunks": 2,
                "imitation_matches_per_chunk": 100,
            },
            "chunks": [
                {"chunk_index": 0, "policy_loss": 0.1,
                 "teacher_top1_match_rate": 0.4},
                {"chunk_index": 1, "policy_loss": 0.05,
                 "teacher_top1_match_rate": 0.5},
            ],
        }
        results = [self._make_result(tmp_path, 42, imi)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        im = summary["runs"][0].get("imitation_metrics", {})
        assert "multi_chunk_imitation" in im
        assert im["multi_chunk_imitation"]["enabled"] is True
        assert "chunks" in im
        assert len(im["chunks"]) == 2

    def test_no_chunks_ok(self, tmp_path: Path):
        """chunks がない run でもクラッシュしない"""
        imi = {
            "teacher_top1_match_rate": 0.5,
        }
        results = [self._make_result(tmp_path, 42, imi)]
        generate_batch_report(tmp_path, results)

        with open(tmp_path / "batch_summary.json") as f:
            summary = json.load(f)
        im = summary["runs"][0].get("imitation_metrics", {})
        assert "chunks" not in im
        assert "multi_chunk_imitation" not in im
