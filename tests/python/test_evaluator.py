"""テスト: evaluator.py — 評価対戦ランナー"""
import pytest
import json
from pathlib import Path

from mahjong_rl.encoders import FlatFeatureEncoder
from mahjong_rl.models import MLPPolicyValueModel
from mahjong_rl.evaluator import EvaluationRunner, EvalMetrics


class TestEvaluationRunner:
    """EvaluationRunner テスト"""

    def test_evaluate_completes(self, tmp_path: Path):
        """評価対戦が完走して EvalMetrics を返す"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        metrics = runner.evaluate(num_matches=2, seed_start=42)

        assert isinstance(metrics, EvalMetrics)
        assert metrics.num_matches == 2
        assert metrics.num_rounds >= 2

    def test_metrics_types_and_ranges(self, tmp_path: Path):
        """metrics の型と値の範囲"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        metrics = runner.evaluate(num_matches=2, seed_start=0)

        assert 1.0 <= metrics.avg_rank <= 4.0
        assert isinstance(metrics.avg_score, float)
        assert 0.0 <= metrics.win_rate <= 1.0
        assert 0.0 <= metrics.deal_in_rate <= 1.0

    def test_save_to_eval_dir(self, tmp_path: Path):
        """eval ディレクトリに JSON が保存される"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        eval_dir = tmp_path / "eval"
        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        metrics = runner.evaluate(num_matches=1, seed_start=42, eval_dir=eval_dir)

        json_path = eval_dir / "eval_metrics.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert "avg_rank" in data
        assert "avg_score" in data
        assert "win_rate" in data
        assert "deal_in_rate" in data
