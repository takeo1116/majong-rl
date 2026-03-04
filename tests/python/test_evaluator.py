"""テスト: evaluator.py — 評価対戦ランナー"""
import pytest
import json
from pathlib import Path

from mahjong_rl.encoders import FlatFeatureEncoder
from mahjong_rl.models import MLPPolicyValueModel
from mahjong_rl.evaluator import (
    EvaluationRunner, EvalMetrics, RotationEvalResult,
    PartialEvalMetrics, aggregate_partials, compute_eval_diff,
    save_partial, load_partials, aggregate_and_save,
    aggregate_rotation_partials,
)


@pytest.mark.slow
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


@pytest.mark.slow
class TestPolicySeatConfig:
    """policy 席設定テスト (CQ-0044)"""

    def test_non_zero_policy_seat(self):
        """policy 席を 0 以外に設定して評価できる"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        metrics = runner.evaluate(num_matches=1, seed_start=42, policy_seats=[2])

        assert isinstance(metrics, EvalMetrics)
        assert metrics.num_matches == 1
        assert metrics.policy_seats == [2]
        assert 1.0 <= metrics.avg_rank <= 4.0

    def test_rotation_all_seats(self):
        """全席ローテーション評価が可能"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        metrics = runner.evaluate(
            num_matches=1, seed_start=42, policy_seats=[0, 1, 2, 3])

        assert metrics.num_matches == 4  # 1 match * 4 seats
        assert metrics.policy_seats == [0, 1, 2, 3]
        assert 1.0 <= metrics.avg_rank <= 4.0

    def test_seat_info_in_saved_json(self, tmp_path: Path):
        """eval 結果に席情報が記録される"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        eval_dir = tmp_path / "eval"
        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        runner.evaluate(
            num_matches=1, seed_start=0,
            eval_dir=eval_dir, policy_seats=[1, 3])

        with open(eval_dir / "eval_metrics.json") as f:
            data = json.load(f)
        assert data["policy_seats"] == [1, 3]

    def test_default_seats_backward_compatible(self):
        """デフォルト (policy_seats 未指定) は従来通り席0"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        metrics = runner.evaluate(num_matches=1, seed_start=42)

        assert metrics.policy_seats == [0]
        assert metrics.num_matches == 1


@pytest.mark.slow
class TestRotationEval:
    """席ローテーション集計テスト (CQ-0045)"""

    def test_rotation_returns_per_seat_and_aggregate(self):
        """ローテーション評価が席別と総合の両方を返す"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        result = runner.evaluate_rotation(num_matches=1, seed_start=42)

        assert isinstance(result, RotationEvalResult)
        assert len(result.per_seat) == 4
        for seat in [0, 1, 2, 3]:
            assert seat in result.per_seat
            assert isinstance(result.per_seat[seat], EvalMetrics)
        assert isinstance(result.aggregate, EvalMetrics)
        assert result.aggregate.policy_seats == [0, 1, 2, 3]

    def test_per_seat_metrics_valid(self):
        """席別指標が妥当な範囲"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        result = runner.evaluate_rotation(num_matches=1, seed_start=0)

        for seat, m in result.per_seat.items():
            assert 1.0 <= m.avg_rank <= 4.0
            assert 0.0 <= m.win_rate <= 1.0
            assert 0.0 <= m.deal_in_rate <= 1.0

    def test_aggregate_matches_total(self):
        """総合の num_matches が席別の合計と一致"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        result = runner.evaluate_rotation(num_matches=1, seed_start=42)

        total = sum(m.num_matches for m in result.per_seat.values())
        assert result.aggregate.num_matches == total

    def test_save_creates_per_seat_and_rotation_files(self, tmp_path: Path):
        """eval ディレクトリに席別と総合の JSON が保存される"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        eval_dir = tmp_path / "eval"
        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        runner.evaluate_rotation(
            num_matches=1, seed_start=42, eval_dir=eval_dir)

        # 席別ファイル
        for seat in [0, 1, 2, 3]:
            assert (eval_dir / f"eval_seat{seat}.json").exists()
        # 総合ファイル
        assert (eval_dir / "eval_rotation.json").exists()

        with open(eval_dir / "eval_rotation.json") as f:
            data = json.load(f)
        assert "avg_rank" in data
        assert data["policy_seats"] == [0, 1, 2, 3]

    def test_partial_seats_rotation(self):
        """一部席のみのローテーションが可能"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        result = runner.evaluate_rotation(
            num_matches=1, seed_start=42, seats=[0, 2])

        assert len(result.per_seat) == 2
        assert 0 in result.per_seat
        assert 2 in result.per_seat
        assert result.aggregate.policy_seats == [0, 2]

    def test_existing_evaluate_unaffected(self):
        """既存の単席 evaluate が壊れていない"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        metrics = runner.evaluate(num_matches=1, seed_start=42)

        assert isinstance(metrics, EvalMetrics)
        assert metrics.num_matches == 1


@pytest.mark.smoke
class TestComputeEvalDiff:
    """学習前後差分レポートテスト (CQ-0056)"""

    def test_diff_has_all_keys(self):
        """diff に主要 4 指標の差分が含まれる"""
        before = {"avg_rank": 2.5, "avg_score": -500.0,
                  "win_rate": 0.2, "deal_in_rate": 0.15}
        after = {"avg_rank": 2.3, "avg_score": 100.0,
                 "win_rate": 0.25, "deal_in_rate": 0.12}

        diff = compute_eval_diff(before, after)

        for key in ["avg_rank", "avg_score", "win_rate", "deal_in_rate"]:
            assert key in diff
            assert "before" in diff[key]
            assert "after" in diff[key]
            assert "delta" in diff[key]

    def test_diff_delta_correct(self):
        """delta が after - before と一致する"""
        before = {"avg_rank": 3.0, "avg_score": -1000.0,
                  "win_rate": 0.1, "deal_in_rate": 0.2}
        after = {"avg_rank": 2.5, "avg_score": 500.0,
                 "win_rate": 0.3, "deal_in_rate": 0.1}

        diff = compute_eval_diff(before, after)

        assert diff["avg_rank"]["delta"] == pytest.approx(-0.5)
        assert diff["avg_score"]["delta"] == pytest.approx(1500.0)
        assert diff["win_rate"]["delta"] == pytest.approx(0.2)
        assert diff["deal_in_rate"]["delta"] == pytest.approx(-0.1)

    def test_diff_preserves_eval_mode(self):
        """eval_mode が before/after に記録される"""
        before = {"avg_rank": 2.5, "avg_score": 0.0,
                  "win_rate": 0.2, "deal_in_rate": 0.1,
                  "eval_mode": "single"}
        after = {"avg_rank": 2.3, "avg_score": 100.0,
                 "win_rate": 0.25, "deal_in_rate": 0.12,
                 "eval_mode": "rotation"}

        diff = compute_eval_diff(before, after)

        assert diff["eval_mode_before"] == "single"
        assert diff["eval_mode_after"] == "rotation"

    def test_diff_missing_key_gives_none_delta(self):
        """before/after に欠損キーがある場合 delta は None"""
        before = {"avg_rank": 2.5}
        after = {"avg_rank": 2.3}

        diff = compute_eval_diff(before, after)

        assert diff["avg_rank"]["delta"] == pytest.approx(-0.2)
        assert diff["avg_score"]["delta"] is None
        assert diff["win_rate"]["delta"] is None


@pytest.mark.smoke
class TestPartialEvalMetrics:
    """PartialEvalMetrics / aggregate_partials テスト (CQ-0067)"""

    def test_partial_to_dict_round_trip(self):
        """to_dict → from_dict で復元できる"""
        partial = PartialEvalMetrics(
            sum_rank=10.0, sum_score=5000.0, wins=3, deal_ins=2,
            num_rounds=20, num_matches=5, policy_seats=[0], worker_id=0,
        )
        d = partial.to_dict()
        restored = PartialEvalMetrics.from_dict(d)
        assert restored.sum_rank == 10.0
        assert restored.sum_score == 5000.0
        assert restored.wins == 3
        assert restored.deal_ins == 2
        assert restored.num_rounds == 20
        assert restored.num_matches == 5
        assert restored.policy_seats == [0]
        assert restored.worker_id == 0

    def test_aggregate_single_partial(self):
        """単一 partial の集約が正しい"""
        partial = PartialEvalMetrics(
            sum_rank=8.0, sum_score=2000.0, wins=4, deal_ins=2,
            num_rounds=10, num_matches=4, policy_seats=[0],
        )
        metrics = aggregate_partials([partial])
        assert metrics.avg_rank == pytest.approx(2.0)
        assert metrics.avg_score == pytest.approx(500.0)
        assert metrics.win_rate == pytest.approx(0.4)
        assert metrics.deal_in_rate == pytest.approx(0.2)
        assert metrics.num_matches == 4
        assert metrics.num_rounds == 10

    def test_aggregate_multiple_partials(self):
        """複数 partial の集約が正しい"""
        p1 = PartialEvalMetrics(
            sum_rank=6.0, sum_score=1000.0, wins=2, deal_ins=1,
            num_rounds=5, num_matches=3, policy_seats=[0],
        )
        p2 = PartialEvalMetrics(
            sum_rank=9.0, sum_score=2000.0, wins=3, deal_ins=2,
            num_rounds=8, num_matches=3, policy_seats=[0],
        )
        metrics = aggregate_partials([p1, p2])
        assert metrics.avg_rank == pytest.approx(15.0 / 6)
        assert metrics.avg_score == pytest.approx(3000.0 / 6)
        assert metrics.win_rate == pytest.approx(5 / 13)
        assert metrics.deal_in_rate == pytest.approx(3 / 13)
        assert metrics.num_matches == 6
        assert metrics.num_rounds == 13

    def test_aggregate_empty_raises(self):
        """空リストで ValueError"""
        with pytest.raises(ValueError, match="空"):
            aggregate_partials([])

    def test_aggregate_preserves_policy_seats(self):
        """集約後に policy_seats が保持される"""
        p = PartialEvalMetrics(
            sum_rank=4.0, sum_score=1000.0, wins=1, deal_ins=0,
            num_rounds=5, num_matches=2, policy_seats=[1, 3],
        )
        metrics = aggregate_partials([p])
        assert metrics.policy_seats == [1, 3]

    def test_partial_save_load(self, tmp_path: Path):
        """partial を JSON に保存・読込できる"""
        partial = PartialEvalMetrics(
            sum_rank=6.0, sum_score=1500.0, wins=2, deal_ins=1,
            num_rounds=8, num_matches=3, policy_seats=[0], worker_id=1,
        )
        path = tmp_path / "partial.json"
        partial.save(path)
        loaded = PartialEvalMetrics.load(path)
        assert loaded.sum_rank == 6.0
        assert loaded.worker_id == 1

    def test_single_process_consistency(self):
        """単一プロセス evaluate と aggregate_partials の結果が一致する"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])
        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")

        # evaluate 経由
        metrics = runner.evaluate(num_matches=2, seed_start=42, policy_seats=[0])

        # evaluate_partial 経由
        partial = runner.evaluate_partial(num_matches=2, seed_start=42, policy_seats=[0])
        metrics_from_partial = aggregate_partials([partial])

        assert metrics.avg_rank == pytest.approx(metrics_from_partial.avg_rank)
        assert metrics.avg_score == pytest.approx(metrics_from_partial.avg_score)
        assert metrics.win_rate == pytest.approx(metrics_from_partial.win_rate)
        assert metrics.deal_in_rate == pytest.approx(metrics_from_partial.deal_in_rate)
        assert metrics.num_matches == metrics_from_partial.num_matches
        assert metrics.num_rounds == metrics_from_partial.num_rounds


@pytest.mark.smoke
class TestPartialSaveLoadAggregate:
    """partial 保存・読込・集約テスト (CQ-0068)"""

    def test_save_and_load_partials(self, tmp_path: Path):
        """save_partial → load_partials で復元できる"""
        partials_dir = tmp_path / "partials"
        p0 = PartialEvalMetrics(
            sum_rank=4.0, sum_score=500.0, wins=1, deal_ins=0,
            num_rounds=5, num_matches=2, policy_seats=[0], worker_id=0)
        p1 = PartialEvalMetrics(
            sum_rank=6.0, sum_score=1000.0, wins=2, deal_ins=1,
            num_rounds=5, num_matches=2, policy_seats=[0], worker_id=1)
        save_partial(p0, partials_dir, worker_id=0)
        save_partial(p1, partials_dir, worker_id=1)

        loaded = load_partials(partials_dir)
        assert len(loaded) == 2
        assert loaded[0].worker_id == 0
        assert loaded[1].worker_id == 1

    def test_aggregate_and_save_creates_json(self, tmp_path: Path):
        """aggregate_and_save が eval_metrics.json を生成する"""
        partials_dir = tmp_path / "partials"
        eval_dir = tmp_path / "eval"
        save_partial(
            PartialEvalMetrics(
                sum_rank=6.0, sum_score=1000.0, wins=2, deal_ins=1,
                num_rounds=8, num_matches=3, policy_seats=[0], worker_id=0),
            partials_dir, worker_id=0)
        save_partial(
            PartialEvalMetrics(
                sum_rank=8.0, sum_score=2000.0, wins=3, deal_ins=1,
                num_rounds=10, num_matches=4, policy_seats=[0], worker_id=1),
            partials_dir, worker_id=1)

        metrics = aggregate_and_save(partials_dir, eval_dir)
        assert (eval_dir / "eval_metrics.json").exists()
        assert metrics.num_matches == 7
        assert metrics.avg_rank == pytest.approx(14.0 / 7)

    def test_aggregate_rotation_partials(self, tmp_path: Path):
        """rotation eval の partial 集約が席別と総合を生成する"""
        partials_dir = tmp_path / "partials"
        eval_dir = tmp_path / "eval"

        # 席0 の partial (2 workers)
        save_partial(
            PartialEvalMetrics(
                sum_rank=4.0, sum_score=500.0, wins=1, deal_ins=0,
                num_rounds=5, num_matches=2, policy_seats=[0], worker_id=0),
            partials_dir, worker_id=0)
        save_partial(
            PartialEvalMetrics(
                sum_rank=6.0, sum_score=1000.0, wins=2, deal_ins=1,
                num_rounds=5, num_matches=2, policy_seats=[0], worker_id=1),
            partials_dir, worker_id=1)
        # 席1 の partial (2 workers)
        save_partial(
            PartialEvalMetrics(
                sum_rank=5.0, sum_score=800.0, wins=1, deal_ins=1,
                num_rounds=5, num_matches=2, policy_seats=[1], worker_id=2),
            partials_dir, worker_id=2)
        save_partial(
            PartialEvalMetrics(
                sum_rank=7.0, sum_score=1200.0, wins=2, deal_ins=0,
                num_rounds=5, num_matches=2, policy_seats=[1], worker_id=3),
            partials_dir, worker_id=3)

        result = aggregate_rotation_partials(partials_dir, eval_dir, seats=[0, 1])

        assert len(result.per_seat) == 2
        assert 0 in result.per_seat
        assert 1 in result.per_seat
        assert result.per_seat[0].num_matches == 4
        assert result.per_seat[1].num_matches == 4
        assert result.aggregate.num_matches == 8
        assert result.aggregate.policy_seats == [0, 1]

        # ファイル生成確認
        assert (eval_dir / "eval_seat0.json").exists()
        assert (eval_dir / "eval_seat1.json").exists()
        assert (eval_dir / "eval_rotation.json").exists()

    def test_partials_and_final_coexist(self, tmp_path: Path):
        """worker 部分結果と最終集約結果が両方 run ディレクトリに残る"""
        partials_dir = tmp_path / "eval" / "partials"
        eval_dir = tmp_path / "eval"
        save_partial(
            PartialEvalMetrics(
                sum_rank=4.0, sum_score=500.0, wins=1, deal_ins=0,
                num_rounds=5, num_matches=2, policy_seats=[0], worker_id=0),
            partials_dir, worker_id=0)

        aggregate_and_save(partials_dir, eval_dir)

        assert (partials_dir / "worker_0.json").exists()
        assert (eval_dir / "eval_metrics.json").exists()


@pytest.mark.slow
class TestEvaluatorDevice:
    """EvaluationRunner デバイス切替テスト (CQ-0064)"""

    def test_cpu_device_works(self):
        """CPU 明示指定で既存動作維持"""
        import torch
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])

        runner = EvaluationRunner(
            model=model, encoder=encoder, observation_mode="full",
            inference_device=torch.device("cpu"))
        metrics = runner.evaluate(num_matches=1, seed_start=42)

        assert isinstance(metrics, EvalMetrics)
        assert 1.0 <= metrics.avg_rank <= 4.0
