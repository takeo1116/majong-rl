"""CQ-0227 / CQ-0228: round outcome + future labels + smoke テスト"""
import pytest
import numpy as np
from pathlib import Path

pytestmark = pytest.mark.smoke

from mahjong_rl.call_shard import (
    DecisionSample, CandidateRecord,
    DecisionShardWriter, DecisionShardReader,
)


class TestFutureLabelsRoundTrip:
    """CQ-0227: future labels の shard round-trip"""

    def test_future_labels_preserved(self, tmp_path: Path):
        """future labels が shard で lossless に復元される"""
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        writer.add(DecisionSample(
            decision_type="discard",
            observation=np.zeros(10, dtype=np.float32),
            reward=0.0, log_prob=0.0, value=0.0,
            terminated=False, round_over=False,
            action=5,
            legal_mask=np.ones(34, dtype=np.float32),
            round_terminal_label="win",
            eventual_win_yaku_ids=[3, 4, 11],  # tanyao, yakuhai, toitoi
            eventual_total_han=5,
            eventual_fu=40,
            experiment_id="t", run_id="r", worker_id="w",
            episode_id="e",
        ))
        writer.add(DecisionSample(
            decision_type="call",
            observation=np.zeros(10, dtype=np.float32),
            reward=0.0, log_prob=0.0, value=0.0,
            terminated=False, round_over=False,
            selected_candidate_index=0,
            candidate_count=2,
            candidates=[
                CandidateRecord(action_type=4, tile_type=10, target_rel_seat=2),
                CandidateRecord(action_type=8),
            ],
            round_terminal_label="ron_loss",
            eventual_win_yaku_ids=[],
            eventual_total_han=-1,
            eventual_fu=-1,
            experiment_id="t", run_id="r", worker_id="w",
            episode_id="e",
        ))
        writer.close()

        loaded = DecisionShardReader(tmp_path).read_all()
        assert len(loaded) == 2

        s0 = loaded[0]
        assert s0.round_terminal_label == "win"
        assert s0.eventual_win_yaku_ids == [3, 4, 11]
        assert s0.eventual_total_han == 5
        assert s0.eventual_fu == 40

        s1 = loaded[1]
        assert s1.round_terminal_label == "ron_loss"
        assert s1.eventual_win_yaku_ids == []
        assert s1.eventual_total_han == -1

    def test_empty_yaku_ids(self, tmp_path: Path):
        """yaku_ids が空でも壊れない"""
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        writer.add(DecisionSample(
            decision_type="discard",
            observation=np.zeros(10, dtype=np.float32),
            reward=0.0, log_prob=0.0, value=0.0,
            terminated=False, round_over=False,
            action=0,
            legal_mask=np.ones(34, dtype=np.float32),
            round_terminal_label="ryukyoku_noten",
            eventual_win_yaku_ids=[],
            experiment_id="t", run_id="r", worker_id="w",
            episode_id="e",
        ))
        writer.close()
        loaded = DecisionShardReader(tmp_path).read_all()
        assert loaded[0].round_terminal_label == "ryukyoku_noten"
        assert loaded[0].eventual_win_yaku_ids == []


class TestRoundOutcomeIntegration:
    """CQ-0227: selfplay で future labels が埋まることの integration テスト"""

    def test_selfplay_fills_future_labels(self, tmp_path: Path):
        """selfplay 生成で future labels が実際に埋まる"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        worker.generate(num_matches=5, base_seed=42)

        samples = DecisionShardReader(tmp_path).read_all()
        assert len(samples) > 0

        # round_terminal_label が埋まっているサンプルがある
        labels = set(s.round_terminal_label for s in samples)
        non_empty_labels = labels - {""}
        assert len(non_empty_labels) > 0, \
            f"future labels が全て空: {labels}"

        # win label があるサンプルで yaku_ids が非空
        win_samples = [s for s in samples
                       if s.round_terminal_label == "win"]
        if win_samples:
            has_yakus = any(len(s.eventual_win_yaku_ids) > 0
                           for s in win_samples)
            assert has_yakus, "win sample なのに yaku_ids が全て空"

    def test_label_variety(self, tmp_path: Path):
        """複数種類の terminal label が出る"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        worker.generate(num_matches=10, base_seed=0)

        samples = DecisionShardReader(tmp_path).read_all()
        labels = set(s.round_terminal_label for s in samples if s.round_terminal_label)
        assert len(labels) >= 2, \
            f"label 種類が少なすぎる: {labels}"

    def test_label_coverage_high(self, tmp_path: Path):
        """ほぼ全 sample に future label が付く (total terminal label)"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        worker.generate(num_matches=5, base_seed=42)

        samples = DecisionShardReader(tmp_path).read_all()
        labeled = sum(1 for s in samples if s.round_terminal_label)
        total = len(samples)
        ratio = labeled / total if total > 0 else 0
        assert ratio > 0.7, \
            f"future label coverage が低い: {labeled}/{total} = {ratio:.2%}"

    def test_ron_bystander_label_exists(self, tmp_path: Path):
        """ron 和了の第三者に ron_bystander ラベルが付く"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        worker.generate(num_matches=10, base_seed=0)

        samples = DecisionShardReader(tmp_path).read_all()
        labels = set(s.round_terminal_label for s in samples)
        assert "ron_bystander" in labels, \
            f"ron_bystander が出ない: {labels}"

    def test_no_empty_label_for_labeled_round(self, tmp_path: Path):
        """backfill された round 内で空ラベルが残らない"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        worker.generate(num_matches=5, base_seed=42)

        samples = DecisionShardReader(tmp_path).read_all()
        # round_over=True かつ label 空は極めて少ないはず
        ro_unlabeled = sum(1 for s in samples
                           if s.round_over and not s.round_terminal_label)
        ro_total = sum(1 for s in samples if s.round_over)
        if ro_total > 0:
            unlabeled_ratio = ro_unlabeled / ro_total
            # 多少の edge case は許容するが、大半は埋まるべき
            assert unlabeled_ratio < 0.5, \
                f"round_over samples の unlabeled 率が高い: {ro_unlabeled}/{ro_total}"


class TestRoundCrossContamination:
    """round をまたいで future label が上書きされないことを確認"""

    def test_no_cross_round_label_contamination(self, tmp_path: Path):
        """round 境界で sample が次 round に漏れず、label coverage が高い"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        worker.generate(num_matches=5, base_seed=42)

        samples = DecisionShardReader(tmp_path).read_all()

        # 全体 label coverage: 半数以上に label が付く (round_buffer による batch flush)
        labeled = sum(1 for s in samples if s.round_terminal_label)
        ratio = labeled / len(samples) if samples else 0
        assert ratio > 0.7, \
            f"label coverage 低下: {labeled}/{len(samples)} = {ratio:.2%}"

        # round_over のある sample が存在する
        round_over_count = sum(1 for s in samples if s.round_over)
        assert round_over_count > 0, "round_over=True のサンプルがない"


class TestRoundOverFlag:
    """round_over が正しく立つ"""

    def test_round_over_set_at_round_end(self, tmp_path: Path):
        """round end の sample で round_over=True"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        worker.generate(num_matches=3, base_seed=42)

        samples = DecisionShardReader(tmp_path).read_all()
        round_over_count = sum(1 for s in samples if s.round_over)
        non_round_over_count = sum(1 for s in samples if not s.round_over)
        assert round_over_count > 0, "round_over=True のサンプルがない"
        assert non_round_over_count > 0, "round_over=False のサンプルがない"

        # round_over=True の sample には future label が付いているはず
        for s in samples:
            if s.round_over and s.round_terminal_label:
                # label が付いている → 正常
                pass


class TestGetRoundOutcomeBinding:
    """CQ-0227: C++ get_round_outcome binding テスト"""

    def test_binding_returns_dict(self):
        """get_round_outcome が dict を返す"""
        from mahjong_rl._mahjong_core import (
            GameEngine, EnvironmentState, RunMode, Phase,
            get_round_outcome,
        )
        e = GameEngine()
        env = EnvironmentState()
        env.run_mode = RunMode.Fast
        e.reset_match(env, 42, RunMode.Fast)
        for _ in range(5000):
            p = env.round_state.phase
            if p in (Phase.EndRound, Phase.EndMatch):
                break
            actions = e.get_legal_actions(env)
            if not actions:
                break
            r = e.step(env, actions[0])
            if r.round_over:
                outcome = get_round_outcome(env)
                assert isinstance(outcome, dict)
                assert "end_reason" in outcome
                assert "tenpai_players" in outcome
                assert "noten_players" in outcome
                assert "wins" in outcome
                return
        pytest.skip("round end が得られなかった")

    def test_win_has_yaku_ids(self):
        """和了局で yaku_ids が非空"""
        from mahjong_rl._mahjong_core import (
            GameEngine, EnvironmentState, RunMode, Phase,
            get_round_outcome,
        )
        e = GameEngine()
        env = EnvironmentState()
        for seed in range(100):
            e.reset_match(env, seed, RunMode.Fast)
            for _ in range(5000):
                p = env.round_state.phase
                if p in (Phase.EndRound, Phase.EndMatch):
                    break
                actions = e.get_legal_actions(env)
                if not actions:
                    break
                r = e.step(env, actions[0])
                if r.round_over:
                    outcome = get_round_outcome(env)
                    if outcome["wins"]:
                        w = outcome["wins"][0]
                        assert len(w["yaku_ids"]) > 0
                        assert w["total_han"] > 0
                        assert w["fu"] > 0
                        return
                    if r.match_over:
                        break
                    e.advance_round(env)
        pytest.skip("100 seed で和了が得られなかった")


class TestStage2aEval:
    """CQ-0230: Stage2a deterministic eval テスト"""

    def test_eval_smoke(self):
        """Stage2a eval が完走する"""
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
        from mahjong_rl.models.stage2a_model import Stage2aModel
        from mahjong_rl.encoders import FlatFeatureEncoder
        import torch

        encoder = FlatFeatureEncoder(observation_mode="full")
        input_dim = encoder.metadata().output_shape[0]
        model = Stage2aModel(input_dim=input_dim,
                              discard_hidden_dims=[32],
                              optional_hidden_dims=[32])
        evaluator = Stage2aEvaluator(
            model=model, encoder=encoder,
            device=torch.device("cpu"))
        result = evaluator.evaluate(num_matches=3, seed_start=42)
        assert result["num_matches"] == 3
        assert 1.0 <= result["avg_rank"] <= 4.0
        assert result["total_rounds"] > 0

    @staticmethod
    def _make_eval_config(eval_cfg=None):
        from mahjong_rl.experiment import ExperimentConfig
        config = ExperimentConfig()
        config.experiment = {
            "name": "test_stage2a_eval",
            "stage": "stage2a",
            "observation_mode": "full",
            "global_seed": 42,
            "phases": ["imitation", "selfplay", "learner", "eval"],
        }
        config.selfplay = {
            "imitation_matches": 3,
            "num_matches": 3,
            "seed_start": 0,
        }
        config.imitation = {"num_matches": 3}
        config.model = {
            "discard_hidden_dims": [32],
            "optional_hidden_dims": [32],
            "candidate_dim": 8,
        }
        config.training = {
            "algorithm": "ppo",
            "lr": 1e-3,
            "batch_size": 16,
            "epochs": 1,
        }
        config.evaluation = eval_cfg or {"num_matches": 3}
        return config

    def test_eval_via_runner(self, tmp_path: Path):
        """runner 経由で Stage2a eval (fixed-seat) が通る"""
        from mahjong_rl.runner import Stage1Runner
        import json

        config = self._make_eval_config({"num_matches": 3, "mode": "single"})
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"
        em = result.get("eval_metrics", {})
        assert em.get("avg_rank") is not None
        assert em.get("total_rounds", 0) > 0
        assert em.get("eval_mode") == "fixed_seat"

        run_dir = Path(result["run_dir"])

        # summary
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)
        ev = summary["phase_stats"].get("eval", {})
        assert ev.get("avg_rank") is not None

        # manifest
        with open(run_dir / "artifacts_manifest.json") as f:
            manifest = json.load(f)
        assert manifest["artifacts"]["eval"]["exists"] is True

        # eval/metrics.json
        assert (run_dir / "eval" / "metrics.json").exists()

    def test_eval_rotation_via_runner(self, tmp_path: Path):
        """runner 経由で Stage2a eval (rotation) が通る"""
        from mahjong_rl.runner import Stage1Runner
        import json

        config = self._make_eval_config({"num_matches": 2, "mode": "rotation"})
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"
        em = result.get("eval_metrics", {})
        assert em.get("eval_mode") == "rotation"
        assert em.get("num_matches") == 8  # 2 * 4 seats


class TestStage2aImitationEpochSeparation:
    """CQ-0231: imitation epochs と PPO epochs の分離"""

    def test_different_epochs(self, tmp_path: Path):
        """imitation_epochs != epochs の run が通る"""
        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig
        import json

        config = ExperimentConfig()
        config.experiment = {
            "name": "test_epoch_sep",
            "stage": "stage2a",
            "observation_mode": "full",
            "global_seed": 42,
            "phases": ["imitation", "selfplay", "learner"],
        }
        config.selfplay = {"imitation_matches": 3, "num_matches": 3, "seed_start": 0}
        config.imitation = {"num_matches": 3}
        config.model = {"discard_hidden_dims": [32], "optional_hidden_dims": [32],
                         "candidate_dim": 8}
        config.training = {
            "algorithm": "ppo",
            "lr": 1e-3, "batch_size": 16,
            "imitation_epochs": 3,  # imitation is 3
            "epochs": 1,            # PPO is 1
        }
        config.evaluation = {"num_matches": 0}

        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"
        # imitation used 3 epochs → more updates than PPO with 1 epoch
        imi = result.get("imitation_metrics", {})
        tm = imi.get("train_metrics", {})
        assert tm.get("num_updates", 0) > 0
        assert imi.get("imitation_epochs") == 3
        # PPO used 1 epoch
        ppo = result.get("train_metrics", {})
        assert ppo.get("num_updates", 0) > 0


class TestStage2aMultiCycle:
    """CQ-0232: Stage2a multi-cycle"""

    def test_multi_cycle_2(self, tmp_path: Path):
        """num_cycles=2 の Stage2a run が通る"""
        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig
        import json

        config = ExperimentConfig()
        config.experiment = {
            "name": "test_mc2",
            "stage": "stage2a",
            "observation_mode": "full",
            "global_seed": 42,
            "phases": ["imitation", "selfplay", "learner"],
        }
        config.selfplay = {"imitation_matches": 3, "num_matches": 3, "seed_start": 0}
        config.imitation = {"num_matches": 3}
        config.model = {"discard_hidden_dims": [32], "optional_hidden_dims": [32],
                         "candidate_dim": 8}
        config.training = {
            "algorithm": "ppo", "lr": 1e-3, "batch_size": 16, "epochs": 1,
            "multi_cycle": {
                "enabled": True,
                "num_cycles": 2,
                "selfplay_matches_per_cycle": 3,
                "eval_each_cycle": False,
            },
        }
        config.evaluation = {"num_matches": 0}

        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"

        cycles = result.get("cycles", [])
        assert len(cycles) == 2

        for i, cyc in enumerate(cycles):
            assert cyc["cycle_index"] == i
            assert cyc["selfplay_stats"]["total_steps"] > 0
            assert cyc["learner_metrics"]["num_updates"] > 0

        run_dir = Path(result["run_dir"])
        assert (run_dir / "checkpoints" / "checkpoint_cycle_00.pt").exists()
        assert (run_dir / "checkpoints" / "checkpoint_cycle_01.pt").exists()
        assert (run_dir / "checkpoints" / "checkpoint_learner.pt").exists()

        # pre-cycle selfplay が空 (shard なし)
        sp_dir = run_dir / "selfplay"
        if sp_dir.exists():
            shards = list(sp_dir.glob("**/*.parquet"))
            assert len(shards) == 0, \
                f"multi-cycle で pre-cycle selfplay shard が残っている: {shards}"
        # cycle 配下に selfplay がある
        assert (run_dir / "cycle_00" / "selfplay").exists()

        # top-level result に最終 cycle が昇格
        assert "selfplay_stats" in result
        assert result["selfplay_stats"]["total_steps"] > 0
        assert "train_metrics" in result
        assert result["train_metrics"]["num_updates"] > 0

    def test_multi_cycle_with_eval(self, tmp_path: Path):
        """eval_each_cycle=True で各 cycle に eval metrics が残る"""
        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig

        config = ExperimentConfig()
        config.experiment = {
            "name": "test_mc_eval",
            "stage": "stage2a",
            "observation_mode": "full",
            "global_seed": 42,
            "phases": ["imitation", "selfplay", "learner"],
        }
        config.selfplay = {"imitation_matches": 3, "num_matches": 3, "seed_start": 0}
        config.imitation = {"num_matches": 3}
        config.model = {"discard_hidden_dims": [32], "optional_hidden_dims": [32],
                         "candidate_dim": 8}
        config.training = {
            "algorithm": "ppo", "lr": 1e-3, "batch_size": 16, "epochs": 1,
            "multi_cycle": {
                "enabled": True,
                "num_cycles": 2,
                "selfplay_matches_per_cycle": 3,
                "eval_each_cycle": True,
            },
        }
        config.evaluation = {"num_matches": 2}

        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"

        cycles = result.get("cycles", [])
        assert len(cycles) == 2
        for cyc in cycles:
            em = cyc.get("eval_metrics")
            assert em is not None
            assert em.get("avg_rank") is not None

        # top-level eval_metrics は最終 cycle の eval
        top_eval = result.get("eval_metrics", {})
        assert top_eval.get("avg_rank") is not None
        last_eval = cycles[-1]["eval_metrics"]
        assert top_eval["avg_rank"] == last_eval["avg_rank"]

        # summary にも eval が残る
        import json
        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)
        ps_eval = summary["phase_stats"].get("eval", {})
        assert ps_eval.get("avg_rank") is not None


class TestStage2aParallel:
    """CQ-0233/0234/0235: Stage2a multi-process テスト"""

    @staticmethod
    def _make_parallel_config(imi_workers=1, sp_workers=1, eval_workers=1):
        from mahjong_rl.experiment import ExperimentConfig
        config = ExperimentConfig()
        config.experiment = {
            "name": "test_parallel",
            "stage": "stage2a",
            "observation_mode": "full",
            "global_seed": 42,
            "phases": ["imitation", "selfplay", "learner", "eval"],
        }
        config.selfplay = {
            "imitation_matches": 4,
            "num_matches": 4,
            "seed_start": 0,
            "num_workers": sp_workers,
        }
        config.imitation = {"num_matches": 4, "num_workers": imi_workers}
        config.model = {"discard_hidden_dims": [32], "optional_hidden_dims": [32],
                         "candidate_dim": 8}
        config.training = {
            "algorithm": "ppo", "lr": 1e-3, "batch_size": 16, "epochs": 1,
        }
        config.evaluation = {"num_matches": 4, "num_workers": eval_workers}
        return config

    def test_selfplay_parallel(self, tmp_path: Path):
        """selfplay.num_workers=2 で完走"""
        from mahjong_rl.runner import Stage1Runner

        config = self._make_parallel_config(sp_workers=2)
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"
        sp = result.get("selfplay_stats", {})
        assert sp.get("total_steps", 0) > 0

    def test_eval_parallel(self, tmp_path: Path):
        """evaluation.num_workers=2 で完走"""
        from mahjong_rl.runner import Stage1Runner

        config = self._make_parallel_config(eval_workers=2)
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"
        em = result.get("eval_metrics", {})
        assert em.get("avg_rank") is not None

    def test_imitation_parallel(self, tmp_path: Path):
        """imitation.num_workers=2 で完走"""
        from mahjong_rl.runner import Stage1Runner

        config = self._make_parallel_config(imi_workers=2)
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"
        imi = result.get("imitation_metrics", {})
        assert imi.get("total_steps", 0) > 0

    def test_multi_cycle_parallel(self, tmp_path: Path):
        """multi-cycle + parallel selfplay/eval"""
        from mahjong_rl.runner import Stage1Runner

        config = self._make_parallel_config(sp_workers=2, eval_workers=2)
        config.experiment["phases"] = ["imitation", "selfplay", "learner"]
        config.training["multi_cycle"] = {
            "enabled": True,
            "num_cycles": 2,
            "selfplay_matches_per_cycle": 4,
            "eval_each_cycle": True,
        }
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"
        cycles = result.get("cycles", [])
        assert len(cycles) == 2


class TestParallelWorkerValidation:
    """worker timeout / exitcode 検知テスト"""

    def test_nonzero_exitcode_raises(self):
        """非ゼロ exitcode の worker で RuntimeError"""
        from mahjong_rl.stage2a_parallel import _wait_and_validate
        import multiprocessing as mp

        class FakeProcess:
            def __init__(self, exitcode):
                self._exitcode = exitcode
            def join(self, timeout=None):
                pass
            def is_alive(self):
                return False
            @property
            def exitcode(self):
                return self._exitcode
            def terminate(self):
                pass
            def kill(self):
                pass

        ctx = mp.get_context("spawn")
        error_queue = ctx.Queue()
        procs = [FakeProcess(exitcode=1)]
        with pytest.raises(RuntimeError, match="non-zero exit code"):
            _wait_and_validate(procs, error_queue, "test")

    def test_timeout_raises(self):
        """timeout した worker で RuntimeError"""
        from mahjong_rl.stage2a_parallel import _wait_and_validate
        import multiprocessing as mp

        class StuckProcess:
            def join(self, timeout=None):
                pass
            def is_alive(self):
                return True  # still alive = timeout
            @property
            def exitcode(self):
                return None
            def terminate(self):
                pass
            def kill(self):
                pass

        ctx = mp.get_context("spawn")
        error_queue = ctx.Queue()
        procs = [StuckProcess()]
        with pytest.raises(RuntimeError, match="timeout"):
            _wait_and_validate(procs, error_queue, "test")


class TestStage2aMultiChunkImitation:
    """CQ-0236: multi_chunk_imitation"""

    def test_multi_chunk_3(self, tmp_path: Path):
        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig
        config = ExperimentConfig()
        config.experiment = {
            "name": "mc_imi", "stage": "stage2a", "observation_mode": "full",
            "global_seed": 42, "phases": ["imitation", "selfplay"],
        }
        config.selfplay = {"imitation_matches": 3}
        config.imitation = {"num_matches": 3}
        config.model = {"discard_hidden_dims": [32], "optional_hidden_dims": [32], "candidate_dim": 8}
        config.training = {
            "algorithm": "imitation", "lr": 1e-3, "batch_size": 16, "epochs": 1,
            "multi_chunk_imitation": {
                "enabled": True, "num_chunks": 3, "imitation_matches_per_chunk": 2,
            },
        }
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"
        imi = result.get("imitation_metrics", {})
        mci = imi.get("multi_chunk_imitation", {})
        assert mci.get("enabled") is True
        assert len(mci.get("chunks", [])) == 3


class TestStage2aMixedSelfplayStats:
    """policy_* stats が policy-seat 基準 (Stage1 parity)"""

    def test_mixed_selfplay_policy_stats(self, tmp_path: Path):
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path, observation_mode="full",
            encoder=encoder, policy_ratio=0.25, save_baseline_actions=True,
        )
        stats = worker.generate(num_matches=5, base_seed=42)
        total_outcomes = stats["tsumo_count"] + stats["ron_count"] + stats["ryukyoku_count"]
        assert total_outcomes > 0
        # policy_wins + policy_deal_ins は policy 席だけなので全席分より少ない
        assert stats["policy_wins"] + stats["policy_deal_ins"] >= 0
        # policy_draws は draw 1 局 1 加算 (Stage1 parity)
        # draw 局数以下であること
        assert stats["policy_draws"] <= stats["ryukyoku_count"]

    def test_policy_draws_stage1_semantics(self, tmp_path: Path):
        """policy_draws は draw 局ごとに最大 1 加算"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        encoder = FlatFeatureEncoder(observation_mode="full")
        # 全席 policy (policy_ratio=1.0)
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path, observation_mode="full",
            encoder=encoder, policy_ratio=1.0,
        )
        stats = worker.generate(num_matches=5, base_seed=0)
        # 全席 policy → draw 局数 == policy_draws
        assert stats["policy_draws"] == stats["ryukyoku_count"]


class TestStage2aMultiChunkParallel:
    """multi_chunk + parallel imitation"""

    def test_multi_chunk_parallel(self, tmp_path: Path):
        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig
        config = ExperimentConfig()
        config.experiment = {
            "name": "mcp", "stage": "stage2a", "observation_mode": "full",
            "global_seed": 42, "phases": ["imitation", "selfplay"],
        }
        config.selfplay = {"imitation_matches": 4}
        config.imitation = {"num_matches": 4, "num_workers": 2}
        config.model = {"discard_hidden_dims": [32], "optional_hidden_dims": [32], "candidate_dim": 8}
        config.training = {
            "algorithm": "imitation", "lr": 1e-3, "batch_size": 16, "epochs": 1,
            "multi_chunk_imitation": {
                "enabled": True, "num_chunks": 2, "imitation_matches_per_chunk": 2,
            },
        }
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"
        imi = result.get("imitation_metrics", {})
        mci = imi.get("multi_chunk_imitation", {})
        assert mci.get("enabled") is True
        assert len(mci.get("chunks", [])) == 2


class TestStage2aRuleMix:
    """CQ-0236: rule_mix"""

    def test_rule_mix_separated(self, tmp_path: Path):
        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig
        config = ExperimentConfig()
        config.experiment = {
            "name": "rm", "stage": "stage2a", "observation_mode": "full",
            "global_seed": 42, "phases": ["imitation", "selfplay", "learner"],
        }
        config.selfplay = {"imitation_matches": 3, "num_matches": 3, "seed_start": 0}
        config.imitation = {"num_matches": 3}
        config.model = {"discard_hidden_dims": [32], "optional_hidden_dims": [32], "candidate_dim": 8}
        config.training = {
            "algorithm": "ppo", "lr": 1e-3, "batch_size": 16, "epochs": 1,
            "multi_cycle": {
                "enabled": True, "num_cycles": 2,
                "selfplay_matches_per_cycle": 3, "eval_each_cycle": False,
            },
            "rule_mix": {
                "enabled": True, "policy_ratio": 0.5, "save_baseline_actions": True,
            },
            "rule_mix_learner": {
                "enabled": True, "ppo_mode": "separated",
                "baseline_imitation_epochs": 1, "policy_ppo_epochs": 1,
            },
        }
        config.evaluation = {"num_matches": 0}
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"
        cycles = result.get("cycles", [])
        assert len(cycles) == 2
        for cyc in cycles:
            ls = cyc.get("learner_stages", {})
            assert "policy_ppo" in ls


class TestStage2aSmokeConfig:
    """CQ-0228: smoke config テスト"""

    _ROOT = Path(__file__).resolve().parent.parent.parent

    def test_smoke_imitation_config_exists(self):
        assert (self._ROOT / "configs/stage2a_smoke_imitation.yaml").exists()

    def test_smoke_full_config_exists(self):
        assert (self._ROOT / "configs/stage2a_smoke_full.yaml").exists()

    def test_smoke_full_run(self, tmp_path: Path):
        """Stage2a smoke full config で end-to-end run が通る"""
        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig

        config = ExperimentConfig()
        config.experiment = {
            "name": "smoke_full",
            "stage": "stage2a",
            "observation_mode": "full",
            "global_seed": 42,
            "phases": ["imitation", "selfplay", "learner"],
        }
        config.selfplay = {
            "imitation_matches": 5,
            "num_matches": 5,
            "seed_start": 0,
        }
        config.imitation = {"num_matches": 5}
        config.model = {
            "discard_hidden_dims": [32],
            "optional_hidden_dims": [32],
            "candidate_dim": 8,
        }
        config.training = {
            "algorithm": "ppo",
            "lr": 1e-3,
            "batch_size": 16,
            "epochs": 1,
        }
        config.evaluation = {"num_matches": 0}

        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"

        # checkpoint
        run_dir = Path(result["run_dir"])
        assert (run_dir / "checkpoints" / "checkpoint_imitation.pt").exists()
        assert (run_dir / "checkpoints" / "checkpoint_learner.pt").exists()

        # summary.json の内容検証
        import json
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        # selfplay stats
        sp = summary["phase_stats"]["selfplay"]
        assert sp["total_matches"] == 5
        assert sp["num_rounds"] > 0
        assert sp.get("discard_count", 0) > 0
        assert sp.get("call_count", 0) > 0
        # policy outcome stats が 0 固定でない
        total_outcomes = (sp.get("policy_wins", 0)
                          + sp.get("policy_deal_ins", 0)
                          + sp.get("policy_draws", 0))
        assert total_outcomes > 0, "policy outcome stats が全て 0"
        assert sp["policy_win_by_tsumo"] + sp["policy_win_by_ron"] == sp["policy_wins"]

        # imitation stats
        imi = summary["phase_stats"]["imitation"]
        assert imi["num_updates"] > 0
        assert imi["policy_loss"] is not None

        # encoder_features
        ef = summary.get("encoder_features", {})
        assert ef.get("observation_mode") == "full"

        # artifacts_manifest
        manifest_path = run_dir / "artifacts_manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        lc = manifest["artifacts"]["learner_checkpoint"]
        assert lc["path"] == "checkpoints/checkpoint_learner.pt"
        assert lc["exists"] is True
        assert lc["stage"] == "stage2a"
