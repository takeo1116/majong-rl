"""CQ-0278: Stage2a selfplay の policy sampling RNG / temperature 反映テスト"""
import pytest
import numpy as np
import torch
from pathlib import Path

pytestmark = pytest.mark.smoke

from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
from mahjong_rl.encoders import FlatFeatureEncoder
from mahjong_rl.models.stage2a_model import Stage2aModel


class TestTemperatureConfig:
    """selfplay.temperature の読み取り"""

    def test_default_temperature_is_1(self, tmp_path):
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=None,
        )
        assert worker._temperature == pytest.approx(1.0)

    def test_temperature_from_selfplay_section(self, tmp_path):
        """runner._as_dict() 由来形式: config['selfplay']['temperature']"""
        worker = Stage2SelfPlayWorker(
            config={"selfplay": {"temperature": 0.5}},
            output_dir=tmp_path,
            observation_mode="full", encoder=None,
        )
        assert worker._temperature == pytest.approx(0.5)

    def test_temperature_flat_dict(self, tmp_path):
        """flat dict 形式: config['temperature']"""
        worker = Stage2SelfPlayWorker(
            config={"temperature": 0.7},
            output_dir=tmp_path,
            observation_mode="full", encoder=None,
        )
        assert worker._temperature == pytest.approx(0.7)

    def test_selfplay_section_takes_priority(self, tmp_path):
        """selfplay.temperature が flat より優先"""
        worker = Stage2SelfPlayWorker(
            config={"selfplay": {"temperature": 0.3}, "temperature": 0.9},
            output_dir=tmp_path,
            observation_mode="full", encoder=None,
        )
        assert worker._temperature == pytest.approx(0.3)

    def test_temperature_is_float(self, tmp_path):
        """int を渡しても float に変換される"""
        worker = Stage2SelfPlayWorker(
            config={"selfplay": {"temperature": 2}},
            output_dir=tmp_path,
            observation_mode="full", encoder=None,
        )
        assert worker._temperature == pytest.approx(2.0)
        assert isinstance(worker._temperature, float)


class TestPolicyDiscardTemperaturePassed:
    """_policy_discard が self._temperature を select_discard_sample に渡す"""

    def test_temperature_passed_to_selector(self, tmp_path, monkeypatch):
        encoder = FlatFeatureEncoder(observation_mode="full")
        dim = encoder.metadata().output_shape[0]
        model = Stage2aModel(input_dim=dim, discard_hidden_dims=[16],
                              optional_hidden_dims=[16])
        worker = Stage2SelfPlayWorker(
            config={"selfplay": {"temperature": 0.42}},
            output_dir=tmp_path,
            observation_mode="full", encoder=encoder, model=model,
        )

        captured = {}

        def fake_select(logits, mask, temperature=1.0):
            captured["temperature"] = temperature
            return 0, -1.0

        monkeypatch.setattr(
            "mahjong_rl.stage2a_selector.select_discard_sample", fake_select)

        feat = np.zeros(dim, dtype=np.float32)
        mask = np.ones(34, dtype=np.float32)
        worker._policy_discard(feat, mask)
        assert captured["temperature"] == pytest.approx(0.42)


class TestPolicyCallTemperaturePassed:
    """_policy_call が self._temperature を select_optional_sample に渡す"""

    def test_temperature_passed_to_selector(self, tmp_path, monkeypatch):
        encoder = FlatFeatureEncoder(observation_mode="full")
        dim = encoder.metadata().output_shape[0]
        model = Stage2aModel(input_dim=dim, discard_hidden_dims=[16],
                              optional_hidden_dims=[16])
        worker = Stage2SelfPlayWorker(
            config={"selfplay": {"temperature": 0.31}},
            output_dir=tmp_path,
            observation_mode="full", encoder=encoder, model=model,
        )

        captured = {}

        def fake_select(scores, mask, temperature=1.0):
            captured["temperature"] = temperature
            return 0, -1.0

        monkeypatch.setattr(
            "mahjong_rl.stage2a_selector.select_optional_sample", fake_select)

        from mahjong_rl.call_shard import CandidateRecord
        feat = np.zeros(dim, dtype=np.float32)
        cand_records = [CandidateRecord(action_type=8)]  # Skip
        # candidates list: any object list of length 1 is fine since
        # _make_cand_records は cand_records が渡れば呼ばれない
        worker._policy_call(feat, candidates=[None],
                             resp_ctx=np.zeros(3, dtype=np.float32),
                             cand_records=cand_records)
        assert captured["temperature"] == pytest.approx(0.31)


class TestTorchSamplingReproducibility:
    """同じ seed → 同じ multinomial 結果、違う seed → 違う結果"""

    def test_same_seed_same_sample(self):
        """torch.manual_seed が同じなら multinomial 結果が一致"""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        def _sample(seed):
            torch.manual_seed(seed)
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=10, replacement=True).tolist()

        s1 = _sample(42)
        s2 = _sample(42)
        assert s1 == s2

    def test_different_seed_different_sample(self):
        """seed が異なれば結果も異なりうる (10 サンプル中少なくとも 1 つ違う)"""
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])  # uniform で違いやすく

        def _sample(seed):
            torch.manual_seed(seed)
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=10, replacement=True).tolist()

        s1 = _sample(42)
        s2 = _sample(43)
        # uniform で 10 サンプルあれば違いが出るはず
        assert s1 != s2


class TestSelfplayMatchSeedReproducibility:
    """同 seed の generate() で torch RNG が反復可能"""

    def _build_worker(self, tmp_path, model):
        encoder = FlatFeatureEncoder(observation_mode="full")
        return Stage2SelfPlayWorker(
            config={"selfplay": {"temperature": 1.0}},
            output_dir=tmp_path,
            observation_mode="full", encoder=encoder, model=model,
        )

    def test_same_seed_reproducible_actions(self, tmp_path):
        """同じ model / seed で 2 回 generate → action sequence が一致"""
        from mahjong_rl.call_shard import DecisionShardReader
        encoder = FlatFeatureEncoder(observation_mode="full")
        dim = encoder.metadata().output_shape[0]
        # 共通 model (deterministic init で重要)
        torch.manual_seed(0)
        model = Stage2aModel(input_dim=dim, discard_hidden_dims=[16],
                              optional_hidden_dims=[16])

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        for d in (out1, out2):
            d.mkdir(parents=True, exist_ok=True)

        w1 = Stage2SelfPlayWorker(
            config={"selfplay": {"temperature": 1.0}},
            output_dir=out1,
            observation_mode="full", encoder=encoder, model=model,
        )
        w1.generate(num_matches=2, base_seed=42)
        s1 = DecisionShardReader(out1).read_all()

        w2 = Stage2SelfPlayWorker(
            config={"selfplay": {"temperature": 1.0}},
            output_dir=out2,
            observation_mode="full", encoder=encoder, model=model,
        )
        w2.generate(num_matches=2, base_seed=42)
        s2 = DecisionShardReader(out2).read_all()

        assert len(s1) == len(s2)
        assert len(s1) > 0
        # discard branch は action int で比較しやすい
        d1 = [s.action for s in s1 if s.decision_type == "discard"]
        d2 = [s.action for s in s2 if s.decision_type == "discard"]
        assert d1 == d2

    def test_different_seed_can_change_actions(self, tmp_path):
        """違う seed では action sequence が変わりうる"""
        from mahjong_rl.call_shard import DecisionShardReader
        encoder = FlatFeatureEncoder(observation_mode="full")
        dim = encoder.metadata().output_shape[0]
        torch.manual_seed(0)
        model = Stage2aModel(input_dim=dim, discard_hidden_dims=[16],
                              optional_hidden_dims=[16])

        out1 = tmp_path / "run42"
        out2 = tmp_path / "run43"
        for d in (out1, out2):
            d.mkdir(parents=True, exist_ok=True)

        w1 = Stage2SelfPlayWorker(
            config={"selfplay": {"temperature": 1.0}},
            output_dir=out1, observation_mode="full",
            encoder=encoder, model=model,
        )
        w1.generate(num_matches=2, base_seed=42)
        s1 = DecisionShardReader(out1).read_all()

        w2 = Stage2SelfPlayWorker(
            config={"selfplay": {"temperature": 1.0}},
            output_dir=out2, observation_mode="full",
            encoder=encoder, model=model,
        )
        w2.generate(num_matches=2, base_seed=43)
        s2 = DecisionShardReader(out2).read_all()

        # env.reset(seed) で配牌が変わるので少なくとも何かは違うはず
        d1 = [s.action for s in s1 if s.decision_type == "discard"]
        d2 = [s.action for s in s2 if s.decision_type == "discard"]
        # 配牌が変わるため少なくとも一部が異なる
        assert d1 != d2


# ========== CQ-0278 follow-up: parallel temperature 伝播 ==========


class TestParallelTemperatureSignature:
    """run_stage2a_selfplay_parallel / _stage2a_selfplay_worker_fn が
    temperature を受け取れる"""

    def test_run_parallel_has_temperature_param(self):
        from mahjong_rl.stage2a_parallel import run_stage2a_selfplay_parallel
        import inspect
        sig = inspect.signature(run_stage2a_selfplay_parallel)
        assert "temperature" in sig.parameters
        # default は 1.0
        assert sig.parameters["temperature"].default == 1.0

    def test_worker_fn_has_temperature_param(self):
        from mahjong_rl.stage2a_parallel import _stage2a_selfplay_worker_fn
        import inspect
        sig = inspect.signature(_stage2a_selfplay_worker_fn)
        assert "temperature" in sig.parameters
        assert sig.parameters["temperature"].default == 1.0


class TestParallelWorkerConfigBuild:
    """parallel worker fn が worker_config に temperature を入れて
    Stage2SelfPlayWorker を作る"""

    def test_subprocess_passes_temperature_to_worker(self, tmp_path, monkeypatch):
        """subprocess を起動せず Stage2SelfPlayWorker への config を捕捉"""
        captured: dict = {}

        original_worker_cls = None

        from mahjong_rl import stage2a_parallel as sap
        from mahjong_rl import stage2_selfplay_worker as ssw

        original_worker_cls = ssw.Stage2SelfPlayWorker

        class FakeWorker:
            def __init__(self, config, output_dir, observation_mode,
                         encoder, model, device, policy_ratio,
                         save_baseline_actions):
                captured["config"] = config

            def generate(self, num_matches, base_seed,
                         experiment_id, run_id, worker_id):
                return {"num_matches": num_matches, "total_steps": 0}

        # subprocess を経由しない: monkeypatch で Stage2SelfPlayWorker を差し替え
        monkeypatch.setattr(
            "mahjong_rl.stage2_selfplay_worker.Stage2SelfPlayWorker",
            FakeWorker)

        # encoder rebuild は本物を使う (model state path None で軽く)
        # _stage2a_selfplay_worker_fn を直接呼ぶ
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        result_q = ctx.Queue()
        error_q = ctx.Queue()

        # ただし subprocess を起こさず、関数を同 process で呼んで内部 import を踏む。
        # _stage2a_selfplay_worker_fn は引数を順序で受け取るのでそのまま呼べる。
        sap._stage2a_selfplay_worker_fn(
            0, str(tmp_path / "worker_0"), "full", {},
            None, {}, 1, 0,
            "exp", "run",
            result_q, error_q,
            "cpu", 1.0, False, 1, None, 0.42,
        )
        assert "config" in captured
        cfg = captured["config"]
        # selfplay 配下に temperature が入る
        assert "selfplay" in cfg
        assert cfg["selfplay"]["temperature"] == pytest.approx(0.42)

    def test_subprocess_default_temperature_is_1(self, tmp_path, monkeypatch):
        """temperature 引数を渡さなければ default 1.0 が伝わる"""
        captured: dict = {}

        from mahjong_rl import stage2a_parallel as sap

        class FakeWorker:
            def __init__(self, config, output_dir, observation_mode,
                         encoder, model, device, policy_ratio,
                         save_baseline_actions):
                captured["config"] = config

            def generate(self, num_matches, base_seed,
                         experiment_id, run_id, worker_id):
                return {"num_matches": num_matches, "total_steps": 0}

        monkeypatch.setattr(
            "mahjong_rl.stage2_selfplay_worker.Stage2SelfPlayWorker",
            FakeWorker)

        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        result_q = ctx.Queue()
        error_q = ctx.Queue()

        # temperature を引数で渡さない (default 1.0)
        sap._stage2a_selfplay_worker_fn(
            0, str(tmp_path / "worker_0"), "full", {},
            None, {}, 1, 0,
            "exp", "run",
            result_q, error_q,
        )
        cfg = captured["config"]
        assert cfg["selfplay"]["temperature"] == pytest.approx(1.0)

    def test_run_parallel_passes_temperature(self, tmp_path, monkeypatch):
        """run_stage2a_selfplay_parallel が temperature を worker_fn に渡す"""
        from mahjong_rl import stage2a_parallel as sap

        captured_args: list = []

        class FakeProcess:
            def __init__(self, target, args):
                captured_args.append(args)
                self._args = args

            def start(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return False

            @property
            def exitcode(self):
                return 0

        class FakeContext:
            def Process(self, target, args):
                return FakeProcess(target, args)

            def Queue(self):
                class Q:
                    def empty(self):
                        return True
                    def get_nowait(self):
                        raise Exception("empty")
                    def put(self, x):
                        pass
                return Q()

        monkeypatch.setattr(sap.mp, "get_context", lambda name: FakeContext())

        sap.run_stage2a_selfplay_parallel(
            output_dir=tmp_path,
            num_workers=2,
            num_matches=2,
            base_seed=42,
            obs_mode="full",
            encoder_config={},
            model_state_path=None,
            model_config={},
            temperature=0.31,
        )
        assert len(captured_args) == 2
        # _stage2a_selfplay_worker_fn の positional 引数列を inspect.signature
        # 経由で位置検索する (CQ-0292 で末尾に optional_flags が増えた)
        import inspect
        sig = inspect.signature(sap._stage2a_selfplay_worker_fn)
        params = list(sig.parameters.keys())
        # captured_args は worker_fn 1st arg 以降の positional
        # = inspect 上の "worker_id" を含む全 positional
        temp_idx = params.index("temperature")
        for args in captured_args:
            assert args[temp_idx] == pytest.approx(0.31)


class TestRunnerPassesTemperature:
    """runner.py の Stage2a parallel selfplay 呼び出しが
    sp_cfg.temperature を渡すこと"""

    def test_runner_passes_temperature_to_parallel(self, tmp_path, monkeypatch):
        """runner._run_selfplay_stage2a_single (parallel path) が
        run_stage2a_selfplay_parallel に temperature を渡す"""
        from mahjong_rl import stage2a_parallel as sap

        captured: dict = {}

        def fake_run_parallel(*args, **kwargs):
            captured.update(kwargs)
            return {"total_steps": 0, "num_matches": 0,
                    "discard_count": 0, "call_count": 0,
                    "num_rounds": 0, "tsumo_count": 0,
                    "ron_count": 0, "ryukyoku_count": 0,
                    "policy_wins": 0, "policy_deal_ins": 0,
                    "policy_draws": 0, "policy_win_by_tsumo": 0,
                    "policy_win_by_ron": 0,
                    "total_matches": 0,
                    "num_workers": 1}

        monkeypatch.setattr(sap, "run_stage2a_selfplay_parallel", fake_run_parallel)
        # runner.py で `from mahjong_rl.stage2a_parallel import ...` を踏むため
        import mahjong_rl.runner as runner_mod
        # 動作上 import 後に local 名前空間に取り込まれているので、
        # その local 参照も差し替える
        runner_mod.run_stage2a_selfplay_parallel = fake_run_parallel  # type: ignore

        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig

        config = ExperimentConfig()
        config.experiment = {
            "name": "test_temp_propagation",
            "stage": "stage2a",
            "observation_mode": "full",
            "global_seed": 42,
            "phases": ["selfplay"],
        }
        config.feature_encoder = {
            "shanten_hint": True,
            "discard_ukeire_hint": True,
        }
        config.selfplay = {
            "num_matches": 2, "seed_start": 0,
            "num_workers": 2,
            "temperature": 0.55,  # CQ-0278 follow-up: 渡るか確認
        }
        config.imitation = {"num_matches": 2}
        config.model = {"discard_hidden_dims": [16],
                         "optional_hidden_dims": [16],
                         "candidate_dim": 8}
        config.training = {"algorithm": "ppo", "lr": 1e-3,
                            "batch_size": 8, "epochs": 1}
        config.evaluation = {"num_matches": 0}

        runner = Stage1Runner(config=config, base_dir=tmp_path)
        try:
            runner.run()
        except Exception:
            # downstream phases が動かなくてもよい。
            # selfplay parallel 呼び出し時の kwargs だけ確認したい。
            pass

        assert "temperature" in captured, (
            f"runner が temperature kwarg を渡していない: keys={list(captured.keys())}")
        assert captured["temperature"] == pytest.approx(0.55)
