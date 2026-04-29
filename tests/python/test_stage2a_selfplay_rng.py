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
