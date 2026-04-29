"""CQ-0276: Stage2a selfplay / eval / parallel への reward_config 伝播テスト"""
import pytest
import numpy as np
import torch
from pathlib import Path

pytestmark = pytest.mark.smoke

from mahjong_rl.stage2_selfplay_worker import (
    Stage2SelfPlayWorker, build_reward_policy_config,
)
from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
from mahjong_rl.encoders import FlatFeatureEncoder
from mahjong_rl.models.stage2a_model import Stage2aModel


class TestBuildRewardPolicyConfig:
    """build_reward_policy_config helper"""

    def test_none_returns_none(self):
        assert build_reward_policy_config(None) is None

    def test_empty_dict_returns_none(self):
        assert build_reward_policy_config({}) is None

    def test_point_delta_scale(self):
        cfg = build_reward_policy_config({"point_delta_scale": 0.0001})
        assert cfg is not None
        assert cfg.point_delta_scale == pytest.approx(0.0001)

    def test_default_scale_is_1(self):
        cfg = build_reward_policy_config({"some_other_key": 1})
        assert cfg is not None
        assert cfg.point_delta_scale == pytest.approx(1.0)


class TestStage2SelfPlayWorkerRewardConfig:
    """Stage2SelfPlayWorker が config.reward を受け取り Stage2Env に渡す"""

    def test_no_reward_config_default(self, tmp_path):
        """reward 未指定 → _reward_config is None"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        assert worker._reward_config is None

    def test_reward_config_built(self, tmp_path):
        """config.reward.point_delta_scale が _reward_config に反映される"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={"reward": {"point_delta_scale": 0.001}},
            output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        assert worker._reward_config is not None
        assert worker._reward_config.point_delta_scale == pytest.approx(0.001)

    def test_reward_scale_affects_shard_rewards(self, tmp_path):
        """reward scale を変えると shard reward の絶対値が変わる"""
        from mahjong_rl.call_shard import DecisionShardReader
        encoder = FlatFeatureEncoder(observation_mode="full")

        def _gen(scale, out_dir):
            worker = Stage2SelfPlayWorker(
                config={"reward": {"point_delta_scale": scale}},
                output_dir=out_dir,
                observation_mode="full", encoder=encoder,
            )
            worker.generate(num_matches=3, base_seed=42)
            samples = DecisionShardReader(out_dir).read_all()
            return [s.reward for s in samples]

        rewards_1 = _gen(1.0, tmp_path / "scale_1")
        rewards_001 = _gen(0.001, tmp_path / "scale_001")
        # 同 seed なので sample 数も挙動も同じはず
        assert len(rewards_1) == len(rewards_001)
        # absolute non-zero rewards should be ~1000x smaller in scale=0.001
        nonzero_1 = sum(abs(r) for r in rewards_1 if r != 0)
        nonzero_001 = sum(abs(r) for r in rewards_001 if r != 0)
        assert nonzero_001 < nonzero_1
        # close to 1000x ratio (allow noise)
        if nonzero_1 > 0:
            ratio = nonzero_001 / nonzero_1
            assert ratio < 0.01  # at least 100x smaller


class TestStage2aEvaluatorRewardConfig:
    """Stage2aEvaluator が reward_config を受け取り Stage2Env に渡す"""

    def test_default_no_reward_config(self):
        encoder = FlatFeatureEncoder(observation_mode="full")
        dim = encoder.metadata().output_shape[0]
        model = Stage2aModel(input_dim=dim, discard_hidden_dims=[16],
                              optional_hidden_dims=[16])
        evaluator = Stage2aEvaluator(
            model=model, encoder=encoder, device=torch.device("cpu"))
        assert evaluator._reward_config is None

    def test_with_reward_config(self):
        encoder = FlatFeatureEncoder(observation_mode="full")
        dim = encoder.metadata().output_shape[0]
        model = Stage2aModel(input_dim=dim, discard_hidden_dims=[16],
                              optional_hidden_dims=[16])
        cfg = build_reward_policy_config({"point_delta_scale": 0.5})
        evaluator = Stage2aEvaluator(
            model=model, encoder=encoder, device=torch.device("cpu"),
            reward_config=cfg)
        assert evaluator._reward_config is not None
        assert evaluator._reward_config.point_delta_scale == pytest.approx(0.5)

    def test_eval_smoke_with_reward_config(self):
        """reward_config 付き eval が crash しない"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        dim = encoder.metadata().output_shape[0]
        model = Stage2aModel(input_dim=dim, discard_hidden_dims=[16],
                              optional_hidden_dims=[16])
        cfg = build_reward_policy_config({"point_delta_scale": 0.001})
        evaluator = Stage2aEvaluator(
            model=model, encoder=encoder, device=torch.device("cpu"),
            reward_config=cfg)
        result = evaluator.evaluate(num_matches=2, seed_start=42)
        assert result["total_rounds"] > 0


class TestParallelWorkerRewardConfig:
    """parallel selfplay worker が reward_config_dict を受け取れる"""

    def test_parallel_selfplay_has_reward_config_param(self):
        """関数シグネチャに reward_config_dict が含まれる"""
        from mahjong_rl.stage2a_parallel import (
            run_stage2a_selfplay_parallel, _stage2a_selfplay_worker_fn,
            run_stage2a_eval_parallel, _stage2a_eval_worker_fn,
        )
        import inspect
        sig = inspect.signature(run_stage2a_selfplay_parallel)
        assert "reward_config_dict" in sig.parameters
        sig = inspect.signature(_stage2a_selfplay_worker_fn)
        assert "reward_config_dict" in sig.parameters
        sig = inspect.signature(run_stage2a_eval_parallel)
        assert "reward_config_dict" in sig.parameters
        sig = inspect.signature(_stage2a_eval_worker_fn)
        assert "reward_config_dict" in sig.parameters
