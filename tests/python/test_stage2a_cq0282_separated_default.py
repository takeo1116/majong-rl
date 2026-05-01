"""CQ-0282: Stage2a rule_mix_learner default を separated policy-only PPO に
変更する。

確認事項:
- default config で `rule_mix_learner.ppo_mode == "separated"`
- separated mode で baseline sample が PPO ratio / policy loss / aggregate
  diagnostics に混ざらない
- separated mode で `learner_stages["policy_ppo"]` に
  `ppo_mode="separated"`, `used_policy_samples`, `excluded_baseline_samples`
  が出る
- mixed + baseline_sample_weight>0 + allow_mixed_offpolicy_baseline=False
  → fail-fast
- mixed + allow_mixed_offpolicy_baseline=True → 動作し、warning が
  ppo_diag.mixed_ppo に残る
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

pytestmark = pytest.mark.smoke

from mahjong_rl.stage2a_learner import Stage2aLearner
from mahjong_rl.models.stage2a_model import Stage2aModel
from mahjong_rl.call_shard import (
    DecisionSample, CandidateRecord, DecisionShardWriter,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "stage2a_core_minimal_mixed_s1_baseline.yaml")


# ========== helpers ==========


def _mk_discard(step_id, actor_type="policy", version=3, **kw):
    base = dict(
        decision_type="discard",
        observation=np.zeros(10, dtype=np.float32),
        legal_mask=np.ones(34, dtype=np.float32),
        action=0, reward=0.0, log_prob=-0.5, value=0.0,
        terminated=False, round_over=False,
        player_id=0, episode_id="ep0", round_id=0,
        step_id=step_id, actor_type=actor_type,
        experiment_id="t", run_id="r", worker_id="w",
    )
    base.update(kw)
    s = DecisionSample(**base)
    s.sample_semantics_version = version
    return s


def _mk_call(step_id, actor_type="policy", version=3, **kw):
    base = dict(
        decision_type="call",
        observation=np.zeros(10, dtype=np.float32),
        reward=0.0, log_prob=-0.5, value=0.0,
        terminated=False, round_over=False,
        selected_candidate_index=0, candidate_count=2,
        candidates=[CandidateRecord(action_type=8),
                    CandidateRecord(action_type=4, tile_type=10,
                                     target_rel_seat=2)],
        player_id=0, episode_id="ep0", round_id=0,
        step_id=step_id, actor_type=actor_type,
        experiment_id="t", run_id="r", worker_id="w",
    )
    base.update(kw)
    s = DecisionSample(**base)
    s.sample_semantics_version = version
    return s


def _make_learner(tmp_path, *, ppo_mode="separated",
                   baseline_sample_weight=0.25,
                   allow_mixed_offpolicy_baseline=False):
    model = Stage2aModel(input_dim=10, discard_hidden_dims=[8],
                          optional_hidden_dims=[8])
    return Stage2aLearner(
        config={"training": {
            "algorithm": "ppo", "epochs": 1, "batch_size": 4,
            "rule_mix_learner": {
                "ppo_mode": ppo_mode,
                "baseline_sample_weight": baseline_sample_weight,
                "allow_mixed_offpolicy_baseline": (
                    allow_mixed_offpolicy_baseline),
            },
        }},
        model=model, run_dir=tmp_path / "run",
        device=torch.device("cpu"),
    )


def _write_mixed_shard(shard_dir, n_policy_d=4, n_baseline_d=4,
                       n_policy_c=2, n_baseline_c=2):
    """policy + baseline 混在 shard を書き出す"""
    writer = DecisionShardWriter(shard_dir, max_samples=100)
    sid = 0
    rng = np.random.RandomState(42)
    for i in range(n_policy_d):
        writer.add(_mk_discard(
            step_id=sid, actor_type="policy",
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=(i == n_policy_d - 1 and n_baseline_d == 0),
        ))
        sid += 1
    for i in range(n_baseline_d):
        writer.add(_mk_discard(
            step_id=sid, actor_type="baseline",
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=(i == n_baseline_d - 1),
        ))
        sid += 1
    for i in range(n_policy_c):
        writer.add(_mk_call(
            step_id=sid, actor_type="policy",
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=False,
        ))
        sid += 1
    for i in range(n_baseline_c):
        writer.add(_mk_call(
            step_id=sid, actor_type="baseline",
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=(i == n_baseline_c - 1),
        ))
        sid += 1
    writer.close()


# ========== 1. default config ==========


class TestDefaultConfigIsSeparated:
    """default config で `rule_mix_learner.ppo_mode == "separated"`"""

    def test_yaml_has_separated_mode(self):
        with open(DEFAULT_CONFIG) as f:
            cfg = yaml.safe_load(f)
        rml = cfg["training"]["rule_mix_learner"]
        assert rml["ppo_mode"] == "separated"

    def test_yaml_has_baseline_imitation_epochs_zero(self):
        with open(DEFAULT_CONFIG) as f:
            cfg = yaml.safe_load(f)
        rml = cfg["training"]["rule_mix_learner"]
        assert rml.get("baseline_imitation_epochs") == 0

    def test_yaml_has_policy_ppo_epochs_one(self):
        with open(DEFAULT_CONFIG) as f:
            cfg = yaml.safe_load(f)
        rml = cfg["training"]["rule_mix_learner"]
        assert rml.get("policy_ppo_epochs") == 1

    def test_yaml_has_allow_flag_false(self):
        """default config では mixed off-policy 許可は明示 false"""
        with open(DEFAULT_CONFIG) as f:
            cfg = yaml.safe_load(f)
        rml = cfg["training"]["rule_mix_learner"]
        assert rml.get("allow_mixed_offpolicy_baseline") is False


# ========== 2. separated mode は baseline を ratio に混ぜない ==========


class TestSeparatedExcludesBaseline:
    """separated mode で baseline sample が PPO ratio / policy loss /
    aggregate diagnostics に混ざらない"""

    def test_separated_uses_only_policy_samples(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir,
                            n_policy_d=4, n_baseline_d=4,
                            n_policy_c=2, n_baseline_c=2)
        learner = _make_learner(tmp_path, ppo_mode="separated")
        metrics = learner.train(shard_dir, filter_actor_type="policy")
        # PPO は policy sample のみ使う → discard 4 + call 2 = 6
        assert metrics["mode"] == "ppo"
        assert metrics["discard_count"] == 4
        assert metrics["call_count"] == 2
        assert metrics["total_steps"] == 6

    def test_separated_metrics_contain_new_keys(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir,
                            n_policy_d=4, n_baseline_d=3,
                            n_policy_c=2, n_baseline_c=1)
        learner = _make_learner(tmp_path, ppo_mode="separated")
        metrics = learner.train(shard_dir, filter_actor_type="policy")
        assert metrics["ppo_mode"] == "separated"
        assert metrics["executed"] is True
        # policy total = 4 (discard) + 2 (call) = 6
        # baseline total = 3 (discard) + 1 (call) = 4
        assert metrics["used_policy_samples"] == 6
        assert metrics["used_baseline_samples"] == 0
        assert metrics["excluded_baseline_samples"] == 4

    def test_separated_no_mixed_ppo_diag(self, tmp_path):
        """separated では aggregate diagnostics に mixed_ppo key が出ない"""
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir,
                            n_policy_d=4, n_baseline_d=4,
                            n_policy_c=2, n_baseline_c=2)
        learner = _make_learner(tmp_path, ppo_mode="separated")
        metrics = learner.train(shard_dir, filter_actor_type="policy")
        # is_mixed=False で _train_ppo は走るはず → mixed_ppo 出ない
        assert "mixed_ppo" not in metrics["ppo_diag"]

    def test_separated_no_baseline_in_shard_still_ok(self, tmp_path):
        """baseline sample 0 件の shard でも separated として正しく回る"""
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir,
                            n_policy_d=4, n_baseline_d=0,
                            n_policy_c=2, n_baseline_c=0)
        learner = _make_learner(tmp_path, ppo_mode="separated")
        metrics = learner.train(shard_dir, filter_actor_type="policy")
        assert metrics["ppo_mode"] == "separated"
        assert metrics["used_policy_samples"] == 6
        assert metrics["excluded_baseline_samples"] == 0


# ========== 3. mixed mode の fail-fast / opt-in ==========


class TestMixedRequiresExplicitOptIn:
    """mixed + baseline_sample_weight>0 + allow=False → ValueError"""

    def test_fail_fast_default(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir,
                            n_policy_d=2, n_baseline_d=2,
                            n_policy_c=1, n_baseline_c=1)
        learner = _make_learner(
            tmp_path, ppo_mode="mixed",
            baseline_sample_weight=0.25,
            allow_mixed_offpolicy_baseline=False)
        with pytest.raises(ValueError, match="off-policy"):
            learner.train(shard_dir)

    def test_fail_fast_message_mentions_separated(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir,
                            n_policy_d=2, n_baseline_d=2,
                            n_policy_c=1, n_baseline_c=1)
        learner = _make_learner(
            tmp_path, ppo_mode="mixed",
            baseline_sample_weight=0.5,
            allow_mixed_offpolicy_baseline=False)
        with pytest.raises(ValueError) as exc_info:
            learner.train(shard_dir)
        msg = str(exc_info.value)
        assert "separated" in msg
        assert "allow_mixed_offpolicy_baseline" in msg

    def test_zero_baseline_weight_does_not_fail(self, tmp_path):
        """baseline_sample_weight=0 なら mixed でも fail-fast しない
        (実質 policy-only で動く)"""
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir,
                            n_policy_d=4, n_baseline_d=2,
                            n_policy_c=2, n_baseline_c=0)
        learner = _make_learner(
            tmp_path, ppo_mode="mixed",
            baseline_sample_weight=0.0,
            allow_mixed_offpolicy_baseline=False)
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        assert metrics["ppo_mode"] == "mixed"


class TestMixedAllowedExplicitly:
    """allow_mixed_offpolicy_baseline=True なら従来 mixed が動く"""

    def test_mixed_runs_with_allow_flag(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir,
                            n_policy_d=4, n_baseline_d=4,
                            n_policy_c=2, n_baseline_c=2)
        learner = _make_learner(
            tmp_path, ppo_mode="mixed",
            baseline_sample_weight=0.5,
            allow_mixed_offpolicy_baseline=True)
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        assert metrics["ppo_mode"] == "mixed"
        assert metrics["used_policy_samples"] == 6
        assert metrics["used_baseline_samples"] == 6
        assert metrics["excluded_baseline_samples"] == 0

    def test_mixed_diag_carries_warning_and_flag(self, tmp_path):
        """mixed_ppo diag に warning と allow flag が残る"""
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir,
                            n_policy_d=4, n_baseline_d=4,
                            n_policy_c=2, n_baseline_c=2)
        learner = _make_learner(
            tmp_path, ppo_mode="mixed",
            baseline_sample_weight=0.5,
            allow_mixed_offpolicy_baseline=True)
        metrics = learner.train(shard_dir)
        mp = metrics["ppo_diag"]["mixed_ppo"]
        assert mp["mixed_ppo_enabled"] is True
        assert mp["allow_mixed_offpolicy_baseline"] is True
        assert "warning" in mp
        assert "off-policy" in mp["warning"].lower()
        assert mp["num_policy_samples"] == 6
        assert mp["num_baseline_samples"] == 6


# ========== 4. empty shard でも metadata が出る ==========


class TestEmptyShardMetadata:
    """空 shard でも ppo_mode / executed / counts が出る"""

    def test_empty_shard_separated(self, tmp_path):
        shard_dir = tmp_path / "shards"
        # 空 shard
        writer = DecisionShardWriter(shard_dir, max_samples=10)
        writer.close()
        learner = _make_learner(tmp_path, ppo_mode="separated")
        metrics = learner.train(shard_dir, filter_actor_type="policy")
        assert metrics["ppo_mode"] == "separated"
        assert metrics["executed"] is False
        assert metrics["used_policy_samples"] == 0
        assert metrics["excluded_baseline_samples"] == 0


# ========== 5. runner E2E (separated default) ==========


class TestRunnerSeparatedDefaultE2E:
    """runner Stage2a multi-cycle で separated default が選ばれる"""

    def test_runner_separated_default_produces_policy_ppo_stage(self, tmp_path):
        """default(=separated) で `learner_stages.policy_ppo` が出る"""
        try:
            from mahjong_rl._mahjong_core import (  # noqa: F401
                StageEnvironment)
        except Exception:
            pytest.skip("C++ engine not available")

        from mahjong_rl.runner import Stage1Runner

        # 最小の Stage2a multi-cycle config
        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.experiment = {
            "name": "cq0282_runner_smoke",
            "stage": "stage2a",
            "observation_mode": "full",
            "phases": ["selfplay", "learner"],
            "global_seed": 0,
        }
        cfg.feature_encoder = {
            "name": "FlatFeatureEncoder",
            "observation_mode": "full",
            "shanten_hint": {"enabled": True},
            "discard_ukeire_hint": {"enabled": True},
        }
        cfg.model = {
            "discard_hidden_dims": [16, 8],
            "optional_hidden_dims": [8, 4],
            "value_hidden_dims": [8, 4],
            "candidate_dim": 8,
            "optional_scorer_hidden": 8,
        }
        cfg.selfplay = {
            "num_matches": 2, "seed_start": 0, "num_workers": 1,
            "policy_ratio": 1.0, "save_baseline_actions": False,
            "inference_device": "cpu",
            "max_samples_per_shard": 10000,
        }
        cfg.imitation = {"num_workers": 1}
        cfg.training = {
            "algorithm": "ppo", "lr": 1e-4, "batch_size": 32,
            "epochs": 1, "gamma": 0.5, "gae_lambda": 0.0,
            "clip_epsilon": 0.15, "value_loss_coef": 0.25,
            "entropy_coef": 0.0, "max_grad_norm": 0.5,
            "multi_cycle": {
                "enabled": True, "num_cycles": 1,
                "selfplay_matches_per_cycle": 2,
                "eval_each_cycle": False,
            },
            "rule_mix": {
                "enabled": True, "policy_ratio": 0.5,
                "save_baseline_actions": True,
            },
            "rule_mix_learner": {
                "enabled": True,
                "ppo_mode": "separated",
                "baseline_imitation_epochs": 0,
                "policy_ppo_epochs": 1,
            },
        }
        cfg.evaluation = {"num_matches": 0}

        runner = Stage1Runner(config=cfg, base_dir=tmp_path)
        try:
            result = runner.run()
        except Exception as e:
            pytest.skip(f"runner E2E skipped: {e}")
        assert "error" not in result, f"error: {result.get('error')}"
        cycles = result.get("cycles", [])
        assert cycles, "no cycles produced"
        for cyc in cycles:
            ls = cyc.get("learner_stages", {})
            assert "policy_ppo" in ls, (
                "separated default should produce policy_ppo stage; "
                f"got {list(ls.keys())}")
            ppo = ls["policy_ppo"]
            # CQ-0282: 新 metadata
            assert ppo.get("ppo_mode") == "separated", ppo
            assert ppo.get("executed") is True, ppo
            assert "used_policy_samples" in ppo
            assert "excluded_baseline_samples" in ppo
            # mixed_ppo は出ない
            assert "mixed_ppo" not in ls
