"""CQ-0292 batch 1: Stage2a optional flag propagation + Stage2aEvaluator
optional decision handling + TSUMO_OPTIONAL Skip semantics.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.smoke


# ========== Section 1: parallel signature propagation ==========


class TestOptionalFlagsSignatures:
    """run_stage2a_selfplay_parallel / run_stage2a_eval_parallel 両方で
    optional_flags パラメータが受け取れる。"""

    def test_run_stage2a_selfplay_parallel_has_optional_flags(self):
        from mahjong_rl.stage2a_parallel import run_stage2a_selfplay_parallel
        sig = inspect.signature(run_stage2a_selfplay_parallel)
        assert "optional_flags" in sig.parameters

    def test_stage2a_selfplay_worker_fn_has_optional_flags(self):
        from mahjong_rl.stage2a_parallel import _stage2a_selfplay_worker_fn
        sig = inspect.signature(_stage2a_selfplay_worker_fn)
        assert "optional_flags" in sig.parameters

    def test_run_stage2a_eval_parallel_has_optional_flags(self):
        from mahjong_rl.stage2a_parallel import run_stage2a_eval_parallel
        sig = inspect.signature(run_stage2a_eval_parallel)
        assert "optional_flags" in sig.parameters

    def test_stage2a_eval_worker_fn_has_optional_flags(self):
        from mahjong_rl.stage2a_parallel import _stage2a_eval_worker_fn
        sig = inspect.signature(_stage2a_eval_worker_fn)
        assert "optional_flags" in sig.parameters


class TestNormalizeOptionalFlags:
    """_normalize_optional_flags が複数の入力形式を扱える。"""

    def test_none_returns_all_false(self):
        from mahjong_rl.stage2a_parallel import _normalize_optional_flags
        out = _normalize_optional_flags(None)
        for k in ("optional_riichi", "optional_tsumo", "optional_ron",
                   "optional_ankan", "optional_kakan", "optional_kyuushu"):
            assert out[k] is False

    def test_bool_dict_passthrough(self):
        from mahjong_rl.stage2a_parallel import _normalize_optional_flags
        out = _normalize_optional_flags({
            "optional_riichi": True,
            "optional_tsumo": True,
        })
        assert out["optional_riichi"] is True
        assert out["optional_tsumo"] is True
        assert out["optional_ron"] is False

    def test_nested_enabled_dict(self):
        from mahjong_rl.stage2a_parallel import _normalize_optional_flags
        out = _normalize_optional_flags({
            "optional_riichi": {"enabled": True},
            "optional_tsumo": {"enabled": False},
        })
        assert out["optional_riichi"] is True
        assert out["optional_tsumo"] is False


class TestSelfPlayWorkerReadOptionalFlag:
    """worker_config 経由で _read_optional_flag が enabled を見る。"""

    def test_top_level_dict_enabled_true(self):
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        cfg = {"optional_riichi": {"enabled": True}}
        assert Stage2SelfPlayWorker._read_optional_flag(
            cfg, "optional_riichi") is True

    def test_training_path_enabled_true(self):
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        cfg = {"training": {"optional_tsumo": {"enabled": True}}}
        assert Stage2SelfPlayWorker._read_optional_flag(
            cfg, "optional_tsumo") is True

    def test_missing_key_returns_false(self):
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        assert Stage2SelfPlayWorker._read_optional_flag(
            {}, "optional_riichi") is False


# ========== Section 2: Stage2aEvaluator optional handling ==========


def _make_evaluator(**flags):
    """簡易 evaluator (encoder/model は最小)。"""
    from mahjong_rl.encoders import FlatFeatureEncoder
    from mahjong_rl.models.stage2a_model import Stage2aModel
    from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
    encoder = FlatFeatureEncoder(observation_mode="full")
    meta = encoder.metadata()
    input_dim = int(np.prod(meta.output_shape))
    model = Stage2aModel(
        input_dim=input_dim,
        discard_hidden_dims=[16, 16],
        optional_hidden_dims=[16, 16],
        value_hidden_dims=[16, 16],
        candidate_dim=8,
        optional_scorer_hidden=8,
    )
    return Stage2aEvaluator(
        model=model, encoder=encoder,
        observation_mode="full",
        device=torch.device("cpu"),
        **flags,
    )


class TestEvaluatorAcceptsOptionalFlags:
    """Stage2aEvaluator が optional_*_enabled を受け取り state に保持する。"""

    def test_default_flags_all_false(self):
        ev = _make_evaluator()
        assert ev._optional_riichi_enabled is False
        assert ev._optional_tsumo_enabled is False
        assert ev._optional_ron_enabled is False
        assert ev._optional_ankan_enabled is False
        assert ev._optional_kakan_enabled is False
        assert ev._optional_kyuushu_enabled is False

    def test_all_flags_true(self):
        ev = _make_evaluator(
            optional_riichi_enabled=True,
            optional_tsumo_enabled=True,
            optional_ron_enabled=True,
            optional_ankan_enabled=True,
            optional_kakan_enabled=True,
            optional_kyuushu_enabled=True,
        )
        assert ev._optional_riichi_enabled is True
        assert ev._optional_tsumo_enabled is True
        assert ev._optional_ron_enabled is True
        assert ev._optional_ankan_enabled is True
        assert ev._optional_kakan_enabled is True
        assert ev._optional_kyuushu_enabled is True


class TestEvaluatorBaselineOptionalIndex:
    """baseline seat の teacher index が仕様どおり。"""

    def test_riichi_returns_1(self):
        from mahjong_rl.env import DecisionType
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
        assert Stage2aEvaluator._baseline_optional_index(
            DecisionType.RIICHI_OPTIONAL) == 1

    def test_tsumo_returns_0(self):
        from mahjong_rl.env import DecisionType
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
        assert Stage2aEvaluator._baseline_optional_index(
            DecisionType.TSUMO_OPTIONAL) == 0

    def test_ron_returns_0(self):
        from mahjong_rl.env import DecisionType
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
        assert Stage2aEvaluator._baseline_optional_index(
            DecisionType.RON_OPTIONAL) == 0

    def test_ankan_returns_1(self):
        from mahjong_rl.env import DecisionType
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
        assert Stage2aEvaluator._baseline_optional_index(
            DecisionType.ANKAN_OPTIONAL) == 1

    def test_kakan_returns_1(self):
        from mahjong_rl.env import DecisionType
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
        assert Stage2aEvaluator._baseline_optional_index(
            DecisionType.KAKAN_OPTIONAL) == 1

    def test_kyuushu_returns_1(self):
        from mahjong_rl.env import DecisionType
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
        assert Stage2aEvaluator._baseline_optional_index(
            DecisionType.KYUUSHU_OPTIONAL) == 1


class TestEvaluatorRunsWithOptionalFlags:
    """optional flags を全部 ON で短い eval を動かしても crash しない。"""

    def test_evaluate_with_all_optional_flags_does_not_crash(self):
        ev = _make_evaluator(
            optional_riichi_enabled=True,
            optional_tsumo_enabled=True,
            optional_ron_enabled=True,
            optional_ankan_enabled=True,
            optional_kakan_enabled=True,
            optional_kyuushu_enabled=True,
        )
        result = ev.evaluate(num_matches=1, seed_start=0, policy_seat=0)
        assert "avg_rank" in result
        assert "avg_score" in result

    def test_evaluate_default_off_runs(self):
        ev = _make_evaluator()
        result = ev.evaluate(num_matches=1, seed_start=0, policy_seat=0)
        assert "avg_rank" in result


# ========== Section 3: TSUMO_OPTIONAL Skip semantics ==========


class TestTsumoOptionalSkipSemantics:
    """TSUMO_OPTIONAL で Skip を選んだ後、同 turn で TsumoWin が
    自動実行されないこと。"""

    def _find_tsumo_optional(self, max_seeds=300):
        """TSUMO_OPTIONAL を出してくれる seed を探す。"""
        from mahjong_rl.env import Stage2Env, DecisionType
        for seed in range(max_seeds):
            env = Stage2Env(observation_mode="full",
                            optional_tsumo_enabled=True)
            env.reset(seed)
            for _ in range(2000):
                if env.decision_type == DecisionType.TSUMO_OPTIONAL:
                    return env, seed
                if env.decision_type == DecisionType.DISCARD:
                    mask, snap = env.get_legal_discard_snapshot()
                    legal = np.where(mask > 0.5)[0]
                    if len(legal) == 0:
                        break
                    env.step_discard_with_snapshot(int(legal[0]), snap)
                elif env.decision_type == DecisionType.RESPONSE:
                    env.step_response(0)
                else:
                    # 他の optional は default off なので来ないが念のため
                    env.step_response(0)
                if env._done:
                    break
        return None, None

    def test_skip_does_not_auto_tsumo(self):
        from mahjong_rl.env import DecisionType
        env, seed = self._find_tsumo_optional()
        if env is None:
            pytest.skip("no TSUMO_OPTIONAL seed found in 300 seeds")

        # candidates: [Win=0, Skip=1]
        round_number_before = env.env_state.match_state.round_number
        scores_before = list(env.env_state.match_state.scores)

        # Skip = idx 1
        candidates = env.response_candidates
        assert len(candidates) >= 2
        env.step_response(1)  # Skip

        # round が終わっていないこと
        assert env._done is False or env.decision_type != DecisionType.TSUMO_OPTIONAL
        # 同 round / 同 score で続行 (自動 TsumoWin が走っていない)
        assert env.env_state.match_state.round_number == round_number_before
        assert list(env.env_state.match_state.scores) == scores_before

    def test_skip_falls_through_to_discard(self):
        """Skip 後は同 player の DISCARD decision に進む。"""
        from mahjong_rl.env import DecisionType
        env, seed = self._find_tsumo_optional()
        if env is None:
            pytest.skip("no TSUMO_OPTIONAL seed found in 300 seeds")

        player_before = env.current_player
        env.step_response(1)  # Skip

        # 自動 TsumoWin が抑止されているので、SelfActionPhase で discard 待ち
        # (RESPONSE 経由で巡回 player に戻る場合もあるが、
        #  少なくとも TSUMO_OPTIONAL に再入は起きない)
        assert env.decision_type != DecisionType.TSUMO_OPTIONAL
        # tsumo が skip set に含まれていること
        assert "tsumo" in env._optional_skipped_this_turn

    def test_default_off_auto_tsumo_still_works(self):
        """default off (optional_tsumo_enabled=False) では
        既存の自動 TsumoWin 挙動が維持される。"""
        from mahjong_rl.env import Stage2Env, DecisionType
        # default off で 1 game を回しても crash しないこと、
        # かつ TSUMO_OPTIONAL が来ないこと。
        env = Stage2Env(observation_mode="full")  # 全 optional default off
        env.reset(0)
        saw_tsumo_optional = False
        for _ in range(2000):
            if env.decision_type == DecisionType.TSUMO_OPTIONAL:
                saw_tsumo_optional = True
                break
            if env.decision_type == DecisionType.DISCARD:
                mask, snap = env.get_legal_discard_snapshot()
                legal = np.where(mask > 0.5)[0]
                if len(legal) == 0:
                    break
                env.step_discard_with_snapshot(int(legal[0]), snap)
            elif env.decision_type == DecisionType.RESPONSE:
                env.step_response(0)
            else:
                break
            if env._done:
                break
        assert saw_tsumo_optional is False
        # _optional_skipped_this_turn は使われない
        assert env._optional_skipped_this_turn == set()


# ========== Section 4: runner helper ==========


class TestRunnerOptionalFlagsHelper:
    """Stage2aRunner._stage2a_optional_flags が training.* から flags を抽出。"""

    def _make_runner_stub(self, training_dict):
        """最小限のスタブ runner。"""
        from mahjong_rl.runner import Stage1Runner
        # Stage1Runner.__init__ は重いので直接 attribute を埋める
        runner = Stage1Runner.__new__(Stage1Runner)

        class _C:
            def __init__(self, d):
                self._d = d

            def __getattr__(self, k):
                return self._d.get(k)

            def get(self, k, default=None):
                return self._d.get(k, default)

        runner._config = type("Cfg", (), {"training": _C(training_dict)})()
        return runner

    def test_all_disabled_default(self):
        runner = self._make_runner_stub({})
        flags = runner._stage2a_optional_flags()
        assert flags == {
            "optional_riichi": False,
            "optional_tsumo": False,
            "optional_ron": False,
            "optional_ankan": False,
            "optional_kakan": False,
            "optional_kyuushu": False,
        }

    def test_partial_enabled(self):
        runner = self._make_runner_stub({
            "optional_riichi": {"enabled": True},
            "optional_kakan": {"enabled": True},
        })
        flags = runner._stage2a_optional_flags()
        assert flags["optional_riichi"] is True
        assert flags["optional_kakan"] is True
        assert flags["optional_tsumo"] is False
        assert flags["optional_ron"] is False
        assert flags["optional_ankan"] is False
        assert flags["optional_kyuushu"] is False
