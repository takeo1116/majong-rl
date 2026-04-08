"""CQ-0271: discard snapshot テスト"""
import pytest
import numpy as np

pytestmark = pytest.mark.smoke

from mahjong_rl.env.stage2_env import Stage2Env, DecisionType
from mahjong_rl._mahjong_core import ActionType


class TestDiscardSnapshot:
    """snapshot から作った legal mask と tile_type 解決の整合"""

    def test_snapshot_mask_and_resolve_consistent(self):
        """snapshot から作った mask 上の tile_type は必ず解決できる"""
        env = Stage2Env(observation_mode="full")
        env.reset(42)
        steps = 0
        for _ in range(5000):
            if env.decision_type != DecisionType.DISCARD:
                break
            mask, snap = env.get_legal_discard_snapshot()
            # mask 上の全 tile_type が snapshot から解決できる
            for t in range(34):
                if mask[t] > 0.5:
                    action = env._resolve_discard_from_snapshot(t, snap)
                    assert action.type == ActionType.Discard
                    assert action.tile // 4 == t
            # 最初の legal tile_type で進める
            tile_type = int(np.argmax(mask))
            _, _, terminated, _, _ = env.step_discard_with_snapshot(
                tile_type, snap)
            steps += 1
            if terminated:
                break
        assert steps > 0

    def test_riichi_discard_preferred(self):
        """立直打牌がある場合は riichi=True の action が返る"""
        env = Stage2Env(observation_mode="full")
        # Run until we might find riichi
        for seed in range(100):
            env.reset(seed)
            for _ in range(5000):
                if env.decision_type != DecisionType.DISCARD:
                    # skip response
                    candidates = env.response_candidates
                    _, _, terminated, _, _ = env.step_response(
                        len(candidates) - 1)  # Skip
                    if terminated:
                        break
                    continue
                mask, snap = env.get_legal_discard_snapshot()
                riichi_tiles = [a for a in snap if a.riichi]
                if riichi_tiles:
                    # riichi tile_type を選んで resolve
                    rt = riichi_tiles[0].tile // 4
                    action = env._resolve_discard_from_snapshot(rt, snap)
                    assert action.riichi is True
                    return  # test passed
                # 普通に進める
                tile_type = int(np.argmax(mask))
                _, _, terminated, _, _ = env.step_discard_with_snapshot(
                    tile_type, snap)
                if terminated:
                    break
        pytest.skip("100 seed で riichi が出なかった")


class TestSnapshotDiagnostics:
    """例外時の診断情報"""

    def test_error_contains_diagnostics(self):
        """snapshot に存在しない tile_type で解決するとエラーに情報が含まれる"""
        env = Stage2Env(observation_mode="full")
        env.reset(42)
        if env.decision_type != DecisionType.DISCARD:
            pytest.skip()
        mask, snap = env.get_legal_discard_snapshot()
        # mask が 0 の tile_type を探す
        illegal_tile = -1
        for t in range(34):
            if mask[t] < 0.5:
                illegal_tile = t
                break
        if illegal_tile < 0:
            pytest.skip("全牌種が合法")
        with pytest.raises(ValueError, match="snapshot 不整合"):
            env._resolve_discard_from_snapshot(illegal_tile, snap)

    def test_legacy_resolve_error_has_info(self):
        """旧 _resolve_discard も診断情報付き"""
        env = Stage2Env(observation_mode="full")
        env.reset(42)
        if env.decision_type != DecisionType.DISCARD:
            pytest.skip()
        mask = env.get_legal_mask()
        illegal_tile = -1
        for t in range(34):
            if mask[t] < 0.5:
                illegal_tile = t
                break
        if illegal_tile < 0:
            pytest.skip()
        with pytest.raises(ValueError, match="legal discard tile_types"):
            env.step_discard(illegal_tile)


class TestSelfplaySmoke:
    """selfplay が snapshot path で通る"""

    def test_selfplay_no_crash(self, tmp_path):
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        stats = worker.generate(num_matches=5, base_seed=42)
        assert stats["discard_count"] > 0
        assert stats["call_count"] > 0


class TestEvalSmoke:
    """eval が snapshot path で通る"""

    def test_eval_no_crash(self):
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
        from mahjong_rl.models.stage2a_model import Stage2aModel
        from mahjong_rl.encoders import FlatFeatureEncoder
        import torch

        encoder = FlatFeatureEncoder(observation_mode="full")
        dim = encoder.metadata().output_shape[0]
        model = Stage2aModel(input_dim=dim, discard_hidden_dims=[32],
                              optional_hidden_dims=[32])
        evaluator = Stage2aEvaluator(
            model=model, encoder=encoder,
            device=torch.device("cpu"))
        result = evaluator.evaluate(num_matches=2, seed_start=42)
        assert result["total_rounds"] > 0
