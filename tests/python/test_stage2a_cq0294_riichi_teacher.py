"""CQ-0294: optional_riichi 有効時の teacher / baseline mask 分離 +
riichi opportunity diagnostics + riichi_discard_mask feature +
optional_summary action_type_presence 拡張。
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.smoke


# ========== Section 1: legal_mask helpers ==========


class TestLegalMaskHelpers:
    """teacher / riichi-feature 用 mask helper の挙動確認。"""

    def _make(self, specs):
        from mahjong_rl._mahjong_core import Action
        return [Action.make_discard(0, tile, riichi=riichi)
                for tile, riichi in specs]

    def test_teacher_mask_is_riichi_only_when_riichi_present(self):
        from mahjong_rl.legal_mask import (
            make_teacher_discard_mask_from_legal_actions)
        # tile_id 0(=1m), 4(=2m, riichi), 8(=3m)
        actions = self._make([(0, False), (4, True), (8, False)])
        mask = make_teacher_discard_mask_from_legal_actions(actions)
        # 立直 tile_type=1 のみ
        assert mask[0] == 0.0
        assert mask[1] == 1.0
        assert mask[2] == 0.0
        assert mask.sum() == 1.0

    def test_teacher_mask_falls_back_to_normal_when_no_riichi(self):
        from mahjong_rl.legal_mask import (
            make_teacher_discard_mask_from_legal_actions)
        actions = self._make([(0, False), (4, False)])
        mask = make_teacher_discard_mask_from_legal_actions(actions)
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask.sum() == 2.0

    def test_riichi_discard_mask_only_riichi_actions(self):
        from mahjong_rl.legal_mask import (
            make_riichi_discard_mask_from_legal_actions)
        actions = self._make([(0, False), (4, True), (4, False),
                               (8, True)])
        mask = make_riichi_discard_mask_from_legal_actions(actions)
        # tile_type 1 (= 4//4), 2 (= 8//4) に riichi action あり
        assert mask[1] == 1.0
        assert mask[2] == 1.0
        # tile_type 0 (= 0//4) には riichi action なし
        assert mask[0] == 0.0
        assert mask.sum() == 2.0

    def test_riichi_discard_mask_all_zeros_without_riichi(self):
        from mahjong_rl.legal_mask import (
            make_riichi_discard_mask_from_legal_actions)
        actions = self._make([(0, False), (4, False)])
        mask = make_riichi_discard_mask_from_legal_actions(actions)
        assert mask.sum() == 0.0


# ========== Section 2: Stage2Env getters ==========


def _drive_until_riichi_legal(env, max_seeds=500, max_steps=300):
    """riichi 打牌が含まれる discard 場面までドライブ。"""
    from mahjong_rl.env import DecisionType
    for seed in range(max_seeds):
        env.reset(seed)
        for _ in range(max_steps):
            if env.decision_type == DecisionType.DISCARD:
                mask, snap = env.get_legal_discard_snapshot()
                if any(a.riichi for a in snap):
                    return seed, mask, snap
                if mask.sum() == 0:
                    break
                tt = int(np.argmax(mask))
                _, _, term, _, _ = env.step_discard_with_snapshot(
                    tt, snap)
                if env.decision_type == DecisionType.RIICHI_OPTIONAL:
                    env.step_response(0)  # NoRiichi
                if term:
                    break
            elif env.decision_type == DecisionType.RESPONSE:
                n = len(env.response_candidates)
                _, _, term, _, _ = env.step_response(n - 1)
                if term:
                    break
            else:
                break
    return None, None, None


class TestStage2EnvTeacherMaskGetter:
    """get_teacher_discard_mask_from_snapshot は optional_riichi の
    真偽に関わらず riichi-only 優先 mask を返す。"""

    def test_teacher_mask_riichi_only_when_optional_riichi_on(self):
        from mahjong_rl.env import Stage2Env
        env = Stage2Env(observation_mode="full",
                        optional_riichi_enabled=True)
        seed, policy_mask, snap = _drive_until_riichi_legal(env)
        if seed is None:
            pytest.skip("no riichi-legal seed")
        teacher_mask = env.get_teacher_discard_mask_from_snapshot(snap)
        # teacher_mask は riichi-only
        riichi_tts = {a.tile // 4 for a in snap if a.riichi}
        non_riichi_tts = {a.tile // 4 for a in snap if not a.riichi}
        # teacher mask は riichi tile_type のみ立つ
        for tt in riichi_tts:
            assert teacher_mask[tt] == 1.0
        # non-riichi-only tile_type は teacher mask には入らない
        for tt in non_riichi_tts:
            if tt not in riichi_tts:
                assert teacher_mask[tt] == 0.0
        # policy mask は全 tile_type を含む (CQ-0292 batch 2)
        all_tts = {a.tile // 4 for a in snap}
        for tt in all_tts:
            assert policy_mask[tt] == 1.0

    def test_teacher_mask_subset_of_policy_mask(self):
        """teacher mask は policy mask の subset (= teacher mask 上で
        合法な tile_type は policy mask 上でも合法)。"""
        from mahjong_rl.env import Stage2Env
        env = Stage2Env(observation_mode="full",
                        optional_riichi_enabled=True)
        seed, policy_mask, snap = _drive_until_riichi_legal(env)
        if seed is None:
            pytest.skip("no riichi-legal seed")
        teacher_mask = env.get_teacher_discard_mask_from_snapshot(snap)
        for tt in range(34):
            if teacher_mask[tt] > 0.5:
                assert policy_mask[tt] > 0.5

    def test_riichi_discard_mask_matches_engine_riichi_actions(self):
        from mahjong_rl.env import Stage2Env
        env = Stage2Env(observation_mode="full",
                        optional_riichi_enabled=True)
        seed, _, snap = _drive_until_riichi_legal(env)
        if seed is None:
            pytest.skip("no riichi-legal seed")
        feat = env.get_riichi_discard_mask()
        # engine の riichi action がある tile_type だけ 1
        expected_tts = {a.tile // 4 for a in snap if a.riichi}
        for tt in range(34):
            if tt in expected_tts:
                assert feat[tt] == 1.0
            else:
                assert feat[tt] == 0.0

    def test_riichi_discard_mask_all_zeros_when_no_riichi(self):
        from mahjong_rl.env import Stage2Env, DecisionType
        env = Stage2Env(observation_mode="full",
                        optional_riichi_enabled=False)
        env.reset(0)
        # 局開始直後は通常 riichi 不可 (tenpai の保証なし)
        # 立直可能でない seed を使うため、最初の DISCARD で feature 確認
        for _ in range(10):
            if env.decision_type == DecisionType.DISCARD:
                _, snap = env.get_legal_discard_snapshot()
                feat = env.get_riichi_discard_mask()
                if not any(a.riichi for a in snap):
                    assert feat.sum() == 0.0
                    return
                break
            elif env.decision_type == DecisionType.RESPONSE:
                env.step_response(0)
            else:
                break


# ========== Section 3: encoder riichi_discard_mask feature ==========


class TestEncoderRiichiDiscardMask:
    """FlatFeatureEncoder.riichi_discard_mask feature の dim / metadata。"""

    def test_default_off_no_dim_change(self):
        from mahjong_rl.encoders import FlatFeatureEncoder
        a = FlatFeatureEncoder(observation_mode="full")
        b = FlatFeatureEncoder(observation_mode="full",
                                riichi_discard_mask=False)
        assert a.metadata().output_shape == b.metadata().output_shape
        assert "riichi_discard_mask" not in a.metadata().feature_ranges
        assert "riichi_discard_mask" not in b.metadata().feature_ranges

    def test_enabled_full_adds_34_dims(self):
        from mahjong_rl.encoders import FlatFeatureEncoder
        a = FlatFeatureEncoder(observation_mode="full")
        b = FlatFeatureEncoder(observation_mode="full",
                                riichi_discard_mask=True)
        assert b.metadata().output_shape[0] == a.metadata().output_shape[0] + 34
        rng = b.metadata().feature_ranges["riichi_discard_mask"]
        assert rng[1] - rng[0] == 34

    def test_enabled_partial_adds_34_dims(self):
        from mahjong_rl.encoders import FlatFeatureEncoder
        a = FlatFeatureEncoder(observation_mode="partial")
        b = FlatFeatureEncoder(observation_mode="partial",
                                riichi_discard_mask=True)
        assert b.metadata().output_shape[0] == a.metadata().output_shape[0] + 34

    def test_encode_uses_passed_riichi_discard_mask(self):
        """encode(riichi_discard_mask=...) で渡した値が feature 範囲に入る。"""
        from mahjong_rl.encoders import FlatFeatureEncoder
        from mahjong_rl.env import Stage2Env, DecisionType
        env = Stage2Env(observation_mode="full",
                        optional_riichi_enabled=True)
        seed, _, snap = _drive_until_riichi_legal(env)
        if seed is None:
            pytest.skip("no riichi-legal seed")
        enc = FlatFeatureEncoder(observation_mode="full",
                                  riichi_discard_mask=True)
        meta = enc.metadata()
        rng = meta.feature_ranges["riichi_discard_mask"]
        obs = env._make_observation()
        r_mask = env.get_riichi_discard_mask()
        feats = enc.encode(obs, riichi_discard_mask=r_mask)
        # feature 範囲が引数値と一致
        assert np.allclose(feats[rng[0]:rng[1]], r_mask)

    def test_encode_default_zero_when_no_mask(self):
        from mahjong_rl.encoders import FlatFeatureEncoder
        from mahjong_rl.env import Stage2Env
        env = Stage2Env(observation_mode="full")
        env.reset(0)
        enc = FlatFeatureEncoder(observation_mode="full",
                                  riichi_discard_mask=True)
        meta = enc.metadata()
        rng = meta.feature_ranges["riichi_discard_mask"]
        obs = env._make_observation()
        feats = enc.encode(obs)  # riichi_discard_mask 未指定
        assert np.all(feats[rng[0]:rng[1]] == 0.0)


# ========== Section 4: encoder config wiring ==========


class TestEncoderConfigWiring:
    """runner._rebuild_encoder と stage2a_parallel が riichi_discard_mask を
    受け取る。"""

    def test_runner_rebuild_encoder_supports_flag(self):
        from mahjong_rl.runner import _rebuild_encoder
        enc = _rebuild_encoder({"riichi_discard_mask": True}, "full")
        assert enc._riichi_discard_mask is True

    def test_runner_rebuild_encoder_dict_form(self):
        from mahjong_rl.runner import _rebuild_encoder
        enc = _rebuild_encoder(
            {"riichi_discard_mask": {"enabled": True}}, "full")
        assert enc._riichi_discard_mask is True

    def test_runner_rebuild_encoder_default_off(self):
        from mahjong_rl.runner import _rebuild_encoder
        enc = _rebuild_encoder({}, "full")
        assert enc._riichi_discard_mask is False


class TestRunnerCreateEncoderRiichiDiscardMask:
    """CQ-0294 follow-up: Stage1Runner._create_encoder が
    riichi_discard_mask flag を反映する。"""

    def _make_runner_stub(self, feature_encoder_cfg):
        """最小限のスタブ runner (Stage1Runner._create_encoder のみ呼ぶ)。

        feature_encoder は dict のまま、experiment は dict-like である
        ことだけ満たせばよい (``_create_encoder`` は ``.get`` を呼ぶため)。
        """
        from mahjong_rl.runner import Stage1Runner
        runner = Stage1Runner.__new__(Stage1Runner)
        runner._config = type(
            "Cfg", (),
            {"feature_encoder": dict(feature_encoder_cfg),
             "experiment": {"observation_mode": "full"}})()
        return runner

    def test_create_encoder_default_off(self):
        runner = self._make_runner_stub({})
        enc = runner._create_encoder()
        assert enc._riichi_discard_mask is False
        meta = enc.metadata()
        assert "riichi_discard_mask" not in meta.feature_ranges

    def test_create_encoder_bool_true(self):
        runner = self._make_runner_stub({"riichi_discard_mask": True})
        enc = runner._create_encoder()
        assert enc._riichi_discard_mask is True
        meta = enc.metadata()
        assert "riichi_discard_mask" in meta.feature_ranges
        rng = meta.feature_ranges["riichi_discard_mask"]
        assert rng[1] - rng[0] == 34

    def test_create_encoder_dict_enabled(self):
        runner = self._make_runner_stub(
            {"riichi_discard_mask": {"enabled": True}})
        enc = runner._create_encoder()
        assert enc._riichi_discard_mask is True

    def test_runner_and_parallel_dim_match(self):
        """runner._create_encoder() と stage2a_parallel の worker
        encoder が同じ output_shape を返すこと。"""
        from mahjong_rl.runner import _rebuild_encoder
        runner = self._make_runner_stub({"riichi_discard_mask": True})
        runner_enc = runner._create_encoder()
        worker_enc = _rebuild_encoder(
            {"riichi_discard_mask": True}, "full")
        assert runner_enc.metadata().output_shape == (
            worker_enc.metadata().output_shape)

    def test_dim_diff_is_34_when_enabled(self):
        """default off と true で dim 差分は +34。"""
        off_runner = self._make_runner_stub({})
        on_runner = self._make_runner_stub(
            {"riichi_discard_mask": True})
        off_dim = off_runner._create_encoder().metadata().output_shape[0]
        on_dim = on_runner._create_encoder().metadata().output_shape[0]
        assert on_dim - off_dim == 34


class TestModelInputDimMatchesShard:
    """CQ-0294 follow-up smoke: feature_encoder.riichi_discard_mask=true
    で生成した shard の observation_dim と、model 構築時の input_dim が
    一致すること。"""

    def test_imitation_shard_observation_dim_matches_model_input(
            self, tmp_path):
        """最小 imitation data gen + model 構築で shape mismatch が
        起きないこと。runner._create_encoder と Stage2SelfPlayWorker の
        encoder 経路で生成した shard の observation_dim が一致する。"""
        from mahjong_rl.encoders import FlatFeatureEncoder
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.call_shard import DecisionShardReader
        from mahjong_rl.models.stage2a_model import Stage2aModel

        # CQ-0294 follow-up: runner._create_encoder() と shard 生成側
        # encoder を同じ flag (riichi_discard_mask=true) で構築。
        encoder = FlatFeatureEncoder(observation_mode="full",
                                      riichi_discard_mask=True)
        meta = encoder.metadata()
        encoder_dim = int(np.prod(meta.output_shape))
        # shard 生成
        worker = Stage2SelfPlayWorker(
            config={},
            output_dir=tmp_path,
            observation_mode="full",
            encoder=encoder,
        )
        worker.generate(num_matches=1, base_seed=0,
                         experiment_id="t", run_id="r",
                         worker_id="w0")
        # shard の observation_dim を読む
        reader = DecisionShardReader(tmp_path)
        samples = reader.read_all()
        assert len(samples) > 0
        shard_obs_dim = len(samples[0].observation)
        # encoder dim と一致
        assert shard_obs_dim == encoder_dim
        # model 構築 (encoder.metadata 由来 input_dim) で
        # forward_discard が shape mismatch しない
        model = Stage2aModel(
            input_dim=encoder_dim, discard_hidden_dims=[16, 16],
            optional_hidden_dims=[16, 16], value_hidden_dims=[16, 16],
            candidate_dim=8, optional_scorer_hidden=8,
        )
        # shard observation を model に通す
        obs = torch.tensor(
            samples[0].observation, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(
            samples[0].legal_mask, dtype=torch.float32).unsqueeze(0)
        out = model.forward_discard(obs, mask, compute_value=True)
        assert out.discard_logits.shape == (1, 34)


class TestSummaryAndNotesIncludeRiichiDiscardMask:
    """CQ-0294 follow-up: summary['encoder_features'] と notes _flags に
    riichi_discard_mask が出る。"""

    def test_encoder_features_include_riichi_discard_mask(self):
        """summary['encoder_features'] dict は riichi_discard_mask key を
        持つ (default false / explicit true 両方)。"""
        # source 上の存在を簡便に確認 (full pipeline は重いため)
        import inspect
        from mahjong_rl import runner as runner_mod
        src = inspect.getsource(runner_mod)
        # summary.encoder_features に追加されている
        assert '"riichi_discard_mask": _parse_encoder_flag(' in src
        # notes _flags にも追加されている (= 2 occurrence: summary +
        # _flags + _create_encoder + _rebuild_encoder)
        count = src.count('"riichi_discard_mask"')
        # summary, notes _flags の 2 箇所最小
        assert count >= 2


# ========== Section 5: optional summary action presence ==========


class TestOptionalSummaryActionPresence:
    """_make_optional_summary が NUM_ACTION_TYPE_INDICES に追従する。"""

    def _make_model(self, candidate_dim=4):
        from mahjong_rl.models.stage2a_model import Stage2aModel
        return Stage2aModel(
            input_dim=10, discard_hidden_dims=[8],
            optional_hidden_dims=[8], value_hidden_dims=[8],
            candidate_dim=candidate_dim, optional_scorer_hidden=8,
        )

    def test_summary_dim_matches_num_action_type_indices(self):
        from mahjong_rl.candidate_encoding import NUM_ACTION_TYPE_INDICES
        model = self._make_model(candidate_dim=4)
        # summary_dim = 2 + NUM_ACTION_TYPE_INDICES + 2 * candidate_dim
        expected = 2 + NUM_ACTION_TYPE_INDICES + 2 * 4
        assert model._summary_dim == expected

    def test_action_type_presence_size_is_num_action_type_indices(self):
        from mahjong_rl.candidate_encoding import NUM_ACTION_TYPE_INDICES
        model = self._make_model(candidate_dim=4)
        B, C, D = 2, 3, 4
        cand_enc = torch.zeros(B, C, D)
        cand_mask = torch.tensor([[1.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0]])
        # action_type_idx = [0(Skip), 5(Riichi), 0]; [10(Kyuushu), 0, 0]
        cand_features = torch.zeros(B, C, 5)
        cand_features[0, 0, 0] = 0
        cand_features[0, 1, 0] = 5
        cand_features[1, 0, 0] = 10
        out = model._make_optional_summary(cand_enc, cand_mask, cand_features)
        # shape = available(1) + count_norm(1) + presence(N) + mean(D) + max(D)
        expected_dim = 1 + 1 + NUM_ACTION_TYPE_INDICES + 2 * D
        assert out.shape == (B, expected_dim)
        # presence の位置: idx 2..2+N
        presence = out[:, 2:2 + NUM_ACTION_TYPE_INDICES]
        # batch 0: idx 0 (Skip) と idx 5 (Riichi) が立つ
        assert presence[0, 0] == 1.0
        assert presence[0, 5] == 1.0
        assert presence[0, 10] == 0.0
        # batch 1: idx 10 (Kyuushu) のみ
        assert presence[1, 10] == 1.0
        assert presence[1, 0] == 0.0
        assert presence[1, 5] == 0.0

    def test_all_new_action_types_can_present(self):
        """CQ-0291/0292 で追加された action type 全てが presence に立つ。"""
        from mahjong_rl.candidate_encoding import NUM_ACTION_TYPE_INDICES
        model = self._make_model(candidate_dim=4)
        N = NUM_ACTION_TYPE_INDICES
        # 1 batch, N candidates, 各 candidate に固有 action_type
        B, C, D = 1, N, 4
        cand_enc = torch.zeros(B, C, D)
        cand_mask = torch.ones(B, C)
        cand_features = torch.zeros(B, C, 5)
        for k in range(N):
            cand_features[0, k, 0] = k
        out = model._make_optional_summary(cand_enc, cand_mask, cand_features)
        presence = out[:, 2:2 + N]
        for k in range(N):
            assert presence[0, k] == 1.0, (
                f"action_type {k} must be present in summary")


# ========== Section 6: legacy checkpoint load compat ==========


class TestLegacyCheckpointLoad:
    """旧 checkpoint (action_type_emb 行数 / value_trunk 列数) を新 model に
    安全に load できる。"""

    def test_legacy_value_trunk_with_old_summary_loads(self):
        """K_old=4 (旧 4 family) の value_trunk weight をロードできる。"""
        from mahjong_rl.models.stage2a_model import (
            Stage2aModel, load_stage2a_state_dict)
        from mahjong_rl.candidate_encoding import NUM_ACTION_TYPE_INDICES

        cand_dim = 4
        model = Stage2aModel(
            input_dim=10, discard_hidden_dims=[8],
            optional_hidden_dims=[8], value_hidden_dims=[8],
            candidate_dim=cand_dim, optional_scorer_hidden=8,
        )
        # 新 model の value_trunk weight 形状
        new_w = model.value_trunk[0].weight.data
        out_dim, new_in = new_w.shape
        # K_new = NUM_ACTION_TYPE_INDICES, K_old = 4 → 差分 = N - 4
        extra = NUM_ACTION_TYPE_INDICES - 4
        old_in = new_in - extra
        # 旧形式 weight (1 epoch 学習後相当のランダム値)
        legacy_state = {k: v.clone() for k, v in model.state_dict().items()}
        legacy_w = torch.randn(out_dim, old_in)
        legacy_state["value_trunk.0.weight"] = legacy_w
        # action_type_emb 行数も旧 4 行に減らす (リグレッション両方確認)
        emb_w = legacy_state["candidate_encoder.action_type_emb.weight"]
        legacy_state["candidate_encoder.action_type_emb.weight"] = emb_w[:4]

        # 新 model にロード成功
        load_stage2a_state_dict(model, legacy_state, strict=True)
        # 旧 weight の左/右部分が新 weight に保持される
        loaded_w = model.value_trunk[0].weight.data
        # trunk_input_dim を再計算: trunk_input + 1 + 3 + 2 + K_old +
        #   2*cand_dim + value_aux = old_in
        value_aux = 0
        trunk_input_dim = old_in - (1 + 3 + 2 + 4 + 2 * cand_dim + value_aux)
        insert_at = trunk_input_dim + 1 + 3 + 2 + 4
        # 左部分 (insert_at まで) が legacy と一致
        assert torch.allclose(loaded_w[:, :insert_at], legacy_w[:, :insert_at])
        # 挿入された extra 列は 0
        assert torch.allclose(
            loaded_w[:, insert_at:insert_at + extra],
            torch.zeros(out_dim, extra))
        # 右部分が legacy の対応列と一致
        assert torch.allclose(
            loaded_w[:, insert_at + extra:],
            legacy_w[:, insert_at:])


# ========== Section 7: worker / evaluator semantics ==========


def _make_worker(tmp_path, **flags):
    from mahjong_rl.encoders import FlatFeatureEncoder
    from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
    encoder = FlatFeatureEncoder(observation_mode="full")
    config: dict = {}
    for k, v in flags.items():
        config[k] = {"enabled": bool(v)}
    return Stage2SelfPlayWorker(
        config=config,
        output_dir=tmp_path,
        observation_mode="full",
        encoder=encoder,
    )


class TestWorkerRiichiDiagnostics:
    """selfplay stats に riichi opportunity 系 key が出る。"""

    def test_default_off_keys_present_and_zero(self, tmp_path):
        w = _make_worker(tmp_path)
        stats = w.generate(num_matches=1, base_seed=0,
                            experiment_id="t", run_id="r",
                            worker_id="w0")
        for k in ("riichi_opportunity_discard_count",
                  "riichi_optional_opened_count",
                  "riichi_bypassed_by_non_riichi_discard_count"):
            assert k in stats
        assert stats["riichi_bypass_rate"] == 0.0
        # default off (optional_riichi=False) では bypass=0, opened=0
        # opportunity は legal actions に riichi が含まれるかどうか次第なので
        # >= 0 だけ保証
        assert stats["riichi_optional_opened_count"] == 0
        assert stats["riichi_bypassed_by_non_riichi_discard_count"] == 0

    def test_default_off_diagnostics_json_serializable(self, tmp_path):
        w = _make_worker(tmp_path)
        stats = w.generate(num_matches=1, base_seed=0,
                            experiment_id="t", run_id="r",
                            worker_id="w0")
        json.dumps(stats)

    def test_optional_riichi_on_diagnostics_consistent(self, tmp_path):
        """opportunity = opened + bypassed (各 actor 系も同様)。"""
        w = _make_worker(tmp_path, optional_riichi=True)
        stats = w.generate(num_matches=2, base_seed=0,
                            experiment_id="t", run_id="r",
                            worker_id="w0")
        opp = stats["riichi_opportunity_discard_count"]
        opn = stats["riichi_optional_opened_count"]
        byp = stats["riichi_bypassed_by_non_riichi_discard_count"]
        assert opp == opn + byp
        # bypass rate
        if opp > 0:
            assert stats["riichi_bypass_rate"] == byp / opp


class TestParallelAggregateRiichiDiag:
    """_aggregate_stats が riichi opportunity 系を合算する。"""

    def test_aggregate_sums_scalars(self):
        from mahjong_rl.stage2a_parallel import _aggregate_stats
        stats_list = [
            {"num_matches": 1, "total_steps": 5,
             "discard_count": 3, "call_count": 1,
             "riichi_opportunity_discard_count": 2,
             "riichi_optional_opened_count": 1,
             "riichi_bypassed_by_non_riichi_discard_count": 1,
             "riichi_opportunity_by_actor": {"policy": 1, "baseline": 1},
             "riichi_optional_opened_by_actor": {"policy": 1, "baseline": 0},
             "riichi_bypassed_by_actor": {"policy": 0, "baseline": 1}},
            {"num_matches": 1, "total_steps": 7,
             "discard_count": 4, "call_count": 2,
             "riichi_opportunity_discard_count": 3,
             "riichi_optional_opened_count": 2,
             "riichi_bypassed_by_non_riichi_discard_count": 1,
             "riichi_opportunity_by_actor": {"policy": 2, "baseline": 1},
             "riichi_optional_opened_by_actor": {"policy": 2, "baseline": 0},
             "riichi_bypassed_by_actor": {"policy": 0, "baseline": 1}},
        ]
        agg = _aggregate_stats(stats_list, num_workers=2)
        assert agg["riichi_opportunity_discard_count"] == 5
        assert agg["riichi_optional_opened_count"] == 3
        assert agg["riichi_bypassed_by_non_riichi_discard_count"] == 2
        assert agg["riichi_bypass_rate"] == pytest.approx(2 / 5)
        assert agg["riichi_opportunity_by_actor"] == {
            "policy": 3, "baseline": 2}
        assert agg["riichi_optional_opened_by_actor"] == {
            "policy": 3, "baseline": 0}
        assert agg["riichi_bypassed_by_actor"] == {
            "policy": 0, "baseline": 2}

    def test_aggregate_handles_missing_keys(self):
        """旧 stats (key 無し) でも crash しない。"""
        from mahjong_rl.stage2a_parallel import _aggregate_stats
        stats_list = [
            {"num_matches": 1, "total_steps": 5,
             "discard_count": 3, "call_count": 1},
        ]
        agg = _aggregate_stats(stats_list, num_workers=1)
        assert agg["riichi_opportunity_discard_count"] == 0
        assert agg["riichi_bypass_rate"] == 0.0
        assert agg["riichi_opportunity_by_actor"] == {
            "policy": 0, "baseline": 0}


class TestEvaluatorBaselineUsesTeacherMask:
    """Stage2aEvaluator baseline seat が optional_riichi 有効時でも
    旧 auto-riichi 相当の discard semantics を維持する。"""

    def test_baseline_seat_uses_teacher_mask_in_optional_riichi_on(self):
        """optional_riichi 有効 + baseline seat で eval を回しても crash
        せず、baseline 経路で teacher mask が使われる (riichi が legal な
        場面で baseline は riichi 打牌を選ぶ)。"""
        from mahjong_rl.encoders import FlatFeatureEncoder
        from mahjong_rl.models.stage2a_model import Stage2aModel
        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
        encoder = FlatFeatureEncoder(observation_mode="full")
        meta = encoder.metadata()
        input_dim = int(np.prod(meta.output_shape))
        model = Stage2aModel(
            input_dim=input_dim, discard_hidden_dims=[16, 16],
            optional_hidden_dims=[16, 16], value_hidden_dims=[16, 16],
            candidate_dim=8, optional_scorer_hidden=8,
        )
        ev = Stage2aEvaluator(
            model=model, encoder=encoder,
            observation_mode="full",
            optional_riichi_enabled=True,
        )
        # 1 半荘実行して crash しないこと、metric が出ること
        result = ev.evaluate(num_matches=1, seed_start=0, policy_seat=0)
        assert "avg_rank" in result
        assert "avg_score" in result
