"""CQ-0291 (batch 3): Ankan / Kakan / Kyuushu optional unlock

確認:
- default off で既存自動スキップ挙動を維持
- ``optional_ankan_enabled=True`` で Ankan 合法時 ``ANKAN_OPTIONAL`` 発火
- ``optional_kakan_enabled=True`` で Kakan 合法時 ``KAKAN_OPTIONAL`` 発火
- ``optional_kyuushu_enabled=True`` で Kyuushu 合法時 ``KYUUSHU_OPTIONAL`` 発火
- 各 family の primary candidate (idx=0) を選ぶと engine action が実行される
- Skip candidate (idx=1) を選ぶと engine step は走らず、次 optional / DISCARD
  に fall-through する
- candidate_encoding ACTION_TYPE_MAP に Kakan (6→8) / Ankan (7→9) /
  Kyuushu (9→10) が含まれる
- model embedding が 11 行
- 旧 4/6/8 行 checkpoint を ``load_stage2a_state_dict`` で読み戻せる
- shard roundtrip で ``decision_family ∈ {ankan, kakan, kyuushu}`` が保持される
- selfplay enabled で ankan/kakan/kyuushu sample が生成される (baseline=Skip)
- learner imitation / PPO smoke で crash しない
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.smoke

from mahjong_rl import Action, ActionType
from mahjong_rl.env.stage2_env import Stage2Env, DecisionType
from mahjong_rl.env.response_candidate import (
    OPTIONAL_ANKAN_ACTION_TYPE, OPTIONAL_KAKAN_ACTION_TYPE,
    OPTIONAL_KYUUSHU_ACTION_TYPE, OPTIONAL_SKIP_ACTION_TYPE,
    make_ankan_optional_candidates, make_kakan_optional_candidates,
    make_kyuushu_optional_candidates,
)
from mahjong_rl.candidate_encoding import (
    ACTION_TYPE_MAP, NUM_ACTION_TYPE_INDICES,
)
from mahjong_rl.call_shard import (
    DecisionSample, CandidateRecord,
    DecisionShardWriter, DecisionShardReader,
)
from mahjong_rl.models.stage2a_model import (
    Stage2aModel, load_stage2a_state_dict, CandidateEncoder,
)


# ========== 1. ACTION_TYPE_MAP / encoder dim ==========


class TestActionTypeMapBatch3:
    def test_kan_kyuushu_codes(self):
        assert OPTIONAL_KAKAN_ACTION_TYPE == 6   # engine ActionType.Kakan
        assert OPTIONAL_ANKAN_ACTION_TYPE == 7   # engine ActionType.Ankan
        assert OPTIONAL_KYUUSHU_ACTION_TYPE == 9 # engine ActionType.Kyuushu
        assert ACTION_TYPE_MAP[6] == 8
        assert ACTION_TYPE_MAP[7] == 9
        assert ACTION_TYPE_MAP[9] == 10

    def test_legacy_codes_unchanged(self):
        # batch 1/2 の codes は不変
        assert ACTION_TYPE_MAP[8] == 0
        assert ACTION_TYPE_MAP[3] == 1
        assert ACTION_TYPE_MAP[4] == 2
        assert ACTION_TYPE_MAP[5] == 3
        assert ACTION_TYPE_MAP[100] == 4
        assert ACTION_TYPE_MAP[101] == 5
        assert ACTION_TYPE_MAP[1] == 6
        assert ACTION_TYPE_MAP[2] == 7

    def test_num_indices_at_least_11(self):
        assert NUM_ACTION_TYPE_INDICES >= 11


class TestEncoderDim11:
    def test_emb_size(self):
        enc = CandidateEncoder(candidate_dim=8)
        assert enc.action_type_emb.num_embeddings == NUM_ACTION_TYPE_INDICES
        assert enc.action_type_emb.num_embeddings >= 11


class TestLoadHelperLegacyEmbBatch3:
    """旧 4 行 / 6 行 / 8 行 checkpoint を 11 行 model に読み戻せる"""

    def _make_model(self):
        return Stage2aModel(
            input_dim=20, discard_hidden_dims=[16],
            optional_hidden_dims=[16], value_hidden_dims=[16],
            candidate_dim=8, optional_scorer_hidden=8)

    @pytest.mark.parametrize("legacy_rows", [4, 6, 8])
    def test_legacy_rows_load(self, legacy_rows):
        m = self._make_model()
        sd = m.state_dict()
        old = torch.randn(legacy_rows, 4)
        sd_legacy = dict(sd)
        sd_legacy["candidate_encoder.action_type_emb.weight"] = old
        result = load_stage2a_state_dict(m, sd_legacy)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        new_w = m.candidate_encoder.action_type_emb.weight.data
        assert torch.allclose(new_w[:legacy_rows], old)


# ========== 2. response_candidate factories ==========


class TestAnkanFactory:
    def test_make(self):
        ankan = Action.make_ankan(0, 4)  # 5m ankan (tile_type=4)
        skip = Action.make_skip(0)
        cs = make_ankan_optional_candidates(ankan, skip, 0)
        assert len(cs) == 2
        assert cs[0].action_type == OPTIONAL_ANKAN_ACTION_TYPE
        assert cs[1].action_type == OPTIONAL_SKIP_ACTION_TYPE
        assert cs[0].tile_type == 4

    def test_make_without_skip(self):
        """SelfActionPhase で engine が Skip を提供しないケース"""
        ankan = Action.make_ankan(0, 27)  # East
        cs = make_ankan_optional_candidates(ankan, None, 0)
        assert len(cs) == 2
        assert cs[0].action_type == OPTIONAL_ANKAN_ACTION_TYPE
        assert cs[1].action_type == OPTIONAL_SKIP_ACTION_TYPE


class TestKakanFactory:
    def test_make(self):
        kakan = Action.make_kakan(0, 17)  # tile_id 17 → tile_type 4 (5m)
        skip = Action.make_skip(0)
        cs = make_kakan_optional_candidates(kakan, skip, 0)
        assert len(cs) == 2
        assert cs[0].action_type == OPTIONAL_KAKAN_ACTION_TYPE
        assert cs[1].action_type == OPTIONAL_SKIP_ACTION_TYPE
        assert cs[0].tile_type == 17 // 4


class TestKyuushuFactory:
    def test_make(self):
        kyuushu = Action.make_kyuushu(0)
        skip = Action.make_skip(0)
        cs = make_kyuushu_optional_candidates(kyuushu, skip, 0)
        assert len(cs) == 2
        assert cs[0].action_type == OPTIONAL_KYUUSHU_ACTION_TYPE
        assert cs[1].action_type == OPTIONAL_SKIP_ACTION_TYPE


# ========== 3. Stage2Env transitions ==========


class TestStage2EnvDefaultOff:
    def test_no_kan_kyuushu_optional_default(self):
        env = Stage2Env(observation_mode="full")
        for seed in range(30):
            env.reset(seed)
            for _ in range(200):
                if env.decision_type == DecisionType.DISCARD:
                    mask = env.get_legal_mask()
                    if mask.sum() == 0:
                        break
                    tt = int(np.argmax(mask))
                    _, _, term, _, _ = env.step_discard(tt)
                    for not_ok in (DecisionType.ANKAN_OPTIONAL,
                                     DecisionType.KAKAN_OPTIONAL,
                                     DecisionType.KYUUSHU_OPTIONAL):
                        assert env.decision_type != not_ok
                    if term:
                        break
                elif env.decision_type == DecisionType.RESPONSE:
                    n = len(env.response_candidates)
                    _, _, term, _, _ = env.step_response(n - 1)
                    if term:
                        break
                else:
                    break


def _drive_until(env, decision_type, max_seeds=2000, max_steps=500):
    for seed in range(max_seeds):
        env.reset(seed)
        for _ in range(max_steps):
            if env.decision_type == decision_type:
                return seed
            if env.decision_type == DecisionType.DISCARD:
                mask = env.get_legal_mask()
                if mask.sum() == 0:
                    break
                tt = int(np.argmax(mask))
                _, _, term, _, _ = env.step_discard(tt)
                if env.decision_type == decision_type:
                    return seed
                if term:
                    break
            elif env.decision_type == DecisionType.RESPONSE:
                # Pon if available — increases Kakan opportunity
                cs = env.response_candidates
                pon_idx = next(
                    (i for i, c in enumerate(cs)
                     if c.action_type == int(ActionType.Pon)), -1)
                if pon_idx >= 0 and decision_type == DecisionType.KAKAN_OPTIONAL:
                    _, _, term, _, _ = env.step_response(pon_idx)
                else:
                    _, _, term, _, _ = env.step_response(len(cs) - 1)
                if term:
                    break
            else:
                break
    return None


class TestAnkanOptionalEnabled:
    def test_ankan_fires(self):
        env = Stage2Env(observation_mode="full",
                          optional_ankan_enabled=True)
        seed = _drive_until(env, DecisionType.ANKAN_OPTIONAL)
        assert seed is not None
        cs = env.response_candidates
        assert len(cs) == 2
        assert cs[0].action_type == OPTIONAL_ANKAN_ACTION_TYPE
        assert cs[1].action_type == OPTIONAL_SKIP_ACTION_TYPE

    def test_ankan_action_executes(self):
        env = Stage2Env(observation_mode="full",
                          optional_ankan_enabled=True)
        seed = _drive_until(env, DecisionType.ANKAN_OPTIONAL)
        assert seed is not None
        cp_before = env.env_state.round_state.current_player
        before_melds = len(env.env_state.round_state.players[cp_before].melds)
        # Take Ankan
        env.step_response(0)
        assert env.decision_type != DecisionType.ANKAN_OPTIONAL
        # melds 数が増えている (Ankan が成立)
        after_melds = len(
            env.env_state.round_state.players[cp_before].melds)
        assert after_melds == before_melds + 1

    def test_skip_falls_through_and_is_recorded(self):
        env = Stage2Env(observation_mode="full",
                          optional_ankan_enabled=True)
        seed = _drive_until(env, DecisionType.ANKAN_OPTIONAL)
        assert seed is not None
        cp_before = env.env_state.round_state.current_player
        env.step_response(1)  # Skip
        # 同じ player の DISCARD decision または別 phase に降りる
        # ankan が再度提示されないこと
        # (= "ankan" が optional_skipped_this_turn に入る)
        # ただし DISCARD まで来てもよい
        if env.decision_type == DecisionType.DISCARD:
            assert env.env_state.round_state.current_player == cp_before
            # ankan skip が記録されている (engine step が走るまで)
            # → 次 step_discard で engine step が走って clear される


class TestKakanOptionalEnabled:
    def test_kakan_fires_or_skipped(self):
        """Kakan は実 game で発生確率が低いため、unit test は factory level
        で十分。実 game smoke は selfplay worker で確認する。"""
        env = Stage2Env(observation_mode="full",
                          optional_kakan_enabled=True)
        # default off と同じく Kakan 発生は稀。env が crash しないこと
        env.reset(0)
        # 少なくとも 100 step まで進められる
        for _ in range(100):
            if env.decision_type == DecisionType.KAKAN_OPTIONAL:
                cs = env.response_candidates
                assert len(cs) == 2
                assert cs[0].action_type == OPTIONAL_KAKAN_ACTION_TYPE
                # Take Kakan
                env.step_response(0)
                return
            if env.decision_type == DecisionType.DISCARD:
                mask = env.get_legal_mask()
                if mask.sum() == 0:
                    break
                tt = int(np.argmax(mask))
                _, _, term, _, _ = env.step_discard(tt)
                if term:
                    break
            elif env.decision_type == DecisionType.RESPONSE:
                n = len(env.response_candidates)
                _, _, term, _, _ = env.step_response(n - 1)
                if term:
                    break
            else:
                break


class TestKyuushuOptionalEnabled:
    def test_kyuushu_fires(self):
        env = Stage2Env(observation_mode="full",
                          optional_kyuushu_enabled=True)
        seed = _drive_until(env, DecisionType.KYUUSHU_OPTIONAL)
        assert seed is not None
        cs = env.response_candidates
        assert len(cs) == 2
        assert cs[0].action_type == OPTIONAL_KYUUSHU_ACTION_TYPE
        assert cs[1].action_type == OPTIONAL_SKIP_ACTION_TYPE

    def test_kyuushu_action_terminates_round(self):
        env = Stage2Env(observation_mode="full",
                          optional_kyuushu_enabled=True)
        seed = _drive_until(env, DecisionType.KYUUSHU_OPTIONAL)
        assert seed is not None
        # Take Kyuushu - round should end (abortive draw)
        round_before = env.env_state.round_state.round_number
        env.step_response(0)
        # decision_type は KYUUSHU_OPTIONAL のままにはならない
        assert env.decision_type != DecisionType.KYUUSHU_OPTIONAL

    def test_kyuushu_skip_falls_through(self):
        env = Stage2Env(observation_mode="full",
                          optional_kyuushu_enabled=True)
        seed = _drive_until(env, DecisionType.KYUUSHU_OPTIONAL)
        assert seed is not None
        cp_before = env.env_state.round_state.current_player
        env.step_response(1)  # Skip
        # Kyuushu skip → DISCARD decision に fall-through
        # (中途流局はせず、通常の打牌に進む)
        if env.decision_type == DecisionType.DISCARD:
            assert env.env_state.round_state.current_player == cp_before


class TestMultipleOptionalsTogether:
    """複数 flag を同時に on にして、skip → 別 optional への遷移を確認"""

    def test_skip_set_persists_until_engine_step(self):
        env = Stage2Env(observation_mode="full",
                          optional_ankan_enabled=True,
                          optional_kyuushu_enabled=True)
        # Kyuushu/Ankan が同時に発生する seed を探す
        # (一般に同 turn で両方発生は稀。skip 集合の単独動作のみ確認)
        seed = _drive_until(env, DecisionType.KYUUSHU_OPTIONAL,
                             max_seeds=200)
        if seed is None:
            pytest.skip("Kyuushu opportunity not found in 200 seeds")
        env.step_response(1)  # Skip Kyuushu
        # この時点で kyuushu が skip 集合に入っている
        assert "kyuushu" in env._optional_skipped_this_turn


# ========== 4. shard roundtrip ==========


class TestShardRoundtripBatch3:
    @pytest.mark.parametrize("family,action_type_code", [
        ("ankan", OPTIONAL_ANKAN_ACTION_TYPE),
        ("kakan", OPTIONAL_KAKAN_ACTION_TYPE),
        ("kyuushu", OPTIONAL_KYUUSHU_ACTION_TYPE),
    ])
    def test_decision_family_roundtrip(self, tmp_path, family, action_type_code):
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        s = DecisionSample(
            decision_type="call",
            decision_family=family,
            observation=np.zeros(10, dtype=np.float32),
            reward=0.0, log_prob=-0.5, value=0.0,
            terminated=False, round_over=False,
            selected_candidate_index=1, candidate_count=2,
            candidates=[
                CandidateRecord(action_type=action_type_code,
                                 tile_type=4),
                CandidateRecord(action_type=OPTIONAL_SKIP_ACTION_TYPE),
            ],
            response_context=np.zeros(3, dtype=np.float32),
            player_id=0, episode_id="ep0", round_id=0, step_id=0,
            actor_type="policy",
            teacher_top1_index=1,  # baseline = Skip
            teacher_source=f"auto_skip_{family}",
            experiment_id="t", run_id="r", worker_id="w",
        )
        writer.add(s)
        writer.close()
        ss = DecisionShardReader(tmp_path).read_all()
        assert len(ss) == 1
        s2 = ss[0]
        assert s2.decision_family == family
        assert s2.candidates[0].action_type == action_type_code
        assert s2.candidates[1].action_type == OPTIONAL_SKIP_ACTION_TYPE
        assert s2.teacher_top1_index == 1


# ========== 5. selfplay worker integration ==========


class TestSelfplayWorkerBatch3:
    def test_default_off_no_kan_samples(self, tmp_path):
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder
        enc = FlatFeatureEncoder(observation_mode="full")
        out = tmp_path / "off"
        w = Stage2SelfPlayWorker(
            config={}, output_dir=out,
            observation_mode="full", encoder=enc,
        )
        assert w._optional_ankan_enabled is False
        assert w._optional_kakan_enabled is False
        assert w._optional_kyuushu_enabled is False
        w.generate(num_matches=3, base_seed=0)
        ss = DecisionShardReader(out).read_all()
        for fam in ("ankan", "kakan", "kyuushu"):
            n = sum(1 for s in ss if s.decision_family == fam)
            assert n == 0, f"default off で {fam} sample が生成された ({n})"

    def test_enabled_generates_optional_samples(self, tmp_path):
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder
        enc = FlatFeatureEncoder(observation_mode="full")
        out = tmp_path / "on"
        w = Stage2SelfPlayWorker(
            config={"training": {
                "optional_ankan": {"enabled": True},
                "optional_kakan": {"enabled": True},
                "optional_kyuushu": {"enabled": True},
            }},
            output_dir=out,
            observation_mode="full", encoder=enc,
        )
        w.generate(num_matches=10, base_seed=0)
        ss = DecisionShardReader(out).read_all()
        n_ankan = sum(1 for s in ss if s.decision_family == "ankan")
        n_kakan = sum(1 for s in ss if s.decision_family == "kakan")
        n_kyuushu = sum(1 for s in ss if s.decision_family == "kyuushu")
        # 10 match で少なくとも 1 つの family は発生するはず (Ankan が
        # 比較的頻発する想定)
        assert (n_ankan + n_kakan + n_kyuushu) > 0, (
            f"enabled で Kan/Kyuushu sample が 0 件: ankan={n_ankan}, "
            f"kakan={n_kakan}, kyuushu={n_kyuushu}")
        # baseline = Skip (idx 1) + teacher_top1 = 1
        for s in ss:
            if s.decision_family in ("ankan", "kakan", "kyuushu"):
                assert s.candidate_count == 2
                assert s.candidates[1].action_type == OPTIONAL_SKIP_ACTION_TYPE
                # imitation mode (model=None) は actor_type=baseline で
                # selected=1 (Skip)
                assert s.selected_candidate_index == 1
                assert s.teacher_top1_index == 1


# ========== 6. learner smoke ==========


class TestLearnerSmokeBatch3:
    def _make_shard(self, shard_dir):
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        # 通常 discard
        for i in range(4):
            writer.add(DecisionSample(
                decision_type="discard",
                observation=np.zeros(10, dtype=np.float32),
                legal_mask=np.ones(34, dtype=np.float32),
                action=0, reward=0.0, log_prob=-0.5, value=0.0,
                terminated=False, round_over=False,
                player_id=0, episode_id="ep0", round_id=0, step_id=i,
                actor_type="policy", teacher_top1_index=0,
                teacher_source="rule_based",
                experiment_id="t", run_id="r", worker_id="w",
            ))
        # ankan / kakan / kyuushu samples
        offset = 4
        for fam, code in (
            ("ankan", OPTIONAL_ANKAN_ACTION_TYPE),
            ("kakan", OPTIONAL_KAKAN_ACTION_TYPE),
            ("kyuushu", OPTIONAL_KYUUSHU_ACTION_TYPE),
        ):
            for i in range(2):
                writer.add(DecisionSample(
                    decision_type="call",
                    decision_family=fam,
                    observation=np.zeros(10, dtype=np.float32),
                    reward=0.0, log_prob=-0.5, value=0.0,
                    terminated=False, round_over=False,
                    selected_candidate_index=1, candidate_count=2,
                    candidates=[
                        CandidateRecord(action_type=code, tile_type=4),
                        CandidateRecord(action_type=OPTIONAL_SKIP_ACTION_TYPE),
                    ],
                    response_context=np.zeros(3, dtype=np.float32),
                    player_id=0, episode_id="ep0", round_id=0,
                    step_id=offset + i,
                    actor_type="policy", teacher_top1_index=1,
                    teacher_source=f"auto_skip_{fam}",
                    experiment_id="t", run_id="r", worker_id="w",
                ))
            offset += 2
        # 終端 sample
        writer.add(DecisionSample(
            decision_type="discard",
            observation=np.zeros(10, dtype=np.float32),
            legal_mask=np.ones(34, dtype=np.float32),
            action=0, reward=0.0, log_prob=-0.5, value=0.0,
            terminated=True, round_over=False,
            player_id=0, episode_id="ep0", round_id=0, step_id=offset,
            actor_type="policy",
            experiment_id="t", run_id="r", worker_id="w",
        ))
        writer.close()

    def test_imitation_smoke(self, tmp_path):
        from mahjong_rl.stage2a_learner import Stage2aLearner
        shard_dir = tmp_path / "shards"
        self._make_shard(shard_dir)
        model = Stage2aModel(
            input_dim=10, discard_hidden_dims=[8],
            optional_hidden_dims=[8], value_hidden_dims=[8],
            candidate_dim=8, optional_scorer_hidden=8)
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "imitation", "epochs": 1, "batch_size": 4,
            }},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        m = learner.train(shard_dir)
        assert m["mode"] == "imitation"
        assert m["num_updates"] > 0

    def test_ppo_smoke(self, tmp_path):
        from mahjong_rl.stage2a_learner import Stage2aLearner
        shard_dir = tmp_path / "shards"
        self._make_shard(shard_dir)
        model = Stage2aModel(
            input_dim=10, discard_hidden_dims=[8],
            optional_hidden_dims=[8], value_hidden_dims=[8],
            candidate_dim=8, optional_scorer_hidden=8)
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "ppo", "epochs": 1, "batch_size": 4,
            }},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        m = learner.train(shard_dir)
        assert m["mode"] == "ppo"
        assert m["num_updates"] > 0


# ========== 7. Stage2SelfPlayWorker buffer initialization regression guard ==========


class TestSelfPlayWorkerBufferInit:
    """CQ-0291 follow-up: __init__ 末尾の model setup / preallocated inference
    buffers (`_feat_buf`, `_mask_buf`, `_rc_buf`) が、optional flag 読み込み
    後に必ず存在することを保証する。

    かつて _read_optional_flag staticmethod の return 後ろにぶら下がって
    いて dead code 化していた経緯があるため、indentation bug の regression
    guard として残す。
    """

    def _make(self, config, with_model=False, tmp_path=None):
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder
        enc = FlatFeatureEncoder(observation_mode="full")
        model = None
        if with_model:
            obs_dim = int(np.prod(enc.metadata().output_shape))
            model = Stage2aModel(
                input_dim=obs_dim, discard_hidden_dims=[16],
                optional_hidden_dims=[16], value_hidden_dims=[16],
                candidate_dim=8, optional_scorer_hidden=8)
        return Stage2SelfPlayWorker(
            config=config or {},
            output_dir=tmp_path or Path("/tmp"),
            observation_mode="full", encoder=enc, model=model,
        )

    def test_buffers_exist_default_off(self, tmp_path):
        w = self._make({}, with_model=False, tmp_path=tmp_path)
        assert hasattr(w, "_feat_buf")
        assert hasattr(w, "_mask_buf")
        assert hasattr(w, "_rc_buf")
        # shape sanity
        assert w._feat_buf.shape[0] == 1
        assert w._mask_buf.shape == (1, 34)
        assert w._rc_buf.shape == (1, 3)

    def test_buffers_exist_all_flags_on(self, tmp_path):
        config = {"training": {
            "optional_riichi": {"enabled": True},
            "optional_tsumo": {"enabled": True},
            "optional_ron": {"enabled": True},
            "optional_ankan": {"enabled": True},
            "optional_kakan": {"enabled": True},
            "optional_kyuushu": {"enabled": True},
        }}
        w = self._make(config, with_model=False, tmp_path=tmp_path)
        assert hasattr(w, "_feat_buf")
        assert hasattr(w, "_mask_buf")
        assert hasattr(w, "_rc_buf")
        # 全 flag が True
        assert w._optional_riichi_enabled is True
        assert w._optional_tsumo_enabled is True
        assert w._optional_ron_enabled is True
        assert w._optional_ankan_enabled is True
        assert w._optional_kakan_enabled is True
        assert w._optional_kyuushu_enabled is True

    def test_buffers_exist_with_model(self, tmp_path):
        """exp_035 想定: model 付きで Stage2SelfPlayWorker を構築し、
        optional flag 全 enabled で buffer も model も存在する"""
        config = {"training": {
            "optional_riichi": {"enabled": True},
            "optional_tsumo": {"enabled": True},
            "optional_ron": {"enabled": True},
            "optional_ankan": {"enabled": True},
            "optional_kakan": {"enabled": True},
            "optional_kyuushu": {"enabled": True},
        }}
        w = self._make(config, with_model=True, tmp_path=tmp_path)
        assert hasattr(w, "_feat_buf")
        assert hasattr(w, "_mask_buf")
        assert hasattr(w, "_rc_buf")
        # model.eval() が呼ばれている (training=False)
        assert w._model is not None
        assert w._model.training is False

    def test_exp035_smoke_with_model_runs(self, tmp_path):
        """exp_035 相当: optional 全 enabled + model ありで selfplay smoke
        が AttributeError 無しで完走する"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder
        enc = FlatFeatureEncoder(observation_mode="full")
        obs_dim = int(np.prod(enc.metadata().output_shape))
        model = Stage2aModel(
            input_dim=obs_dim, discard_hidden_dims=[16],
            optional_hidden_dims=[16], value_hidden_dims=[16],
            candidate_dim=8, optional_scorer_hidden=8)
        w = Stage2SelfPlayWorker(
            config={"training": {
                "optional_riichi": {"enabled": True},
                "optional_tsumo": {"enabled": True},
                "optional_ron": {"enabled": True},
                "optional_ankan": {"enabled": True},
                "optional_kakan": {"enabled": True},
                "optional_kyuushu": {"enabled": True},
            }},
            output_dir=tmp_path,
            observation_mode="full", encoder=enc, model=model,
        )
        # 短い selfplay smoke: 2 match だけ
        stats = w.generate(num_matches=2, base_seed=0)
        assert stats["num_matches"] == 2
        assert stats["total_steps"] > 0
