"""CQ-0291 (batch 2): TsumoWin / Ron optional unlock

確認:
- default off で既存自動 Tsumo / Ron 挙動を維持
- ``optional_tsumo_enabled=True`` で TsumoWin 合法時 ``TSUMO_OPTIONAL`` 発火
- TsumoWin candidate (idx=0) を選ぶと round が和了終了
- Skip candidate (idx=1) を選ぶと和了せず ``DISCARD`` decision に fall-through
- ``optional_ron_enabled=True`` で Ron 合法時 ``RON_OPTIONAL`` 発火
- Ron candidate (idx=0) を選ぶと round が和了終了
- Skip candidate (idx=1) を選ぶと見逃しで進行
- candidate_encoding ACTION_TYPE_MAP に TsumoWin (1→6) / Ron (2→7) が含まれる
- model embedding が 8 行
- 旧 4 行 / 6 行 checkpoint を ``load_stage2a_state_dict`` で読み戻せる
- shard roundtrip で ``decision_family ∈ {tsumo, ron}`` が保持される
- selfplay enabled で tsumo / ron sample が生成される
- learner imitation / PPO smoke で crash しない
- batch 1 (Riichi) tests が引き続き通る
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.smoke

from mahjong_rl import Action, ActionType
from mahjong_rl.env.stage2_env import Stage2Env, DecisionType
from mahjong_rl.env.response_candidate import (
    OPTIONAL_TSUMO_ACTION_TYPE, OPTIONAL_RON_ACTION_TYPE,
    OPTIONAL_SKIP_ACTION_TYPE,
    make_tsumo_optional_candidates, make_ron_optional_candidates,
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


class TestActionTypeMapBatch2:
    def test_tsumo_ron_codes(self):
        assert OPTIONAL_TSUMO_ACTION_TYPE == 1   # engine ActionType.TsumoWin
        assert OPTIONAL_RON_ACTION_TYPE == 2     # engine ActionType.Ron
        assert OPTIONAL_SKIP_ACTION_TYPE == 8    # engine ActionType.Skip
        assert ACTION_TYPE_MAP[1] == 6
        assert ACTION_TYPE_MAP[2] == 7

    def test_legacy_codes_unchanged(self):
        assert ACTION_TYPE_MAP[8] == 0  # Skip
        assert ACTION_TYPE_MAP[3] == 1  # Chi
        assert ACTION_TYPE_MAP[4] == 2  # Pon
        assert ACTION_TYPE_MAP[5] == 3  # Daiminkan
        assert ACTION_TYPE_MAP[100] == 4
        assert ACTION_TYPE_MAP[101] == 5

    def test_num_indices_at_least_8(self):
        assert NUM_ACTION_TYPE_INDICES >= 8


class TestEncoderDim8:
    def test_encoder_emb_8(self):
        enc = CandidateEncoder(candidate_dim=8)
        assert enc.action_type_emb.num_embeddings == NUM_ACTION_TYPE_INDICES
        assert enc.action_type_emb.num_embeddings >= 8


class TestLoadHelperLegacyEmb:
    """旧 4/6 行 checkpoint を 8 行 model に読み戻せる"""

    def _make_model(self):
        return Stage2aModel(
            input_dim=20, discard_hidden_dims=[16],
            optional_hidden_dims=[16], value_hidden_dims=[16],
            candidate_dim=8, optional_scorer_hidden=8)

    def test_legacy_4row_loads(self):
        m = self._make_model()
        sd = m.state_dict()
        old = torch.randn(4, 4)
        sd_legacy = dict(sd)
        sd_legacy["candidate_encoder.action_type_emb.weight"] = old
        result = load_stage2a_state_dict(m, sd_legacy)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        new_w = m.candidate_encoder.action_type_emb.weight.data
        assert torch.allclose(new_w[:4], old)

    def test_legacy_6row_loads(self):
        m = self._make_model()
        sd = m.state_dict()
        old = torch.randn(6, 4)
        sd_legacy = dict(sd)
        sd_legacy["candidate_encoder.action_type_emb.weight"] = old
        result = load_stage2a_state_dict(m, sd_legacy)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        new_w = m.candidate_encoder.action_type_emb.weight.data
        assert torch.allclose(new_w[:6], old)


# ========== 2. response_candidate factories ==========


class TestTsumoOptionalCandidates:
    def test_make_with_skip(self):
        tsumo = Action.make_tsumo_win(0)
        skip = Action.make_skip(0)
        cs = make_tsumo_optional_candidates(tsumo, skip, 0)
        assert len(cs) == 2
        assert cs[0].action_type == OPTIONAL_TSUMO_ACTION_TYPE
        assert cs[1].action_type == OPTIONAL_SKIP_ACTION_TYPE

    def test_make_without_skip(self):
        """SelfActionPhase で engine が Skip を提供しないケース"""
        tsumo = Action.make_tsumo_win(0)
        cs = make_tsumo_optional_candidates(tsumo, None, 0)
        assert len(cs) == 2
        assert cs[0].action_type == OPTIONAL_TSUMO_ACTION_TYPE
        assert cs[1].action_type == OPTIONAL_SKIP_ACTION_TYPE


class TestRonOptionalCandidates:
    def test_make(self):
        # Action.make_ron(actor, target)
        ron = Action.make_ron(0, 1)
        skip = Action.make_skip(0)
        cs = make_ron_optional_candidates(ron, skip, current_player=0)
        assert len(cs) == 2
        assert cs[0].action_type == OPTIONAL_RON_ACTION_TYPE
        assert cs[1].action_type == OPTIONAL_SKIP_ACTION_TYPE
        # rel_seat: target=1, current=0 → (1-0)%4 = 1
        assert cs[0].target_rel_seat == 1


# ========== 3. Stage2Env transitions ==========


class TestStage2EnvDefaultOff:
    """default off で TSUMO_OPTIONAL / RON_OPTIONAL は発火しない"""

    def test_no_tsumo_optional_default(self):
        env = Stage2Env(observation_mode="full",
                          optional_tsumo_enabled=False)
        for seed in range(50):
            env.reset(seed)
            for _ in range(300):
                if env.decision_type == DecisionType.DISCARD:
                    mask = env.get_legal_mask()
                    if mask.sum() == 0:
                        break
                    tt = int(np.argmax(mask))
                    _, _, term, _, _ = env.step_discard(tt)
                    assert env.decision_type != DecisionType.TSUMO_OPTIONAL
                    if term:
                        break
                elif env.decision_type == DecisionType.RESPONSE:
                    n = len(env.response_candidates)
                    _, _, term, _, _ = env.step_response(n - 1)
                    if term:
                        break
                else:
                    break

    def test_no_ron_optional_default(self):
        env = Stage2Env(observation_mode="full",
                          optional_ron_enabled=False)
        for seed in range(50):
            env.reset(seed)
            for _ in range(300):
                if env.decision_type == DecisionType.DISCARD:
                    mask = env.get_legal_mask()
                    if mask.sum() == 0:
                        break
                    tt = int(np.argmax(mask))
                    _, _, term, _, _ = env.step_discard(tt)
                    assert env.decision_type != DecisionType.RON_OPTIONAL
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
    """指定 decision_type に到達するまで env を進める"""
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
                n = len(env.response_candidates)
                _, _, term, _, _ = env.step_response(n - 1)
                if term:
                    break
            else:
                break
    return None


class TestTsumoOptionalEnabled:
    def test_tsumo_fires(self):
        env = Stage2Env(observation_mode="full",
                          optional_tsumo_enabled=True)
        seed = _drive_until(env, DecisionType.TSUMO_OPTIONAL)
        assert seed is not None
        cs = env.response_candidates
        assert len(cs) == 2
        assert cs[0].action_type == OPTIONAL_TSUMO_ACTION_TYPE
        assert cs[1].action_type == OPTIONAL_SKIP_ACTION_TYPE

    def test_tsumo_choice_wins(self):
        env = Stage2Env(observation_mode="full",
                          optional_tsumo_enabled=True)
        seed = _drive_until(env, DecisionType.TSUMO_OPTIONAL)
        assert seed is not None
        # TsumoWin 選択
        env.step_response(0)
        # round が終了したかどうかは env._last_round_outcome で確認できる
        # 少なくとも TSUMO_OPTIONAL のままにはならない
        assert env.decision_type != DecisionType.TSUMO_OPTIONAL

    def test_skip_falls_through_to_discard(self):
        """Skip 選択時、engine step は走らず DISCARD decision に fall-through"""
        env = Stage2Env(observation_mode="full",
                          optional_tsumo_enabled=True)
        seed = _drive_until(env, DecisionType.TSUMO_OPTIONAL)
        assert seed is not None
        cp_before = env.env_state.round_state.current_player
        env.step_response(1)  # Skip
        # 同じ player の DISCARD decision に降りる
        assert env.decision_type == DecisionType.DISCARD
        assert env.env_state.round_state.current_player == cp_before
        # この時点で同じ TsumoWin が再提示されないこと
        # (= _tsumo_skipped_this_turn フラグが効く)
        # 通常の discard を進めて確認
        mask = env.get_legal_mask()
        assert mask.sum() > 0
        tt = int(np.argmax(mask))
        env.step_discard(tt)
        # discard 後はフラグが clear される (新しい turn / phase)
        # CQ-0291 batch 3: skipped 状態は集合化されたので空であることを確認
        assert "tsumo" not in env._optional_skipped_this_turn


class TestRonOptionalEnabled:
    def test_ron_fires(self):
        env = Stage2Env(observation_mode="full",
                          optional_ron_enabled=True)
        seed = _drive_until(env, DecisionType.RON_OPTIONAL)
        assert seed is not None
        cs = env.response_candidates
        assert len(cs) == 2
        assert cs[0].action_type == OPTIONAL_RON_ACTION_TYPE
        assert cs[1].action_type == OPTIONAL_SKIP_ACTION_TYPE

    def test_ron_choice_wins(self):
        env = Stage2Env(observation_mode="full",
                          optional_ron_enabled=True)
        seed = _drive_until(env, DecisionType.RON_OPTIONAL)
        assert seed is not None
        env.step_response(0)  # Ron
        assert env.decision_type != DecisionType.RON_OPTIONAL

    def test_ron_skip_does_not_terminate_round_immediately(self):
        env = Stage2Env(observation_mode="full",
                          optional_ron_enabled=True)
        seed = _drive_until(env, DecisionType.RON_OPTIONAL)
        assert seed is not None
        # Skip
        env.step_response(1)
        # 見逃し → round が即終了することは原則ない
        # decision_type が RON_OPTIONAL のままにならない
        assert env.decision_type != DecisionType.RON_OPTIONAL


# ========== 4. shard roundtrip ==========


class TestShardRoundtripBatch2:
    def test_decision_family_tsumo(self, tmp_path):
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        s = DecisionSample(
            decision_type="call",
            decision_family="tsumo",
            observation=np.zeros(10, dtype=np.float32),
            reward=0.0, log_prob=-0.5, value=0.0,
            terminated=False, round_over=False,
            selected_candidate_index=0, candidate_count=2,
            candidates=[
                CandidateRecord(action_type=OPTIONAL_TSUMO_ACTION_TYPE),
                CandidateRecord(action_type=OPTIONAL_SKIP_ACTION_TYPE),
            ],
            player_id=0, episode_id="ep0", round_id=0, step_id=0,
            actor_type="policy",
            experiment_id="t", run_id="r", worker_id="w",
        )
        writer.add(s)
        writer.close()
        ss = DecisionShardReader(tmp_path).read_all()
        assert len(ss) == 1
        assert ss[0].decision_family == "tsumo"
        assert ss[0].candidates[0].action_type == OPTIONAL_TSUMO_ACTION_TYPE
        assert ss[0].candidates[1].action_type == OPTIONAL_SKIP_ACTION_TYPE

    def test_decision_family_ron(self, tmp_path):
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        s = DecisionSample(
            decision_type="call",
            decision_family="ron",
            observation=np.zeros(10, dtype=np.float32),
            reward=0.0, log_prob=-0.5, value=0.0,
            terminated=False, round_over=False,
            selected_candidate_index=0, candidate_count=2,
            candidates=[
                CandidateRecord(action_type=OPTIONAL_RON_ACTION_TYPE,
                                 tile_type=4, target_rel_seat=1),
                CandidateRecord(action_type=OPTIONAL_SKIP_ACTION_TYPE),
            ],
            player_id=0, episode_id="ep0", round_id=0, step_id=0,
            actor_type="policy",
            experiment_id="t", run_id="r", worker_id="w",
        )
        writer.add(s)
        writer.close()
        ss = DecisionShardReader(tmp_path).read_all()
        assert ss[0].decision_family == "ron"
        assert ss[0].candidates[0].action_type == OPTIONAL_RON_ACTION_TYPE
        assert ss[0].candidates[0].target_rel_seat == 1


# ========== 5. selfplay worker integration ==========


class TestSelfplayWorkerBatch2:
    def test_default_off_no_optional_win_samples(self, tmp_path):
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder
        enc = FlatFeatureEncoder(observation_mode="full")
        out = tmp_path / "off"
        w = Stage2SelfPlayWorker(
            config={}, output_dir=out,
            observation_mode="full", encoder=enc,
        )
        assert w._optional_tsumo_enabled is False
        assert w._optional_ron_enabled is False
        w.generate(num_matches=3, base_seed=0)
        ss = DecisionShardReader(out).read_all()
        for fam in ("tsumo", "ron"):
            n = sum(1 for s in ss if s.decision_family == fam)
            assert n == 0, f"default off で {fam} sample が生成された ({n})"

    def test_enabled_generates_tsumo_and_ron_samples(self, tmp_path):
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder
        enc = FlatFeatureEncoder(observation_mode="full")
        out = tmp_path / "on"
        w = Stage2SelfPlayWorker(
            config={"training": {
                "optional_tsumo": {"enabled": True},
                "optional_ron": {"enabled": True},
            }},
            output_dir=out,
            observation_mode="full", encoder=enc,
        )
        assert w._optional_tsumo_enabled is True
        assert w._optional_ron_enabled is True
        w.generate(num_matches=20, base_seed=0)
        ss = DecisionShardReader(out).read_all()
        n_tsumo = sum(1 for s in ss if s.decision_family == "tsumo")
        n_ron = sum(1 for s in ss if s.decision_family == "ron")
        # 20 match で少なくとも 1 つは Tsumo or Ron が出るはず
        assert (n_tsumo + n_ron) > 0, (
            f"enabled で tsumo/ron sample が 0 件: tsumo={n_tsumo}, "
            f"ron={n_ron}")
        # baseline = auto-win (idx 0) + teacher_top1 = 0
        for s in ss:
            if s.decision_family in ("tsumo", "ron"):
                assert s.candidate_count == 2
                assert s.candidates[1].action_type == OPTIONAL_SKIP_ACTION_TYPE
                # imitation mode (model=None) は actor_type=baseline で
                # selected=0 (Win)
                assert s.selected_candidate_index == 0
                assert s.teacher_top1_index == 0


# ========== 6. learner smoke ==========


class TestLearnerSmokeBatch2:
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
        # tsumo optional samples
        for i in range(3):
            writer.add(DecisionSample(
                decision_type="call",
                decision_family="tsumo",
                observation=np.zeros(10, dtype=np.float32),
                reward=0.0, log_prob=-0.5, value=0.0,
                terminated=False, round_over=False,
                selected_candidate_index=0, candidate_count=2,
                candidates=[
                    CandidateRecord(action_type=OPTIONAL_TSUMO_ACTION_TYPE),
                    CandidateRecord(action_type=OPTIONAL_SKIP_ACTION_TYPE),
                ],
                response_context=np.zeros(3, dtype=np.float32),
                player_id=0, episode_id="ep0", round_id=0, step_id=4 + i,
                actor_type="policy", teacher_top1_index=0,
                teacher_source="auto_tsumo",
                experiment_id="t", run_id="r", worker_id="w",
            ))
        # ron optional samples
        for i in range(3):
            writer.add(DecisionSample(
                decision_type="call",
                decision_family="ron",
                observation=np.zeros(10, dtype=np.float32),
                reward=0.0, log_prob=-0.5, value=0.0,
                terminated=(i == 2), round_over=False,
                selected_candidate_index=0, candidate_count=2,
                candidates=[
                    CandidateRecord(action_type=OPTIONAL_RON_ACTION_TYPE,
                                     tile_type=4, target_rel_seat=1),
                    CandidateRecord(action_type=OPTIONAL_SKIP_ACTION_TYPE),
                ],
                response_context=np.zeros(3, dtype=np.float32),
                player_id=0, episode_id="ep0", round_id=0, step_id=7 + i,
                actor_type="policy", teacher_top1_index=0,
                teacher_source="auto_ron",
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
