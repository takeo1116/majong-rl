"""CQ-0291 (batch 1): Stage02b Riichi/NoRiichi optional unlock

確認:
- default off (`optional_riichi_enabled=False`) で Stage2Env は既存の自動
  Riichi 挙動 (RIICHI_OPTIONAL を発生させない) を維持する
- enabled で riichi 可能な tile_type を選んだとき RIICHI_OPTIONAL が発火し、
  candidates が `[NoRiichi(idx 0), Riichi(idx 1)]` で並ぶ
- Riichi candidate を選ぶと player.is_riichi が True になり、立直棒 1000 が
  支払われる
- NoRiichi candidate を選ぶと player.is_riichi が False のままになる
- candidate_encoding ACTION_TYPE_MAP に NoRiichi (100→4) / Riichi (101→5)
  が含まれる
- model `CandidateEncoder.action_type_emb` が 6 行 (Skip..Riichi)
- 旧 4 行 checkpoint を `load_stage2a_state_dict` で読み戻すと、上 4 行
  だけ復元され、行 4/5 (NoRiichi/Riichi) は init 値のまま
- `DecisionSample.decision_family` の shard 書き出し / 読み戻しが round-trip
  する
- selfplay worker (default off) で既存挙動を維持し、enabled で riichi
  optional sample が生成される
- learner smoke が riichi optional samples を含む shard で crash しない
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.smoke

from mahjong_rl import Action, ActionType, NUM_TILE_TYPES
from mahjong_rl.env.stage2_env import Stage2Env, DecisionType
from mahjong_rl.env.response_candidate import (
    OPTIONAL_NORIICHI_ACTION_TYPE, OPTIONAL_RIICHI_ACTION_TYPE,
    make_riichi_optional_candidates,
)
from mahjong_rl.candidate_encoding import (
    ACTION_TYPE_MAP, NUM_ACTION_TYPE_INDICES,
    OPTIONAL_NORIICHI, OPTIONAL_RIICHI,
)
from mahjong_rl.call_shard import (
    DecisionSample, CandidateRecord,
    DecisionShardWriter, DecisionShardReader,
)
from mahjong_rl.models.stage2a_model import (
    Stage2aModel, load_stage2a_state_dict, CandidateEncoder,
)


# ========== 1. ACTION_TYPE_MAP / Encoder dim ==========


class TestActionTypeMapExtended:
    def test_synthetic_codes_present(self):
        assert OPTIONAL_NORIICHI == 100
        assert OPTIONAL_RIICHI == 101
        assert ACTION_TYPE_MAP[OPTIONAL_NORIICHI] == 4
        assert ACTION_TYPE_MAP[OPTIONAL_RIICHI] == 5

    def test_legacy_codes_unchanged(self):
        assert ACTION_TYPE_MAP[8] == 0  # Skip
        assert ACTION_TYPE_MAP[3] == 1  # Chi
        assert ACTION_TYPE_MAP[4] == 2  # Pon
        assert ACTION_TYPE_MAP[5] == 3  # Daiminkan

    def test_num_indices(self):
        # CQ-0291 batch 1 で 6 行、batch 2 で 8 行に拡張済み
        assert NUM_ACTION_TYPE_INDICES >= 6


class TestCandidateEncoderDim:
    def test_encoder_emb_size(self):
        enc = CandidateEncoder(candidate_dim=8)
        # batch 1 で 6 行、batch 2 で 8 行に拡張
        assert enc.action_type_emb.num_embeddings == NUM_ACTION_TYPE_INDICES
        assert enc.action_type_emb.embedding_dim == 4


class TestLoadHelperLegacyEmb:
    """旧 4 行 checkpoint を 6 行 model に読み戻せる"""

    def test_legacy_4row_checkpoint_loads(self):
        # 6 行 model
        model = Stage2aModel(
            input_dim=20, discard_hidden_dims=[16],
            optional_hidden_dims=[16], value_hidden_dims=[16],
            candidate_dim=8, optional_scorer_hidden=8)
        # 旧 4 行 emb の state を作る
        sd = model.state_dict()
        old_emb = torch.randn(4, 4)
        sd_legacy = dict(sd)
        sd_legacy["candidate_encoder.action_type_emb.weight"] = old_emb
        # crash しないこと
        result = load_stage2a_state_dict(model, sd_legacy)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        # 上 4 行が old_emb と一致
        new_emb = model.candidate_encoder.action_type_emb.weight.data
        assert torch.allclose(new_emb[:4], old_emb)
        # 行 4/5 は init 値のまま (= sd の元の値)
        orig_emb = sd["candidate_encoder.action_type_emb.weight"]
        assert torch.allclose(new_emb[4:], orig_emb[4:])


# ========== 2. response_candidate factory ==========


class TestRiichiOptionalCandidates:
    def test_make_candidates(self):
        # Action.make_discard(player_id, tile, riichi)
        non_riichi = Action.make_discard(0, 17, False)  # 通常5m, non-riichi
        riichi = Action.make_discard(0, 17, True)        # 通常5m, riichi
        cands = make_riichi_optional_candidates(non_riichi, riichi, 0)
        assert len(cands) == 2
        # idx 0 = NoRiichi, idx 1 = Riichi
        assert cands[0].action_type == OPTIONAL_NORIICHI_ACTION_TYPE
        assert cands[1].action_type == OPTIONAL_RIICHI_ACTION_TYPE
        # tile_type 一致
        assert cands[0].tile_type == 17 // 4  # = 4 (5m)
        assert cands[1].tile_type == 17 // 4
        # Action は engine の Discard
        assert cands[0].action.tile == 17
        assert cands[0].action.riichi is False
        assert cands[1].action.tile == 17
        assert cands[1].action.riichi is True


# ========== 3. Stage2Env transitions ==========


class TestStage2EnvDefaultOff:
    """default off で既存の自動 Riichi 挙動を維持する"""

    def test_no_riichi_optional_emitted(self):
        env = Stage2Env(observation_mode="full",
                          optional_riichi_enabled=False)
        # 多くの seed で discard を回しても RIICHI_OPTIONAL が発生しない
        for seed in range(50):
            env.reset(seed)
            for _ in range(200):
                if env.decision_type == DecisionType.DISCARD:
                    mask = env.get_legal_mask()
                    if mask.sum() == 0:
                        break
                    tt = int(np.argmax(mask))
                    _, _, term, _, _ = env.step_discard(tt)
                    # default off では RIICHI_OPTIONAL に遷移しない
                    assert env.decision_type != DecisionType.RIICHI_OPTIONAL
                    if term:
                        break
                elif env.decision_type == DecisionType.RESPONSE:
                    n = len(env.response_candidates)
                    _, _, term, _, _ = env.step_response(n - 1)  # Skip
                    if term:
                        break
                else:
                    break


class TestStage2EnvEnabled:
    """enabled で riichi/non-riichi 同時に合法な場面で RIICHI_OPTIONAL 発火"""

    def _drive_until_riichi_optional(self, env, max_seeds=500, max_steps=300):
        """RIICHI_OPTIONAL に到達するまで env を進める

        CQ-0292 (batch 2) 対応: ``optional_riichi_enabled=True`` のとき
        legal mask には riichi/non-riichi 両方の tile_type が含まれる。
        単純な ``np.argmax`` は非 riichi tile を先に選び、結果として
        RIICHI_OPTIONAL に到達するのが round 終盤になりがちで test 不安定。
        snapshot を見て riichi 打牌が legal な tile_type を優先する。
        """
        for seed in range(max_seeds):
            env.reset(seed)
            for _ in range(max_steps):
                if env.decision_type == DecisionType.DISCARD:
                    mask, snap = env.get_legal_discard_snapshot()
                    if mask.sum() == 0:
                        break
                    riichi_tile_types = sorted({
                        a.tile // 4 for a in snap if a.riichi
                    })
                    if riichi_tile_types:
                        tt = riichi_tile_types[0]
                    else:
                        tt = int(np.argmax(mask))
                    obs, _, term, _, _ = env.step_discard_with_snapshot(
                        tt, snap)
                    if env.decision_type == DecisionType.RIICHI_OPTIONAL:
                        return seed, obs
                    if term:
                        break
                elif env.decision_type == DecisionType.RESPONSE:
                    n = len(env.response_candidates)
                    _, _, term, _, _ = env.step_response(n - 1)
                    if term:
                        break
                elif env.decision_type == DecisionType.RIICHI_OPTIONAL:
                    return seed, env._make_observation()
                else:
                    break
        return None, None

    def test_riichi_optional_fires(self):
        env = Stage2Env(observation_mode="full",
                          optional_riichi_enabled=True)
        seed, _ = self._drive_until_riichi_optional(env)
        assert seed is not None, "RIICHI_OPTIONAL に到達しなかった"
        assert env.decision_type == DecisionType.RIICHI_OPTIONAL
        cands = env.response_candidates
        assert len(cands) == 2
        assert cands[0].action_type == OPTIONAL_NORIICHI_ACTION_TYPE
        assert cands[1].action_type == OPTIONAL_RIICHI_ACTION_TYPE
        assert cands[0].action.riichi is False
        assert cands[1].action.riichi is True
        # 同 tile_type
        assert cands[0].tile_type == cands[1].tile_type
        # CQ-0290: 通常牌優先で同 tile_id を使うことが多い (両方 normal id)
        # ただし「赤しか無い」ケースは発生しうるので tile_id 一致までは
        # 強制しない
        assert (cands[0].action.tile // 4) == cands[1].action.tile // 4

    def test_riichi_choice_sets_is_riichi_true(self):
        """Riichi candidate (idx=1) を選ぶと player.is_riichi が True"""
        env = Stage2Env(observation_mode="full",
                          optional_riichi_enabled=True)
        seed, _ = self._drive_until_riichi_optional(env)
        assert seed is not None
        # 立直する player (= 直前 discard を行う player)
        cp = env.env_state.round_state.current_player
        env.step_response(1)  # Riichi
        # discard 後 phase が advance する。is_riichi は self_action 後に評価
        # → riichi 宣言した player の is_riichi は True
        assert env.env_state.round_state.players[cp].is_riichi is True

    def test_no_riichi_choice_keeps_is_riichi_false(self):
        """NoRiichi candidate (idx=0) を選ぶと player.is_riichi は False"""
        env = Stage2Env(observation_mode="full",
                          optional_riichi_enabled=True)
        seed, _ = self._drive_until_riichi_optional(env)
        assert seed is not None
        cp = env.env_state.round_state.current_player
        env.step_response(0)  # NoRiichi
        assert env.env_state.round_state.players[cp].is_riichi is False

    def test_decision_type_back_to_discard_or_response(self):
        """RIICHI_OPTIONAL → step_response 後は DISCARD / RESPONSE / 終局へ
        遷移する (RIICHI_OPTIONAL のままにはならない)"""
        env = Stage2Env(observation_mode="full",
                          optional_riichi_enabled=True)
        seed, _ = self._drive_until_riichi_optional(env)
        assert seed is not None
        env.step_response(1)
        assert env.decision_type != DecisionType.RIICHI_OPTIONAL


# ========== 4. shard roundtrip ==========


class TestShardRoundtripDecisionFamily:
    def test_decision_family_default_response(self, tmp_path):
        """default 値 'response' で writer/reader が roundtrip する"""
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        s = DecisionSample(
            decision_type="call",
            observation=np.zeros(10, dtype=np.float32),
            reward=0.0, log_prob=-0.5, value=0.0,
            terminated=False, round_over=False,
            selected_candidate_index=0, candidate_count=1,
            candidates=[CandidateRecord(action_type=8)],
            player_id=0, episode_id="ep0", round_id=0, step_id=0,
            actor_type="policy",
            experiment_id="t", run_id="r", worker_id="w",
        )
        writer.add(s)
        writer.close()
        samples = DecisionShardReader(tmp_path).read_all()
        assert len(samples) == 1
        assert samples[0].decision_family == "response"

    def test_decision_family_riichi_roundtrip(self, tmp_path):
        """decision_family='riichi' を書いて読み戻す"""
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        s = DecisionSample(
            decision_type="call",
            decision_family="riichi",
            observation=np.zeros(10, dtype=np.float32),
            reward=0.0, log_prob=-0.5, value=0.0,
            terminated=False, round_over=False,
            selected_candidate_index=1, candidate_count=2,
            candidates=[
                CandidateRecord(action_type=OPTIONAL_NORIICHI),
                CandidateRecord(action_type=OPTIONAL_RIICHI),
            ],
            player_id=0, episode_id="ep0", round_id=0, step_id=0,
            actor_type="policy",
            experiment_id="t", run_id="r", worker_id="w",
        )
        writer.add(s)
        writer.close()
        samples = DecisionShardReader(tmp_path).read_all()
        assert len(samples) == 1
        s2 = samples[0]
        assert s2.decision_family == "riichi"
        assert s2.candidates[0].action_type == OPTIONAL_NORIICHI
        assert s2.candidates[1].action_type == OPTIONAL_RIICHI
        assert s2.selected_candidate_index == 1


# ========== 5. selfplay worker integration ==========


class TestSelfplayWorkerIntegration:
    """selfplay worker が default off / enabled で挙動を切り替える"""

    def test_default_off_no_riichi_samples(self, tmp_path):
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder
        enc = FlatFeatureEncoder(observation_mode="full")
        out = tmp_path / "off"
        w = Stage2SelfPlayWorker(
            config={}, output_dir=out,
            observation_mode="full", encoder=enc,
        )
        assert w._optional_riichi_enabled is False
        w.generate(num_matches=3, base_seed=0)
        samples = DecisionShardReader(out).read_all()
        assert len(samples) > 0
        # default off では riichi family が生成されない
        riichi = [s for s in samples if s.decision_family == "riichi"]
        assert len(riichi) == 0, (
            f"default off で riichi sample が生成された ({len(riichi)} 件)")

    def test_enabled_generates_riichi_samples(self, tmp_path):
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder
        enc = FlatFeatureEncoder(observation_mode="full")
        out = tmp_path / "on"
        w = Stage2SelfPlayWorker(
            config={"training": {"optional_riichi": {"enabled": True}}},
            output_dir=out,
            observation_mode="full", encoder=enc,
        )
        assert w._optional_riichi_enabled is True
        w.generate(num_matches=10, base_seed=0)
        samples = DecisionShardReader(out).read_all()
        riichi = [s for s in samples if s.decision_family == "riichi"]
        assert len(riichi) > 0, (
            "enabled で riichi sample が生成されなかった (10 match)")
        # baseline teacher = Riichi(=1) を保つ
        for s in riichi:
            assert s.candidate_count == 2
            assert s.candidates[0].action_type == OPTIONAL_NORIICHI
            assert s.candidates[1].action_type == OPTIONAL_RIICHI
            assert s.teacher_top1_index == 1
            # imitation mode (model=None) では actor_type=baseline で
            # selected_candidate_index=1 (= 自動 Riichi 互換)
            assert s.selected_candidate_index == 1


# ========== 6. learner smoke ==========


class TestLearnerSmokeWithRiichiSamples:
    """riichi optional samples を含む shard で learner が crash しない"""

    def test_imitation_smoke(self, tmp_path):
        from mahjong_rl.stage2a_learner import Stage2aLearner

        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        # 通常 discard sample
        for i in range(4):
            writer.add(DecisionSample(
                decision_type="discard",
                observation=np.zeros(10, dtype=np.float32),
                legal_mask=np.ones(34, dtype=np.float32),
                action=0, reward=0.0, log_prob=-0.5, value=0.0,
                terminated=(i == 3), round_over=False,
                player_id=0, episode_id="ep0", round_id=0, step_id=i,
                actor_type="policy", teacher_top1_index=0,
                teacher_source="rule_based",
                experiment_id="t", run_id="r", worker_id="w",
            ))
        # riichi optional samples
        for i in range(4):
            writer.add(DecisionSample(
                decision_type="call",
                decision_family="riichi",
                observation=np.zeros(10, dtype=np.float32),
                reward=0.0, log_prob=-0.5, value=0.0,
                terminated=(i == 3), round_over=False,
                selected_candidate_index=1, candidate_count=2,
                candidates=[
                    CandidateRecord(action_type=OPTIONAL_NORIICHI,
                                     tile_type=4),
                    CandidateRecord(action_type=OPTIONAL_RIICHI,
                                     tile_type=4),
                ],
                response_context=np.zeros(3, dtype=np.float32),
                player_id=0, episode_id="ep0", round_id=0, step_id=4 + i,
                actor_type="policy", teacher_top1_index=1,
                teacher_source="auto_riichi",
                experiment_id="t", run_id="r", worker_id="w",
            ))
        writer.close()

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
        # crash しないこと
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "imitation"
        assert metrics["num_updates"] > 0

    def test_ppo_smoke(self, tmp_path):
        from mahjong_rl.stage2a_learner import Stage2aLearner

        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        for i in range(8):
            writer.add(DecisionSample(
                decision_type="discard",
                observation=np.zeros(10, dtype=np.float32),
                legal_mask=np.ones(34, dtype=np.float32),
                action=0, reward=0.0, log_prob=-0.5, value=0.0,
                terminated=(i == 7), round_over=False,
                player_id=0, episode_id="ep0", round_id=0, step_id=i,
                actor_type="policy",
                experiment_id="t", run_id="r", worker_id="w",
            ))
        for i in range(4):
            writer.add(DecisionSample(
                decision_type="call",
                decision_family="riichi",
                observation=np.zeros(10, dtype=np.float32),
                reward=0.0, log_prob=-0.5, value=0.0,
                terminated=(i == 3), round_over=False,
                selected_candidate_index=1, candidate_count=2,
                candidates=[
                    CandidateRecord(action_type=OPTIONAL_NORIICHI,
                                     tile_type=4),
                    CandidateRecord(action_type=OPTIONAL_RIICHI,
                                     tile_type=4),
                ],
                response_context=np.zeros(3, dtype=np.float32),
                player_id=0, episode_id="ep0", round_id=0, step_id=8 + i,
                actor_type="policy",
                experiment_id="t", run_id="r", worker_id="w",
            ))
        writer.close()

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
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        assert metrics["num_updates"] > 0
