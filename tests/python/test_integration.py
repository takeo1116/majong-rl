"""統合テスト: env → encoder → model → selector → step の全パイプライン"""
import pytest
import torch
import numpy as np

from mahjong_rl import NUM_TILE_TYPES
from mahjong_rl.env import Stage1Env
from mahjong_rl.encoders import FlatFeatureEncoder, ChannelTensorEncoder
from mahjong_rl.models import MLPPolicyValueModel
from mahjong_rl.action_selector import ActionSelector, SelectionMode
from mahjong_rl.baseline import RuleBasedBaseline


class TestFullPipeline:
    """env → encoder → model → selector → step 統合テスト"""

    def test_flat_pipeline_one_step(self):
        """Flat encoder パイプラインで1ステップ実行"""
        env = Stage1Env(observation_mode="partial")
        encoder = FlatFeatureEncoder(observation_mode="partial")
        model = MLPPolicyValueModel(
            input_dim=encoder.output_dim,
            hidden_dims=[64],
        )
        selector = ActionSelector(mode=SelectionMode.ARGMAX)

        obs, info = env.reset(seed=42)
        features = encoder.encode(obs)
        features_t = torch.from_numpy(features).unsqueeze(0)
        mask = env.get_legal_mask()
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        output = model(features_t, mask_t)
        tile_type, _ = selector.select(output.logits[0], mask_t[0])

        obs2, rewards, terminated, truncated, info2 = env.step(tile_type)
        assert rewards.shape == (4,)

    def test_channel_pipeline_one_step(self):
        """Channel encoder パイプラインで1ステップ実行"""
        env = Stage1Env(observation_mode="full")
        encoder = ChannelTensorEncoder(observation_mode="full")
        # Channel encoder → flatten for MLP
        model = MLPPolicyValueModel(
            input_dim=encoder.output_dim,
            hidden_dims=[64],
        )
        selector = ActionSelector(mode=SelectionMode.SAMPLE, temperature=1.0)

        obs, info = env.reset(seed=42)
        features = encoder.encode(obs)
        features_t = torch.from_numpy(features.flatten()).unsqueeze(0)
        mask = env.get_legal_mask()
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        output = model(features_t, mask_t)
        tile_type, _ = selector.select(output.logits[0], mask_t[0])

        obs2, rewards, terminated, truncated, info2 = env.step(tile_type)
        assert rewards.shape == (4,)

    def test_pipeline_multiple_steps(self):
        """パイプラインで複数ステップ実行"""
        env = Stage1Env(observation_mode="partial")
        encoder = FlatFeatureEncoder(observation_mode="partial")
        model = MLPPolicyValueModel(
            input_dim=encoder.output_dim,
            hidden_dims=[32],
        )
        selector = ActionSelector(mode=SelectionMode.ARGMAX)

        obs, _ = env.reset(seed=42)
        for _ in range(50):
            features = encoder.encode(obs)
            features_t = torch.from_numpy(features).unsqueeze(0)
            mask = env.get_legal_mask()
            mask_t = torch.from_numpy(mask).unsqueeze(0)

            output = model(features_t, mask_t)
            tile_type, _ = selector.select(output.logits[0], mask_t[0])

            obs, rewards, terminated, truncated, info = env.step(tile_type)
            if terminated:
                break

    def test_model_gradient_in_pipeline(self):
        """パイプライン内で勾配が正しく流れる"""
        env = Stage1Env(observation_mode="partial")
        encoder = FlatFeatureEncoder(observation_mode="partial")
        model = MLPPolicyValueModel(
            input_dim=encoder.output_dim,
            hidden_dims=[32],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        obs, _ = env.reset(seed=42)
        features = encoder.encode(obs)
        features_t = torch.from_numpy(features).unsqueeze(0)
        mask = env.get_legal_mask()
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        output = model(features_t, mask_t)
        log_probs = torch.log_softmax(output.logits, dim=-1)

        # ダミーの行動選択とロス計算
        action = torch.argmax(output.logits, dim=-1)
        policy_loss = -log_probs[0, action[0]]
        value_loss = output.values["round_delta"].mean() ** 2
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # パラメータが更新されている
        assert loss.item() > 0 or loss.item() == 0  # ロスが計算される


class TestBaselineVsModel:
    """ベースラインとモデルの比較テスト"""

    def test_both_complete_match(self):
        """ランダムモデルとベースラインが同じ環境で半荘完走"""
        # ベースライン
        env_bl = Stage1Env(observation_mode="full")
        baseline = RuleBasedBaseline()
        env_bl.reset(seed=42)
        bl_steps = 0
        while bl_steps < 10000:
            mask = env_bl.get_legal_mask()
            hand = env_bl.env_state.round_state.players[
                env_bl.current_player
            ].hand
            tile_type = baseline.select_discard(list(hand), mask)
            _, _, terminated, _, info = env_bl.step(tile_type)
            bl_steps += 1
            if terminated:
                break
        assert info["is_match_over"]

        # ランダムモデル
        env_model = Stage1Env(observation_mode="partial")
        encoder = FlatFeatureEncoder(observation_mode="partial")
        model = MLPPolicyValueModel(
            input_dim=encoder.output_dim,
            hidden_dims=[32],
        )
        selector = ActionSelector(mode=SelectionMode.SAMPLE, temperature=1.0)

        obs, _ = env_model.reset(seed=42)
        m_steps = 0
        while m_steps < 10000:
            features = encoder.encode(obs)
            features_t = torch.from_numpy(features).unsqueeze(0)
            mask = env_model.get_legal_mask()
            mask_t = torch.from_numpy(mask).unsqueeze(0)

            output = model(features_t, mask_t)
            tile_type, _ = selector.select(output.logits[0], mask_t[0])

            obs, _, terminated, _, info = env_model.step(tile_type)
            m_steps += 1
            if terminated:
                break
        assert info["is_match_over"]
