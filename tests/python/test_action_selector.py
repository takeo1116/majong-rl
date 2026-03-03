"""CQ-0032: ActionSelector + Legal Mask テスト"""
import pytest
import numpy as np
import torch
from mahjong_rl.legal_mask import make_discard_mask, make_discard_mask_from_legal_actions
from mahjong_rl.action_selector import ActionSelector, SelectionMode
from mahjong_rl import (
    GameEngine, EnvironmentState, Action, ActionType, NUM_TILE_TYPES,
)


class TestLegalMask:
    """legal mask 生成テスト"""

    def test_mask_from_hand(self):
        # TileId 0,1,2,3 = 1m (type 0), TileId 4,5 = 2m (type 1)
        hand = [0, 1, 4, 5, 36, 72, 108]  # 1m*2, 2m*2, 1p, 1s, 東
        mask = make_discard_mask(hand)
        assert mask.shape == (NUM_TILE_TYPES,)
        assert mask.dtype == np.float32
        assert mask[0] == 1.0   # 1m
        assert mask[1] == 1.0   # 2m
        assert mask[9] == 1.0   # 1p
        assert mask[18] == 1.0  # 1s
        assert mask[27] == 1.0  # 東
        assert mask[2] == 0.0   # 3m なし
        assert mask[33] == 0.0  # 中 なし

    def test_mask_from_legal_actions(self, engine, initialized_env):
        actions = engine.get_legal_actions(initialized_env)
        mask = make_discard_mask_from_legal_actions(actions)
        assert mask.shape == (NUM_TILE_TYPES,)
        assert mask.sum() > 0

        # 手牌からのマスクと一致確認
        hand = initialized_env.round_state.players[
            initialized_env.round_state.current_player
        ].hand
        hand_mask = make_discard_mask(list(hand))
        # legal actions マスクは hand マスクのサブセット
        assert np.all(mask <= hand_mask + 1e-6)

    def test_empty_hand(self):
        mask = make_discard_mask([])
        assert mask.sum() == 0.0


class TestActionSelector:
    """ActionSelector テスト"""

    def test_argmax_selects_highest_legal(self):
        selector = ActionSelector(mode=SelectionMode.ARGMAX)
        logits = torch.zeros(NUM_TILE_TYPES)
        logits[5] = 10.0  # 6m に高いロジット
        legal_mask = torch.zeros(NUM_TILE_TYPES)
        legal_mask[3] = 1.0
        legal_mask[5] = 1.0
        legal_mask[10] = 1.0

        action, log_prob = selector.select(logits, legal_mask)
        assert action == 5

    def test_argmax_respects_mask(self):
        selector = ActionSelector(mode=SelectionMode.ARGMAX)
        logits = torch.zeros(NUM_TILE_TYPES)
        logits[5] = 100.0  # 6m に非常に高いロジットだが非合法
        legal_mask = torch.zeros(NUM_TILE_TYPES)
        legal_mask[3] = 1.0  # 4m のみ合法

        action, _ = selector.select(logits, legal_mask)
        assert action == 3  # 非合法の 5 ではなく合法の 3 を選ぶ

    def test_sampling_never_selects_illegal(self):
        selector = ActionSelector(mode=SelectionMode.SAMPLE)
        logits = torch.randn(NUM_TILE_TYPES)
        legal_mask = torch.zeros(NUM_TILE_TYPES)
        legal_mask[0] = 1.0
        legal_mask[10] = 1.0
        legal_mask[20] = 1.0

        for _ in range(1000):
            action, _ = selector.select(logits, legal_mask)
            assert action in [0, 10, 20], f"非合法手 {action} が選択された"

    def test_temperature_affects_distribution(self):
        """低温度ではより集中した分布になる"""
        logits = torch.zeros(NUM_TILE_TYPES)
        logits[0] = 2.0
        logits[1] = 1.0
        legal_mask = torch.ones(NUM_TILE_TYPES)

        selector_high = ActionSelector(mode=SelectionMode.SAMPLE, temperature=10.0)
        selector_low = ActionSelector(mode=SelectionMode.SAMPLE, temperature=0.1)

        counts_high = np.zeros(NUM_TILE_TYPES)
        counts_low = np.zeros(NUM_TILE_TYPES)
        for _ in range(500):
            a, _ = selector_high.select(logits, legal_mask)
            counts_high[a] += 1
            a, _ = selector_low.select(logits, legal_mask)
            counts_low[a] += 1

        # 低温度では action 0 の選択率が高いはず
        assert counts_low[0] > counts_high[0]

    def test_log_prob_valid(self):
        selector = ActionSelector(mode=SelectionMode.SAMPLE)
        logits = torch.randn(NUM_TILE_TYPES)
        legal_mask = torch.ones(NUM_TILE_TYPES)
        _, log_prob = selector.select(logits, legal_mask)
        assert log_prob.item() <= 0  # log probability は非正

    def test_batch_select(self):
        selector = ActionSelector(mode=SelectionMode.SAMPLE)
        batch_size = 8
        logits = torch.randn(batch_size, NUM_TILE_TYPES)
        legal_mask = torch.ones(batch_size, NUM_TILE_TYPES)
        # 一部を非合法に
        legal_mask[:, 30:] = 0.0

        actions, log_probs = selector.select_batch(logits, legal_mask)
        assert actions.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)
        assert (actions < 30).all()  # 非合法手を選ばない

    def test_batch_argmax(self):
        selector = ActionSelector(mode=SelectionMode.ARGMAX)
        batch_size = 4
        logits = torch.zeros(batch_size, NUM_TILE_TYPES)
        logits[0, 5] = 10.0
        logits[1, 10] = 10.0
        logits[2, 15] = 10.0
        logits[3, 20] = 10.0
        legal_mask = torch.ones(batch_size, NUM_TILE_TYPES)

        actions, _ = selector.select_batch(logits, legal_mask)
        assert actions.tolist() == [5, 10, 15, 20]
