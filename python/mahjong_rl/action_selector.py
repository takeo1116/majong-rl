"""Stage 1 用 ActionSelector: モデル出力 + legal mask → 行動選択"""
import torch
from enum import Enum


class SelectionMode(Enum):
    """選択方式"""
    ARGMAX = "argmax"
    SAMPLE = "sample"


class ActionSelector:
    """モデル出力ロジット + legal mask から行動を選択する

    Stage 1 では 34 種牌ロジットから打牌 TileType を選択する。
    """

    def __init__(
        self,
        mode: SelectionMode = SelectionMode.SAMPLE,
        temperature: float = 1.0,
    ):
        self.mode = mode
        self.temperature = temperature

    def select(
        self,
        logits: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        """単一サンプルの行動選択

        Args:
            logits: (34,) ロジット
            legal_mask: (34,) float32, 1.0=合法

        Returns:
            (選択された TileType, log_prob)
        """
        scaled = logits / self.temperature
        masked = scaled + (1.0 - legal_mask) * (-1e9)
        probs = torch.softmax(masked, dim=-1)

        if self.mode == SelectionMode.ARGMAX:
            action = torch.argmax(probs).item()
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().item()

        log_prob = torch.log(probs[action] + 1e-10)
        return action, log_prob

    def select_batch(
        self,
        logits: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """バッチ版の行動選択

        Args:
            logits: (batch, 34) ロジット
            legal_mask: (batch, 34) float32, 1.0=合法

        Returns:
            (actions: (batch,), log_probs: (batch,))
        """
        scaled = logits / self.temperature
        masked = scaled + (1.0 - legal_mask) * (-1e9)
        probs = torch.softmax(masked, dim=-1)

        if self.mode == SelectionMode.ARGMAX:
            actions = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample()

        log_probs = torch.log(
            probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-10
        )
        return actions, log_probs
