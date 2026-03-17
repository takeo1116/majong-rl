"""MLPPolicyValueModel: MLP ベースの打牌ポリシー + バリューモデル"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .base import DiscardPolicyModel, PolicyValueOutput

_LARGE_NEGATIVE = -1e9


class MLPPolicyValueModel(DiscardPolicyModel):
    """MLP ベースのポリシー・バリューモデル

    入力: フラット特徴量 (batch, input_dim)
    出力: 34種打牌ロジット + バリューヘッド

    CQ-0203: policy_direct_hints 有効時は、指定された牌別 hint を
    shared trunk から除外し、policy logits 直前に専用 branch で加算する。

    Args:
        input_dim: 入力特徴量の次元数
        hidden_dims: 隠れ層の次元数リスト
        value_heads: バリューヘッドの名前リスト
        policy_direct_hints_config: CQ-0203 direct hint branch 設定
        direct_hint_ranges: source 名 -> (start, end) の index range
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        value_heads: Sequence[str] = ("round_delta",),
        value_aux_dim: int = 0,
        policy_tower_config: dict | None = None,
        value_tower_config: dict | None = None,
        policy_direct_hints_config: dict | None = None,
        direct_hint_ranges: dict[str, tuple[int, int]] | None = None,
    ):
        super().__init__()
        self._value_head_names = list(value_heads)
        self._value_aux_dim = value_aux_dim

        # CQ-0203: direct hint branch setup
        pdh = policy_direct_hints_config or {}
        self._direct_hints_enabled = pdh.get("enabled", False)
        self._direct_hint_ranges = direct_hint_ranges or {}
        self._direct_hint_sources: list[str] = []

        trunk_input_dim = input_dim
        if self._direct_hints_enabled:
            sources = pdh.get("sources", [])
            self._direct_hint_sources = list(sources)
            # trunk 入力から direct hint 次元を除外
            excluded_dim = sum(
                (e - s) for s, e in
                (self._direct_hint_ranges[src] for src in self._direct_hint_sources)
            )
            trunk_input_dim = input_dim - excluded_dim

            num_sources = len(self._direct_hint_sources)
            tile_emb_dim = pdh.get("tile_embedding_dim", 4)
            local_hidden = pdh.get("local_hidden_dim", 16)
            self._tile_embedding = nn.Embedding(34, tile_emb_dim)
            local_in = num_sources + tile_emb_dim
            self._local_scorer = nn.Sequential(
                nn.Linear(local_in, local_hidden),
                nn.ReLU(),
                nn.Linear(local_hidden, 1),
            )
            gate_cfg = pdh.get("context_gate", {})
            self._context_gate_enabled = gate_cfg.get("enabled", False)
            self._context_gate: nn.Module | None = None
            # register tile_ids buffer
            self.register_buffer(
                "_tile_ids", torch.arange(34, dtype=torch.long))

        # 共有トランク
        layers: list[nn.Module] = []
        in_dim = trunk_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)
        trunk_out_dim = in_dim

        # context gate (trunk_out_dim が確定してから作成)
        if self._direct_hints_enabled and self._context_gate_enabled:
            self._context_gate = nn.Linear(trunk_out_dim, 34)

        # CQ-0157: policy tower (optional)
        policy_head_in = trunk_out_dim
        if policy_tower_config is not None and policy_tower_config.get("enabled", False):
            pt_hidden = policy_tower_config.get("hidden_dim", trunk_out_dim)
            self.policy_tower: nn.Module | None = nn.Sequential(
                nn.Linear(trunk_out_dim, pt_hidden),
                nn.ReLU(),
            )
            policy_head_in = pt_hidden
        else:
            self.policy_tower = None

        # ポリシーヘッド
        self.policy_head = nn.Linear(policy_head_in, 34)

        # CQ-0157: value tower (optional)
        value_tower_in = trunk_out_dim + value_aux_dim
        if value_tower_config is not None and value_tower_config.get("enabled", False):
            vt_hidden = value_tower_config.get("hidden_dim", trunk_out_dim)
            self.value_tower: nn.Module | None = nn.Sequential(
                nn.Linear(value_tower_in, vt_hidden),
                nn.ReLU(),
            )
            value_head_in = vt_hidden
        else:
            self.value_tower = None
            value_head_in = value_tower_in

        # バリューヘッド群 (CQ-0151: value_aux_dim > 0 なら trunk + aux を入力)
        self.value_heads_modules = nn.ModuleDict({
            name: nn.Linear(value_head_in, 1) for name in value_heads
        })

    def _split_features(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """CQ-0203: features を global 部分と direct hint 部分に分離する

        Returns:
            (global_features, direct_hints[B, 34, K])
        """
        # 除外する index range を集め、残す部分と hint 部分に分ける
        total_dim = features.size(-1)
        exclude_ranges = [
            self._direct_hint_ranges[src]
            for src in self._direct_hint_sources
        ]
        # hint 部分: [B, K*34] -> reshape -> [B, 34, K]
        hint_parts = []
        for s, e in exclude_ranges:
            hint_parts.append(features[:, s:e])  # each [B, 34]

        # global 部分: exclude_ranges 以外の連続部分を concat
        keep_indices: list[int] = []
        excluded = set()
        for s, e in exclude_ranges:
            excluded.update(range(s, e))
        for i in range(total_dim):
            if i not in excluded:
                keep_indices.append(i)
        keep_idx = torch.tensor(keep_indices, device=features.device, dtype=torch.long)
        global_features = features.index_select(1, keep_idx)

        # direct hints: [B, K, 34] -> [B, 34, K]
        if hint_parts:
            hints_stacked = torch.stack(hint_parts, dim=1)  # [B, K, 34]
            direct_hints = hints_stacked.transpose(1, 2)    # [B, 34, K]
        else:
            direct_hints = torch.zeros(
                features.size(0), 34, 0, device=features.device)

        return global_features, direct_hints

    def forward(self, features: Tensor, legal_mask: Tensor,
                value_aux_features: Tensor | None = None) -> PolicyValueOutput:
        if self._direct_hints_enabled:
            global_features, direct_hints = self._split_features(features)
            h = self.trunk(global_features)
        else:
            h = self.trunk(features)

        # CQ-0157: policy tower (optional)
        h_policy = self.policy_tower(h) if self.policy_tower is not None else h
        base_logits = self.policy_head(h_policy)

        # CQ-0203: direct hint branch
        if self._direct_hints_enabled:
            B = features.size(0)
            tile_emb = self._tile_embedding(
                self._tile_ids.expand(B, -1))  # [B, 34, tile_emb_dim]
            local_input = torch.cat(
                [direct_hints, tile_emb], dim=-1)  # [B, 34, K+tile_emb_dim]
            delta_logits = self._local_scorer(
                local_input).squeeze(-1)  # [B, 34]
            if self._context_gate is not None:
                gate = torch.sigmoid(self._context_gate(h))  # [B, 34]
                delta_logits = gate * delta_logits
            logits = base_logits + delta_logits
        else:
            logits = base_logits

        logits = logits + (1.0 - legal_mask) * _LARGE_NEGATIVE

        # バリューヘッド (CQ-0151: value_aux_features があれば concat)
        # CQ-0153: aux 未渡し時は zero pad で value head 入力次元を合わせる
        if self._value_aux_dim > 0:
            if value_aux_features is not None:
                h_value = torch.cat([h, value_aux_features], dim=-1)
            else:
                zeros = torch.zeros(
                    h.size(0), self._value_aux_dim, device=h.device, dtype=h.dtype)
                h_value = torch.cat([h, zeros], dim=-1)
        else:
            h_value = h
        # CQ-0157: value tower (optional)
        h_value = self.value_tower(h_value) if self.value_tower is not None else h_value
        values = {
            name: head(h_value) for name, head in self.value_heads_modules.items()
        }

        return PolicyValueOutput(logits=logits, values=values)

    @property
    def value_head_names(self) -> list[str]:
        return self._value_head_names
