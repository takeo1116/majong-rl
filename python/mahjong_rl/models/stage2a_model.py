"""CQ-0225: Stage2a DiscardPolicy / CallPolicy 分離モデル

discard_trunk と optional_trunk を分離し、optional 側は candidate scorer で
legal candidate ごとの scalar score を出す。

forward() は decision_type に応じて discard path / optional path を分岐する。
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
import torch.nn as nn


RESPONSE_CONTEXT_DIM = 3  # tile_type/34 + rel_seat/4 + menzen_flag


class Stage2aOutput(NamedTuple):
    """Stage2a model の出力"""
    discard_logits: torch.Tensor | None  # (B, 34) or None
    optional_scores: torch.Tensor | None     # (B, max_cands) or None
    values: dict[str, torch.Tensor]      # {"round_delta": (B, 1)}


# ---------- Candidate Encoder ----------

class CandidateEncoder(nn.Module):
    """legal candidate を固定長ベクトルにエンコードする

    入力: action_type_idx, tile_type, rel_seat, consumed_tile_types
    出力: (candidate_dim,)
    """

    def __init__(self, candidate_dim: int = 16):
        super().__init__()
        self._candidate_dim = candidate_dim
        # action_type: Skip=0, Chi=1, Pon=2, Daiminkan=3 → 4 種 (no padding)
        self.action_type_emb = nn.Embedding(4, 4)
        # tile_type: valid=1..34 (0-based +1), padding=0 → vocab=35
        self.tile_emb = nn.Embedding(35, 8, padding_idx=0)
        # rel_seat: valid=1..4, padding=0 → vocab=5
        self.seat_emb = nn.Embedding(5, 4, padding_idx=0)
        # consumed tile types: valid=1..34, padding=0 → vocab=35
        self.consumed_emb = nn.Embedding(35, 4, padding_idx=0)
        # input: 4 + 8 + 4 + 4*3 = 28
        self.fc = nn.Linear(28, candidate_dim)

    def forward(self, cand_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cand_features: (B, max_cands, 6) int
                [action_type_idx, tile_type_1based, rel_seat_1based,
                 consumed0_1based, consumed1_1based, consumed2_1based]
                0 = padding sentinel (embedding の padding_idx=0)

        Returns:
            (B, max_cands, candidate_dim)
        """
        at = self.action_type_emb(cand_features[..., 0])     # (B, C, 4)
        tt = self.tile_emb(cand_features[..., 1])             # (B, C, 8)
        rs = self.seat_emb(cand_features[..., 2])             # (B, C, 4)
        c0 = self.consumed_emb(cand_features[..., 3])         # (B, C, 4)
        c1 = self.consumed_emb(cand_features[..., 4])         # (B, C, 4)
        c2 = self.consumed_emb(cand_features[..., 5])         # (B, C, 4)
        x = torch.cat([at, tt, rs, c0, c1, c2], dim=-1)      # (B, C, 28)
        return torch.relu(self.fc(x))                          # (B, C, candidate_dim)


# ---------- Stage2a Model ----------

class Stage2aModel(nn.Module):
    """Stage2a 3-branch モデル (CQ-0242)

    discard_trunk: x_state -> h_discard -> discard_logits (34)
    optional_trunk: x_state -> h_optional, candidate scorer
    value_trunk: x_state + decision_context -> value (独立)
    """

    # optional_summary の固定長次元
    # optional_available(1) + candidate_count_norm(1) + action_type_presence(4)
    # + candidate_embedding_mean(cand_dim) + candidate_embedding_max(cand_dim)
    _SUMMARY_FIXED = 6  # available + count_norm + 4 action types

    def __init__(
        self,
        input_dim: int,
        discard_hidden_dims: Sequence[int] = (256, 128),
        optional_hidden_dims: Sequence[int] = (128, 64),
        value_hidden_dims: Sequence[int] = (128, 64),
        candidate_dim: int = 16,
        optional_scorer_hidden: int = 32,
        value_aux_dim: int = 0,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._value_aux_dim = value_aux_dim
        self._candidate_dim = candidate_dim

        # Discard trunk
        discard_layers: list[nn.Module] = []
        prev = input_dim
        for h in discard_hidden_dims:
            discard_layers.append(nn.Linear(prev, h))
            discard_layers.append(nn.ReLU())
            prev = h
        self.discard_trunk = nn.Sequential(*discard_layers)
        self.discard_head = nn.Linear(prev, 34)

        # Optional trunk (input: x_state + response_context)
        opt_layers: list[nn.Module] = []
        prev_c = input_dim + RESPONSE_CONTEXT_DIM
        for h in optional_hidden_dims:
            opt_layers.append(nn.Linear(prev_c, h))
            opt_layers.append(nn.ReLU())
            prev_c = h
        self.optional_trunk = nn.Sequential(*opt_layers)
        self._optional_trunk_dim = prev_c

        # Candidate encoder + scorer
        self.candidate_encoder = CandidateEncoder(candidate_dim)
        self.optional_scorer = nn.Sequential(
            nn.Linear(prev_c + candidate_dim, optional_scorer_hidden),
            nn.ReLU(),
            nn.Linear(optional_scorer_hidden, 1),
        )

        # Value trunk (CQ-0242: independent from policy)
        # input: x_state + decision_family(1) + response_context + optional_summary
        summary_dim = self._SUMMARY_FIXED + candidate_dim * 2  # mean + max
        val_input = input_dim + 1 + RESPONSE_CONTEXT_DIM + summary_dim + value_aux_dim
        val_layers: list[nn.Module] = []
        prev_v = val_input
        for h in value_hidden_dims:
            val_layers.append(nn.Linear(prev_v, h))
            val_layers.append(nn.ReLU())
            prev_v = h
        self.value_trunk = nn.Sequential(*val_layers)
        self.value_head = nn.Linear(prev_v, 1)
        self._summary_dim = summary_dim

    def _compute_value(
        self, features: torch.Tensor,
        decision_family: float,
        response_context: torch.Tensor | None = None,
        optional_summary: torch.Tensor | None = None,
        value_aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """CQ-0242: independent value trunk"""
        B = features.size(0)
        df = torch.full((B, 1), decision_family, device=features.device)
        if response_context is None:
            response_context = torch.zeros(B, RESPONSE_CONTEXT_DIM, device=features.device)
        if optional_summary is None:
            optional_summary = torch.zeros(B, self._summary_dim, device=features.device)
        parts = [features, df, response_context, optional_summary]
        if self._value_aux_dim > 0 and value_aux_features is not None:
            parts.append(value_aux_features)
        elif self._value_aux_dim > 0:
            parts.append(torch.zeros(B, self._value_aux_dim, device=features.device))
        val_in = torch.cat(parts, dim=-1)
        h_v = self.value_trunk(val_in)
        return self.value_head(h_v)

    def _make_optional_summary(
        self, cand_enc: torch.Tensor, cand_mask: torch.Tensor,
        cand_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """candidate 集合を固定長 summary に圧縮

        action_type_presence: Skip(0)/Chi(1)/Pon(2)/Daiminkan(3) の有無
        """
        B, C, D = cand_enc.shape
        # available flag
        available = (cand_mask.sum(dim=-1, keepdim=True) > 0).float()
        # count norm: 固定上限 10 で正規化 (batch 非依存)
        _MAX_CANDS_NORM = 10.0
        count_norm = cand_mask.sum(dim=-1, keepdim=True) / _MAX_CANDS_NORM
        # action type presence from raw candidate features
        # cand_features[..., 0] is action_type_idx: Skip=0, Chi=1, Pon=2, Daiminkan=3
        at_presence = torch.zeros(B, 4, device=cand_enc.device)
        if cand_features is not None:
            at_idx = cand_features[..., 0]  # (B, C)
            for k in range(4):
                # any valid candidate with this action type?
                match = ((at_idx == k) & (cand_mask > 0.5)).float()
                at_presence[:, k] = (match.sum(dim=-1) > 0).float()
        else:
            at_presence[:, 0] = (cand_mask.sum(dim=-1) > 0).float()
        # mean / max pooling
        masked = cand_enc * cand_mask.unsqueeze(-1)
        count = cand_mask.sum(dim=-1, keepdim=True).clamp(min=1).unsqueeze(-1)
        emb_mean = masked.sum(dim=1) / count.squeeze(-1)
        emb_max = masked.max(dim=1).values
        return torch.cat([available, count_norm, at_presence, emb_mean, emb_max], dim=-1)

    def forward_discard(
        self,
        features: torch.Tensor,
        legal_mask: torch.Tensor,
        value_aux_features: torch.Tensor | None = None,
        compute_value: bool = True,
    ) -> Stage2aOutput:
        """discard decision の forward"""
        h_d = self.discard_trunk(features)
        logits = self.discard_head(h_d)
        logits = logits + (1.0 - legal_mask) * (-1e9)

        values: dict[str, torch.Tensor] = {}
        if compute_value:
            value = self._compute_value(
                features, decision_family=0.0,
                value_aux_features=value_aux_features)
            values["round_delta"] = value

        return Stage2aOutput(
            discard_logits=logits,
            optional_scores=None,
            values=values,
        )

    def forward_optional(
        self,
        features: torch.Tensor,
        cand_features: torch.Tensor,
        cand_mask: torch.Tensor,
        value_aux_features: torch.Tensor | None = None,
        response_context: torch.Tensor | None = None,
        compute_value: bool = True,
    ) -> Stage2aOutput:
        """optional decision の forward

        Args:
            features: (B, input_dim)
            cand_features: (B, max_cands, 6) int — candidate encoding
            cand_mask: (B, max_cands) float — 1.0 = valid candidate
            value_aux_features: (B, aux_dim) optional

        Returns:
            Stage2aOutput with discard_logits=None, optional_scores, values
        """
        B_feat = features.size(0)
        if response_context is None:
            response_context = torch.zeros(B_feat, RESPONSE_CONTEXT_DIM, device=features.device)
        opt_input = torch.cat([features, response_context], dim=-1)
        h_c = self.optional_trunk(opt_input)  # (B, optional_trunk_dim)

        # Candidate encoding
        cand_enc = self.candidate_encoder(cand_features)  # (B, C, cand_dim)

        # Score each candidate
        B, C, _ = cand_enc.shape
        h_expanded = h_c.unsqueeze(1).expand(-1, C, -1)  # (B, C, optional_trunk_dim)
        scorer_input = torch.cat([h_expanded, cand_enc], dim=-1)
        scores = self.optional_scorer(scorer_input).squeeze(-1)  # (B, C)

        # Mask invalid candidates
        scores = scores + (1.0 - cand_mask) * (-1e9)

        # Value (independent trunk, optional → decision_family=1)
        values: dict[str, torch.Tensor] = {}
        if compute_value:
            opt_summary = self._make_optional_summary(
                cand_enc, cand_mask, cand_features=cand_features)
            value = self._compute_value(
                features, decision_family=1.0,
                response_context=response_context,
                optional_summary=opt_summary,
                value_aux_features=value_aux_features)
            values["round_delta"] = value

        return Stage2aOutput(
            discard_logits=None,
            optional_scores=scores,
            values=values,
        )
