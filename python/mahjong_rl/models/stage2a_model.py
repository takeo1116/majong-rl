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
    semantic: dict | None = None  # CQ-0256: {terminal_logits, yaku_logits, semantic_summary}


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
        semantic_aux_config: dict | None = None,
        direct_hint_ranges: dict[str, tuple[int, int]] | None = None,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._value_aux_dim = value_aux_dim
        self._candidate_dim = candidate_dim
        sa = semantic_aux_config or {}
        self._semantic_aux_enabled = sa.get("enabled", False)

        # CQ-0265: direct hint branch for discard path
        self._direct_hint_ranges = direct_hint_ranges or {}
        self._direct_hint_sources = sorted(self._direct_hint_ranges.keys())
        self._direct_hints_enabled = len(self._direct_hint_sources) > 0
        excluded_dim = sum(
            e - s for s, e in self._direct_hint_ranges.values()
        ) if self._direct_hints_enabled else 0
        trunk_input_dim = input_dim - excluded_dim

        if self._direct_hints_enabled:
            num_sources = len(self._direct_hint_sources)
            tile_emb_dim = 4
            local_hidden = 16
            self._tile_embedding = nn.Embedding(34, tile_emb_dim)
            self._local_scorer = nn.Sequential(
                nn.Linear(num_sources + tile_emb_dim, local_hidden),
                nn.ReLU(),
                nn.Linear(local_hidden, 1),
            )
            self.register_buffer("_tile_ids", torch.arange(34, dtype=torch.long))

        # Discard trunk
        discard_layers: list[nn.Module] = []
        prev = trunk_input_dim
        for h in discard_hidden_dims:
            discard_layers.append(nn.Linear(prev, h))
            discard_layers.append(nn.ReLU())
            prev = h
        self.discard_trunk = nn.Sequential(*discard_layers)
        self.discard_head = nn.Linear(prev, 34)
        self._discard_trunk_out_dim = prev

        # CQ-0265: context gate (trunk hidden → 34-way sigmoid gate)
        if self._direct_hints_enabled:
            self._context_gate = nn.Linear(prev, 34)

        # Optional trunk (input: x_state + response_context)
        # Uses trunk_input_dim (hint excluded)
        opt_layers: list[nn.Module] = []
        prev_c = trunk_input_dim + RESPONSE_CONTEXT_DIM
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
        # Uses trunk_input_dim (hint excluded)
        summary_dim = self._SUMMARY_FIXED + candidate_dim * 2  # mean + max
        val_input = trunk_input_dim + 1 + RESPONSE_CONTEXT_DIM + summary_dim + value_aux_dim
        val_layers: list[nn.Module] = []
        prev_v = val_input
        for h in value_hidden_dims:
            val_layers.append(nn.Linear(prev_v, h))
            val_layers.append(nn.ReLU())
            prev_v = h
        self.value_trunk = nn.Sequential(*val_layers)
        self.value_head = nn.Linear(prev_v, 1)
        self._summary_dim = summary_dim

        # CQ-0256: semantic auxiliary trunk + heads
        self._semantic_summary_dim = 0
        if self._semantic_aux_enabled:
            from mahjong_rl.outcome_vocab import NUM_TERMINAL_CLASSES, NUM_YAKU
            sa_proj = sa.get("policy_projection_dim", 16)
            # reuse value_trunk hidden → semantic heads
            self.terminal_head = nn.Linear(prev_v, NUM_TERMINAL_CLASSES)
            self.yaku_head = nn.Linear(prev_v, NUM_YAKU)
            self.semantic_proj = nn.Linear(prev_v, sa_proj)
            # summary dim = terminal_probs + yaku_probs + projection
            self._semantic_summary_dim = NUM_TERMINAL_CLASSES + NUM_YAKU + sa_proj

            # expand discard / optional trunk input to accept semantic summary
            # rebuild first layer of each trunk to accept wider input
            d_first_in = trunk_input_dim + self._semantic_summary_dim
            o_first_in = trunk_input_dim + RESPONSE_CONTEXT_DIM + self._semantic_summary_dim
            # replace first linear layers
            d_first_hidden = discard_hidden_dims[0] if discard_hidden_dims else 128
            o_first_hidden = optional_hidden_dims[0] if optional_hidden_dims else 64
            old_d = list(self.discard_trunk)
            old_d[0] = nn.Linear(d_first_in, d_first_hidden)
            self.discard_trunk = nn.Sequential(*old_d)
            old_o = list(self.optional_trunk)
            old_o[0] = nn.Linear(o_first_in, o_first_hidden)
            self.optional_trunk = nn.Sequential(*old_o)

    def _split_features(self, features: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """CQ-0265: features を global (trunk 入力) と direct hints に分離する

        Returns:
            (global_features, direct_hints)
            direct_hints: (B, 34, K) or None
        """
        if not self._direct_hints_enabled:
            return features, None

        exclude_ranges = [
            self._direct_hint_ranges[src]
            for src in self._direct_hint_sources
        ]
        # hint parts
        hint_parts = [features[:, s:e] for s, e in exclude_ranges]  # each (B, 34)

        # global: keep everything except excluded ranges
        total_dim = features.size(-1)
        excluded = set()
        for s, e in exclude_ranges:
            excluded.update(range(s, e))
        keep = [i for i in range(total_dim) if i not in excluded]
        keep_idx = torch.tensor(keep, device=features.device, dtype=torch.long)
        global_features = features.index_select(1, keep_idx)

        hints_stacked = torch.stack(hint_parts, dim=1)  # (B, K, 34)
        direct_hints = hints_stacked.transpose(1, 2)     # (B, 34, K)
        return global_features, direct_hints

    def _apply_direct_hints(self, base_logits: torch.Tensor,
                             direct_hints: torch.Tensor,
                             h_trunk: torch.Tensor) -> torch.Tensor:
        """CQ-0265: tile-wise local scorer + context gate で delta logits を加算"""
        B = base_logits.size(0)
        tile_emb = self._tile_embedding(
            self._tile_ids.expand(B, -1))  # (B, 34, tile_emb_dim)
        local_input = torch.cat([direct_hints, tile_emb], dim=-1)
        delta_logits = self._local_scorer(local_input).squeeze(-1)  # (B, 34)
        gate = torch.sigmoid(self._context_gate(h_trunk))  # (B, 34)
        return base_logits + gate * delta_logits

    def _compute_value_hidden(
        self, features, decision_family, response_context=None,
        optional_summary=None, value_aux_features=None,
    ) -> torch.Tensor:
        """value trunk の hidden 表現を返す (value_head 前)"""
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
        return self.value_trunk(val_in)

    def _compute_semantic(self, h_value: torch.Tensor) -> dict:
        """CQ-0256: semantic heads + detached summary for policy"""
        terminal_logits = self.terminal_head(h_value)
        yaku_logits = self.yaku_head(h_value)
        proj = self.semantic_proj(h_value)
        # summary: detach for policy input
        summary = torch.cat([
            torch.softmax(terminal_logits, dim=-1).detach(),
            torch.sigmoid(yaku_logits).detach(),
            proj.detach(),
        ], dim=-1)
        return {
            "terminal_logits": terminal_logits,
            "yaku_logits": yaku_logits,
            "semantic_summary": summary,
        }

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
        # CQ-0265: split features into global (trunk) + direct hints
        global_features, direct_hints = self._split_features(features)

        # CQ-0256: semantic summary → policy input (always when enabled)
        semantic = None
        policy_input = global_features
        h_v = None
        if self._semantic_aux_enabled:
            h_v = self._compute_value_hidden(
                global_features, decision_family=0.0,
                value_aux_features=value_aux_features)
            semantic = self._compute_semantic(h_v)
            policy_input = torch.cat([global_features, semantic["semantic_summary"]], dim=-1)

        h_d = self.discard_trunk(policy_input)
        logits = self.discard_head(h_d)

        # CQ-0265: direct hint branch (discard only)
        if self._direct_hints_enabled and direct_hints is not None:
            logits = self._apply_direct_hints(logits, direct_hints, h_d)

        logits = logits + (1.0 - legal_mask) * (-1e9)

        values: dict[str, torch.Tensor] = {}
        if compute_value:
            if self._semantic_aux_enabled and h_v is not None:
                values["round_delta"] = self.value_head(h_v)
            else:
                value = self._compute_value(
                    global_features, decision_family=0.0,
                    value_aux_features=value_aux_features)
                values["round_delta"] = value

        return Stage2aOutput(
            discard_logits=logits,
            optional_scores=None,
            values=values,
            semantic=semantic,
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
        # CQ-0265: strip direct hints (optional/value/semantic don't use them)
        global_features, _ = self._split_features(features)

        B_feat = global_features.size(0)
        if response_context is None:
            response_context = torch.zeros(B_feat, RESPONSE_CONTEXT_DIM, device=global_features.device)

        # CQ-0256: semantic summary → policy input (always when enabled)
        semantic = None
        h_v = None
        if self._semantic_aux_enabled:
            cand_enc_pre = self.candidate_encoder(cand_features)
            opt_summary_pre = self._make_optional_summary(
                cand_enc_pre, cand_mask, cand_features=cand_features)
            h_v = self._compute_value_hidden(
                global_features, decision_family=1.0,
                response_context=response_context,
                optional_summary=opt_summary_pre,
                value_aux_features=value_aux_features)
            semantic = self._compute_semantic(h_v)
            opt_input = torch.cat([global_features, response_context,
                                    semantic["semantic_summary"]], dim=-1)
        else:
            opt_input = torch.cat([global_features, response_context], dim=-1)
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

        # Value
        values: dict[str, torch.Tensor] = {}
        if compute_value:
            if self._semantic_aux_enabled:
                values["round_delta"] = self.value_head(h_v)
            else:
                opt_summary = self._make_optional_summary(
                    cand_enc, cand_mask, cand_features=cand_features)
                value = self._compute_value(
                    global_features, decision_family=1.0,
                    response_context=response_context,
                    optional_summary=opt_summary,
                    value_aux_features=value_aux_features)
                values["round_delta"] = value

        return Stage2aOutput(
            discard_logits=None,
            optional_scores=scores,
            values=values,
            semantic=semantic,
        )
