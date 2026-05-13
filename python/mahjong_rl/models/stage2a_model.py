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

# CQ-0288: 旧 checkpoint 由来で互換的に無視する key prefix
_LEGACY_DROPPED_KEY_PREFIXES: tuple[str, ...] = (
    "semantic_proj.",
)


class Stage2aOutput(NamedTuple):
    """Stage2a model の出力"""
    discard_logits: torch.Tensor | None  # (B, 34) or None
    optional_scores: torch.Tensor | None     # (B, max_cands) or None
    values: dict[str, torch.Tensor]      # {"round_delta": (B, 1)}
    semantic: dict | None = None  # CQ-0256: {terminal_logits, yaku_logits, semantic_summary}


def load_stage2a_state_dict(model: "Stage2aModel",
                             state_dict: dict,
                             strict: bool = True) -> "torch.nn.modules.module._IncompatibleKeys":
    """CQ-0288 / CQ-0291: 旧 Stage2a checkpoint との互換ロード

    旧 checkpoint との互換性のために以下を行う:

    - CQ-0288: `semantic_proj.weight` / `semantic_proj.bias` を strict load
      前に drop する
    - CQ-0291 batch 1/2: `candidate_encoder.action_type_emb.weight` の行数が
      増えた場合 (旧 4 → 新 6 → 新 8 など)、旧の上 N 行を新 weight の
      上 N 行にコピーし、残り (新行は新 family, batch 1: NoRiichi=4 / Riichi=5、
      batch 2: TsumoWin=6 / Ron=7) は init 値のままにする。
      これにより旧 checkpoint は新 family が未使用な状態で互換動作する

    それ以外の missing/unexpected はそのまま fail-fast する。

    Args:
        model: Stage2aModel
        state_dict: 入力 state dict
        strict: 既定 True

    Returns:
        ``torch.nn.Module.load_state_dict`` の戻り (NamedTuple)
    """
    # CQ-0288: semantic_proj.* を drop
    filtered: dict = {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(p) for p in _LEGACY_DROPPED_KEY_PREFIXES)
    }

    # CQ-0291: action_type_emb の行数差分に対応 (旧 4 行 → 新 6 行)
    EMB_KEY = "candidate_encoder.action_type_emb.weight"
    if EMB_KEY in filtered:
        src_w = filtered[EMB_KEY]
        try:
            tgt_w = model.candidate_encoder.action_type_emb.weight.data
        except AttributeError:
            tgt_w = None
        if (tgt_w is not None and src_w.dim() == 2 and tgt_w.dim() == 2
                and src_w.shape[0] < tgt_w.shape[0]
                and src_w.shape[1] == tgt_w.shape[1]):
            # 上 N 行を src からコピー、残り行は target init 値のまま
            new_w = tgt_w.detach().clone()
            new_w[: src_w.shape[0]] = src_w
            filtered[EMB_KEY] = new_w

    # CQ-0294: optional_summary の action_type_presence を 4 → 11 に拡張した
    # ことで value_trunk.0.weight の column 数が +7 ずれる。旧 checkpoint を
    # 安全にロードできるよう、新 family の presence 列に 0 を挿入する。
    VAL_KEY = "value_trunk.0.weight"
    if VAL_KEY in filtered:
        src_w = filtered[VAL_KEY]
        try:
            tgt_w = model.value_trunk[0].weight.data
        except (AttributeError, IndexError):
            tgt_w = None
        if (tgt_w is not None and src_w.dim() == 2 and tgt_w.dim() == 2
                and src_w.shape[0] == tgt_w.shape[0]
                and src_w.shape[1] < tgt_w.shape[1]):
            # value_trunk 入力レイアウト (CQ-0294 前):
            #   [trunk_input | df=1 | rc=3 | summary | value_aux]
            #   summary = [available=1, count_norm=1, at_presence=K_old=4,
            #              emb_mean=cand_dim, emb_max=cand_dim]
            # 新レイアウト: K_new = NUM_ACTION_TYPE_INDICES (=11)。
            # 差分は (K_new - K_old) 列で、at_presence 末尾と emb_mean の
            # 間に挿入する。挿入位置 = trunk_input + 1 + rc + 2 + K_old。
            from mahjong_rl.candidate_encoding import NUM_ACTION_TYPE_INDICES
            from mahjong_rl.encoders.flat_encoder import FlatFeatureEncoder
            cand_dim = getattr(model, "_candidate_dim", 0)
            value_aux = getattr(model, "_value_aux_dim", 0)
            extra = tgt_w.shape[1] - src_w.shape[1]
            # K_new is fixed in current code; K_old = K_new - extra
            k_new = NUM_ACTION_TYPE_INDICES
            k_old = k_new - extra
            # sanity: cols match the expected shape with K_old action types
            expected_old_cols = (
                src_w.shape[1] - (1 + 3 + 2 + k_old + 2 * cand_dim + value_aux))
            # expected_old_cols = trunk_input_dim
            if (k_old in (4, 6, 8) and expected_old_cols >= 0
                    and 1 + 3 + 2 + k_old <= src_w.shape[1]):
                trunk_input_dim = expected_old_cols
                insert_at = trunk_input_dim + 1 + 3 + 2 + k_old
                # 挿入する extra 列は 0
                left = src_w[:, :insert_at]
                right = src_w[:, insert_at:]
                pad = torch.zeros(
                    src_w.shape[0], extra, dtype=src_w.dtype,
                    device=src_w.device)
                new_w = torch.cat([left, pad, right], dim=1)
                if new_w.shape == tgt_w.shape:
                    filtered[VAL_KEY] = new_w

    return model.load_state_dict(filtered, strict=strict)


# ---------- Candidate Encoder ----------

class CandidateEncoder(nn.Module):
    """legal candidate を固定長ベクトルにエンコードする

    入力: action_type_idx, tile_type, rel_seat, consumed_tile_types
    出力: (candidate_dim,)
    """

    def __init__(self, candidate_dim: int = 16):
        super().__init__()
        self._candidate_dim = candidate_dim
        # action_type embedding 行 (CQ-0291):
        #   0=Skip, 1=Chi, 2=Pon, 3=Daiminkan
        #   4=NoRiichi, 5=Riichi (batch 1)
        #   6=TsumoWin, 7=Ron (batch 2)
        #   8=Kakan, 9=Ankan, 10=Kyuushu (batch 3)
        # default off では batch 1-3 の追加行は未使用 (init 値のまま)
        from mahjong_rl.candidate_encoding import NUM_ACTION_TYPE_INDICES
        self.action_type_emb = nn.Embedding(NUM_ACTION_TYPE_INDICES, 4)
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
    # optional_available(1) + candidate_count_norm(1)
    #   + action_type_presence(NUM_ACTION_TYPE_INDICES)
    #   + candidate_embedding_mean(cand_dim) + candidate_embedding_max(cand_dim)
    # CQ-0294: action_type_presence を NUM_ACTION_TYPE_INDICES (=11) に拡張
    # (Skip/Chi/Pon/Daiminkan/NoRiichi/Riichi/TsumoWin/Ron/Kakan/Ankan/Kyuushu)
    @classmethod
    def _summary_fixed_dim(cls) -> int:
        from mahjong_rl.candidate_encoding import NUM_ACTION_TYPE_INDICES
        return 2 + NUM_ACTION_TYPE_INDICES  # available + count_norm + presence
    _SUMMARY_FIXED = 2 + 11  # available + count_norm + 11 action types

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
        semantic_only_ranges: dict[str, tuple[int, int]] | None = None,
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

        # CQ-0273: semantic-only features (value/semantic trunk に残すが policy trunk から除外)
        self._semantic_only_ranges = semantic_only_ranges or {}
        self._semantic_only_sources = sorted(self._semantic_only_ranges.keys())
        self._semantic_only_enabled = len(self._semantic_only_sources) > 0
        semantic_only_dim = sum(
            e - s for s, e in self._semantic_only_ranges.values()
        ) if self._semantic_only_enabled else 0

        trunk_input_dim = input_dim - excluded_dim  # after hint exclusion
        policy_trunk_dim = trunk_input_dim - semantic_only_dim  # further exclude for policy

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

        # Discard trunk (uses policy_trunk_dim: hint + semantic_only excluded)
        discard_layers: list[nn.Module] = []
        prev = policy_trunk_dim
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
        # Uses policy_trunk_dim (hint + semantic_only excluded)
        opt_layers: list[nn.Module] = []
        prev_c = policy_trunk_dim + RESPONSE_CONTEXT_DIM
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
        # CQ-0288: semantic_proj は detach 後に summary に詰めるだけで loss を
        # 受けず、dead weight だったため削除した。`policy_projection_dim`
        # config は互換性のため受け取るが無視する (ignored/deprecated)。
        self._semantic_summary_dim = 0
        if self._semantic_aux_enabled:
            from mahjong_rl.outcome_vocab import NUM_TERMINAL_CLASSES, NUM_YAKU
            # reuse value_trunk hidden → semantic heads
            self.terminal_head = nn.Linear(prev_v, NUM_TERMINAL_CLASSES)
            self.yaku_head = nn.Linear(prev_v, NUM_YAKU)
            # summary dim = terminal_probs + yaku_probs (CQ-0288: proj 削除)
            self._semantic_summary_dim = NUM_TERMINAL_CLASSES + NUM_YAKU

            # expand discard / optional trunk input to accept semantic summary
            # rebuild first layer of each trunk to accept wider input
            d_first_in = policy_trunk_dim + self._semantic_summary_dim
            o_first_in = policy_trunk_dim + RESPONSE_CONTEXT_DIM + self._semantic_summary_dim
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
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """CQ-0265/0273: features を分離する

        Returns:
            (policy_features, value_features, direct_hints)
            policy_features: hint + semantic_only excluded
            value_features: hint excluded only (semantic_only は含む)
            direct_hints: (B, 34, K) or None
        """
        total_dim = features.size(-1)

        # Build excluded index sets
        hint_excluded = set()
        for src in self._direct_hint_sources:
            s, e = self._direct_hint_ranges[src]
            hint_excluded.update(range(s, e))

        semantic_only_excluded = set()
        for src in self._semantic_only_sources:
            s, e = self._semantic_only_ranges[src]
            semantic_only_excluded.update(range(s, e))

        # value_features: hint excluded only
        value_keep = [i for i in range(total_dim) if i not in hint_excluded]
        # policy_features: hint + semantic_only excluded
        policy_keep = [i for i in range(total_dim)
                       if i not in hint_excluded and i not in semantic_only_excluded]

        if len(value_keep) == total_dim and len(policy_keep) == total_dim:
            # nothing excluded
            return features, features, None

        device = features.device
        value_idx = torch.tensor(value_keep, device=device, dtype=torch.long)
        policy_idx = torch.tensor(policy_keep, device=device, dtype=torch.long)
        value_features = features.index_select(1, value_idx)
        policy_features = features.index_select(1, policy_idx)

        # direct hints
        direct_hints = None
        if self._direct_hints_enabled:
            hint_parts = [features[:, s:e]
                          for src in self._direct_hint_sources
                          for s, e in [self._direct_hint_ranges[src]]]
            hints_stacked = torch.stack(hint_parts, dim=1)  # (B, K, 34)
            direct_hints = hints_stacked.transpose(1, 2)     # (B, 34, K)

        return policy_features, value_features, direct_hints

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
        """CQ-0256: semantic heads + detached summary for policy

        CQ-0288: 旧来の `semantic_proj(h_value).detach()` 成分は削除した
        (dead weight だったため)。summary は terminal/yaku prediction のみ。
        """
        terminal_logits = self.terminal_head(h_value)
        yaku_logits = self.yaku_head(h_value)
        # summary: detach for policy input
        summary = torch.cat([
            torch.softmax(terminal_logits, dim=-1).detach(),
            torch.sigmoid(yaku_logits).detach(),
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

        CQ-0294: action_type_presence を NUM_ACTION_TYPE_INDICES に拡張。
        各 idx (Skip=0, Chi=1, Pon=2, Daiminkan=3, NoRiichi=4, Riichi=5,
        TsumoWin=6, Ron=7, Kakan=8, Ankan=9, Kyuushu=10) について
        valid candidate に存在するかどうかを 0/1 で立てる。
        """
        from mahjong_rl.candidate_encoding import NUM_ACTION_TYPE_INDICES
        B, C, D = cand_enc.shape
        # available flag
        available = (cand_mask.sum(dim=-1, keepdim=True) > 0).float()
        # count norm: 固定上限 10 で正規化 (batch 非依存)
        _MAX_CANDS_NORM = 10.0
        count_norm = cand_mask.sum(dim=-1, keepdim=True) / _MAX_CANDS_NORM
        # action type presence from raw candidate features
        # cand_features[..., 0] is action_type_idx
        at_presence = torch.zeros(
            B, NUM_ACTION_TYPE_INDICES, device=cand_enc.device)
        if cand_features is not None:
            at_idx = cand_features[..., 0]  # (B, C)
            valid = (cand_mask > 0.5)
            for k in range(NUM_ACTION_TYPE_INDICES):
                match = ((at_idx == k) & valid).float()
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
        # CQ-0265/0273: split features
        policy_features, value_features, direct_hints = self._split_features(features)

        # CQ-0256: semantic summary → policy input (always when enabled)
        semantic = None
        policy_input = policy_features
        h_v = None
        if self._semantic_aux_enabled:
            h_v = self._compute_value_hidden(
                value_features, decision_family=0.0,
                value_aux_features=value_aux_features)
            semantic = self._compute_semantic(h_v)
            policy_input = torch.cat([policy_features, semantic["semantic_summary"]], dim=-1)

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
                    value_features, decision_family=0.0,
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
        # CQ-0265/0273: split features
        policy_features, value_features, _ = self._split_features(features)

        B_feat = policy_features.size(0)
        if response_context is None:
            response_context = torch.zeros(B_feat, RESPONSE_CONTEXT_DIM, device=policy_features.device)

        # CQ-0256: semantic summary → policy input (always when enabled)
        semantic = None
        h_v = None
        if self._semantic_aux_enabled:
            cand_enc_pre = self.candidate_encoder(cand_features)
            opt_summary_pre = self._make_optional_summary(
                cand_enc_pre, cand_mask, cand_features=cand_features)
            h_v = self._compute_value_hidden(
                value_features, decision_family=1.0,
                response_context=response_context,
                optional_summary=opt_summary_pre,
                value_aux_features=value_aux_features)
            semantic = self._compute_semantic(h_v)
            opt_input = torch.cat([policy_features, response_context,
                                    semantic["semantic_summary"]], dim=-1)
        else:
            opt_input = torch.cat([policy_features, response_context], dim=-1)
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
                    value_features, decision_family=1.0,
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
