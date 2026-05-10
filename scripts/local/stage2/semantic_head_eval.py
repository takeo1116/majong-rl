#!/usr/bin/env python3
"""CQ-0258: semantic auxiliary trunk の eval-only diagnostics

Stage02a checkpoint + shard から terminal_head / yaku_head の予測品質を測定する。

使い方:
    python scripts/local/stage2/semantic_head_eval.py \
        --config configs/stage2a_xxx.yaml \
        --checkpoint runs/xxx/checkpoints/checkpoint_imitation.pt \
        --shard-dir runs/xxx/selfplay/ \
        --output-dir results/semantic_eval/ \
        --label imitation

    # final checkpoint との比較
    python scripts/local/stage2/semantic_head_eval.py \
        --config configs/stage2a_xxx.yaml \
        --checkpoint runs/xxx/checkpoints/checkpoint_learner.pt \
        --shard-dir runs/xxx/selfplay/ \
        --output-dir results/semantic_eval/ \
        --label final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# project root を path に追加
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT / "python"))


def main():
    parser = argparse.ArgumentParser(
        description="Stage02a semantic head eval-only diagnostics")
    parser.add_argument("--config", required=True, help="config YAML path")
    parser.add_argument("--checkpoint", required=True, help="checkpoint .pt path")
    parser.add_argument("--shard-dir", required=True, help="shard directory")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--label", default="", help="eval label (e.g. imitation, final)")
    parser.add_argument("--device", default="cpu", help="inference device")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    import yaml
    import torch
    import numpy as np

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mc = cfg.get("model", {})
    ec = cfg.get("feature_encoder", cfg.get("encoder", {}))

    # encoder
    from mahjong_rl.encoders import FlatFeatureEncoder
    from mahjong_rl.runner import _parse_encoder_flag
    enc = FlatFeatureEncoder(
        observation_mode=cfg.get("experiment", {}).get("observation_mode", "full"),
        shanten_hint=_parse_encoder_flag(ec, "shanten_hint"),
        discard_ukeire_hint=_parse_encoder_flag(ec, "discard_ukeire_hint"),
        current_shanten_input=_parse_encoder_flag(ec, "current_shanten"),
        shape_hint=_parse_encoder_flag(ec, "shape_hint"),
        turn_context=_parse_encoder_flag(ec, "turn_context"),
        opponent_current_shanten=_parse_encoder_flag(ec, "opponent_current_shanten"),
        opponent_tenpai_flag=_parse_encoder_flag(ec, "opponent_tenpai_flag"),
        danger_mask=_parse_encoder_flag(ec, "danger_mask"),
        tile_presence_flags=_parse_encoder_flag(ec, "tile_presence_flags"),
    )
    input_dim = int(np.prod(enc.metadata().output_shape))

    # model
    from mahjong_rl.models.stage2a_model import Stage2aModel
    sa_config = mc.get("semantic_aux") or {}
    meta = enc.metadata()
    # CQ-0265: direct hint ranges
    hint_ranges = {s: meta.feature_ranges[s]
                   for s in ("shanten_hint", "discard_ukeire_hint")
                   if s in meta.feature_ranges}
    # CQ-0273: semantic-only ranges
    sem_only = {}
    if sa_config.get("tile_presence_flags_semantic_only", False):
        if "tile_presence_flags" in meta.feature_ranges:
            sem_only["tile_presence_flags"] = meta.feature_ranges["tile_presence_flags"]
    model = Stage2aModel(
        input_dim=input_dim,
        discard_hidden_dims=mc.get("discard_hidden_dims", [256, 128]),
        optional_hidden_dims=mc.get("optional_hidden_dims",
            mc.get("call_hidden_dims", [128, 64])),
        value_hidden_dims=mc.get("value_hidden_dims", [128, 64]),
        candidate_dim=mc.get("candidate_dim", 16),
        optional_scorer_hidden=mc.get("optional_scorer_hidden", 32),
        semantic_aux_config=sa_config if sa_config else None,
        direct_hint_ranges=hint_ranges if hint_ranges else None,
        semantic_only_ranges=sem_only if sem_only else None,
    )

    device = torch.device(args.device)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # CQ-0288: 旧 checkpoint の semantic_proj.* keys を互換的に drop
    from mahjong_rl.models.stage2a_model import load_stage2a_state_dict
    load_stage2a_state_dict(model, sd)

    # eval
    from mahjong_rl.semantic_eval import evaluate_semantic_heads, format_summary

    result = evaluate_semantic_heads(
        model=model,
        shard_dir=Path(args.shard_dir),
        device=device,
        batch_size=args.batch_size,
        label=args.label,
        checkpoint_path=args.checkpoint,
    )

    # output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.label}" if args.label else ""
    json_path = out_dir / f"semantic_eval{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"JSON: {json_path}")

    summary_path = out_dir / f"semantic_eval{suffix}_summary.md"
    summary_text = format_summary(result)
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"Summary: {summary_path}")

    # stdout
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
