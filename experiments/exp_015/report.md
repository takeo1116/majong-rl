# Experiment Report: exp_015

作成日: 2026-03-07  
対象: `experiments/exp_015/runbook.md`

## 1. 実験概要

目的: `shanten_hint` on/off を **warm start + PPO** 条件で比較し、
imitation-only（exp_014）で見えた小幅改善が PPO 後にも残るかを確認する。

比較条件:
- A: `feature_encoder.shanten_hint={"enabled":false}`
- B: `feature_encoder.shanten_hint={"enabled":true}`

共通条件（主要）:
- seeds: `42,43,44,45,46`
- `selfplay.num_matches=200`, `selfplay.num_workers=10`
- `imitation.num_workers=10`, `selfplay.imitation_matches=25`, `training.imitation_epochs=4`
- `training.lr=0.0001`, `epochs=4`, `value_loss_coef=0.25`, `batch_size=256`, `gae_lambda=0.95`
- eval: `rotation`, `num_matches=50`, `num_workers=10`

## 2. 実行結果

| 条件 | shanten_hint | batch_dir | success |
|---|---|---|---:|
| A | off | `runs/20260307_stage1_full_flat_mlp_imitation_then_ppo_batch_9b66f4ab` | 5/5 |
| B | on  | `runs/20260307_stage1_full_flat_mlp_imitation_then_ppo_batch_bca44946` | 5/5 |

両条件とも `aggregate.eval_mode=rotation`。  
各 run で `eval_before/eval_rotation.json`, `eval/eval_rotation.json`, `eval/eval_diff.json` を確認。

## 3. 主評価（eval_before -> eval の delta）

mean ± std（seed=5）

| 条件 | Δavg_rank | Δavg_score | Δdeal_in_rate | Δwin_rate |
|---|---:|---:|---:|---:|
| A (off) | +0.0450 ± 0.0352 | -600.5 ± 327.5 | +0.00234 ± 0.00822 | -0.00845 ± 0.00331 |
| B (on)  | +0.0550 ± 0.0874 | -869.7 ± 1274.1 | +0.00120 ± 0.00703 | -0.00923 ± 0.01002 |

差分（B - A）:
- `Δavg_rank`: **+0.0100**（悪化）
- `Δavg_score`: **-269.2**（悪化）
- `Δdeal_in_rate`: **-0.00114**（改善）
- `Δwin_rate`: **-0.00078**（悪化）

所見:
- 主評価優先順（`Δavg_rank -> Δavg_score -> Δdeal_in_rate -> Δwin_rate`）では **A(off) 優位**。
- B(on) は `Δdeal_in_rate` だけ僅かに良いが、`Δavg_rank` と `Δavg_score` が悪化。

## 4. after 指標（eval 後）

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A (off) | 3.4220 ± 0.0271 | -13685.7 ± 508.5 | 0.04905 ± 0.00235 | 0.57085 ± 0.00890 |
| B (on)  | 3.4430 ± 0.1008 | -14025.1 ± 1464.5 | 0.04208 ± 0.00657 | 0.57113 ± 0.01399 |

差分（B - A）:
- `avg_rank`: **+0.0210**（悪化）
- `avg_score`: **-339.4**（悪化）
- `win_rate`: **-0.00696**（悪化）
- `deal_in_rate`: **+0.00027**（わずか悪化）

所見:
- after 指標でも **A(off) が全面優位**。

## 5. 追跡情報（encoder / 入力次元）

- A: `summary.encoder_features.shanten_hint=false`, `input_dim=455`
- B: `summary.encoder_features.shanten_hint=true`, `input_dim=489`

`config.yaml` / `summary.json` / `notes.md` で on/off と input_dim を追跡可能。

## 6. 時間・補助観測

1 run 平均（sec）

| 条件 | imitation | selfplay | eval_before | learner | eval | total |
|---|---:|---:|---:|---:|---:|---:|
| A (off) | 45.12 | 17.20 | 266.46 | 14.56 | 265.37 | 608.71 |
| B (on)  | 46.08 | 42.21 | 266.91 | 14.43 | 263.43 | 633.06 |

補助観測:
- imitation loss（mean）
  - A: 2.23765
  - B: 2.23279

所見:
- B(on) は total で約 +24.4s/run（約 +4.0%）。
- 今回は selfplay 時間差が大きく出た（A 17.2s vs B 42.2s）。再現性確認の余地あり。

## 7. 結論

1. **Runbook 15 の条件では `shanten_hint=on` は採用見送り**。  
2. 主評価（delta）・after 指標ともに、総合で `shanten_hint=off` が良い。  
3. exp_014（imitation-only）で見えた小幅改善は、PPO を含めると維持できなかった。

## 8. 次アクション

1. 当面の baseline は `shanten_hint=off` を維持。  
2. `shanten_hint` は現形で固定導入せず、必要なら将来「入れ方変更（弱化/別表現）」で再検討。  
3. 追加で切り分けるなら、`imitation` の教師再現度（行動一致率）を直接測る小実験を先に実施する。
