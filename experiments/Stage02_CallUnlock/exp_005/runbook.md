# Experiment Runbook: exp_005

作成日: 2026-03-30  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_004/runbook.md`
- `experiments/Stage02_CallUnlock/exp_004/report.md`
- `configs/stage2a_core_minimal_mixed_minimality_baseline.yaml`

## 1. 背景

`exp_004` では、Stage02a A `core_minimal` を固定して `mixed` PPO の安定化条件を探索した。

その結果、

- M1 `policy_ratio=0.50, baseline_sample_weight=0.50`
- M2 `policy_ratio=0.50, baseline_sample_weight=0.25`
- M3 `M2 + policy_anchor.coef=1.0`

はいずれも不十分で、

- M4 `M3 + lr=1e-4 + clip_epsilon=0.10 + max_grad_norm=0.30`

だけが 20 cycle を通して stable に mixed PPO を回せた。

ただし M4 では以下の 3 ノブを同時に変更している。

1. `lr: 3e-4 -> 1e-4`
2. `clip_epsilon: 0.15 -> 0.10`
3. `max_grad_norm: 0.50 -> 0.30`

したがって次の課題は、**この 3 ノブのうち何が本当に必要で、どれが単独で効くのかを切り分けること**である。

## 2. 問い

M4 の mixed 安定化において、

1. `low_lr`
2. `low_clip`
3. `tight_gradclip`

のどれが必要条件か。

また、

1. `low_lr`
2. `low_clip`
3. `tight_gradclip`

のどれが単独でも効くか。

## 3. 仮説

現時点の仮説は次の順。

1. 主犯は `lr=3e-4` の update size であり、`low_lr` が最も効く
2. `clip_epsilon=0.10` は ratio drift 抑制に補助的に効く
3. `max_grad_norm=0.30` は追加の safety margin として効くが、単独の主因ではない

つまり、いまの本命仮説は

- `low_lr` が必要
- `low_clip` は補助的に必要かもしれない
- `tight_gradclip` は単独では効きにくい

である。

## 4. この実験の位置づけ

この `exp_005` は、`exp_004` で見つかった M4 をそのまま seed 拡張する前に、
**最小有効条件を確定するためのアブレーション実験**である。

狙いは以下。

- mixed baseline を必要以上に保守的な条件で固定しない
- どのノブが効いて mixed を救っているのか理解する
- 今後の A/B/C feature 比較や seed 拡張の足場を、より軽く説明可能な条件にする

## 5. 参照点

### unstable anchor-only reference (M3)

- `exp_004` M3
- run label: `M3_pr050_bsw025_anchor10` （`experiments/Stage02_CallUnlock/exp_004/run_map.json` を参照）
- final `avg_rank=3.33`, `win_rate=0.0905`
- final `ratio_mean=1.17e4`, `clip_fraction=0.3791`, `anchor_kl_discard=2.9179`

### stable mixed candidate (M4)

- `exp_004` M4
- run label: `M4_pr050_bsw025_anchor10_lr1e4_clip010_gn03` （`experiments/Stage02_CallUnlock/exp_004/run_map.json` を参照）
- final `avg_rank=2.23`, `win_rate=0.2487`
- final `ratio_mean=1.0280`, `clip_fraction=0.3053`, `anchor_kl_discard=0.0165`

この 2 条件の差分だけを今回切り分ける。

## 6. 共通条件

固定条件:

- A `core_minimal`
- `training.rule_mix_learner.ppo_mode = "mixed"`
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.coef = 1.0`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`
- `training.imitation_eval.eval_each_chunk = false`

ベース config:

- `configs/stage2a_core_minimal_mixed_minimality_baseline.yaml`

`exp_004` と同様、imitation 直後 eval も保存し、

- imitation 直後から良いのか
- PPO を回して悪化するのか
- PPO 後にさらに改善するのか

を追えるようにする。

## 7. 比較条件

### A. 必要性テスト: M4 から 1 ノブずつ戻す

#### N1 `no_low_lr`

- `lr = 3e-4`
- `clip_epsilon = 0.10`
- `max_grad_norm = 0.30`

狙い:

- `low_lr` を外すと M4 が壊れるかを見る

#### N2 `no_low_clip`

- `lr = 1e-4`
- `clip_epsilon = 0.15`
- `max_grad_norm = 0.30`

狙い:

- `low_clip` を外すと M4 が壊れるかを見る

#### N3 `no_tight_gradclip`

- `lr = 1e-4`
- `clip_epsilon = 0.10`
- `max_grad_norm = 0.50`

狙い:

- `tight_gradclip` を外すと M4 が壊れるかを見る

### B. 単独有効性テスト: M3 に 1 ノブずつ足す

#### S1 `low_lr_only`

- `lr = 1e-4`
- `clip_epsilon = 0.15`
- `max_grad_norm = 0.50`

狙い:

- `low_lr` だけで M3 を救えるかを見る

#### S2 `low_clip_only`

- `lr = 3e-4`
- `clip_epsilon = 0.10`
- `max_grad_norm = 0.50`

狙い:

- `low_clip` だけで M3 を救えるかを見る

#### S3 `tight_gradclip_only`

- `lr = 3e-4`
- `clip_epsilon = 0.15`
- `max_grad_norm = 0.30`

狙い:

- `tight_gradclip` だけで M3 を救えるかを見る

## 8. 実行順

1. N1
2. N2
3. N3
4. S1
5. S2
6. S3

理由:

- まず M4 の stable 条件のどこを外すと壊れるかを見る
- 次に M3 の unstable 条件を、どのノブで単独 rescue できるかを見る

## 9. 判定方法

primary 指標:

1. `ratio_mean`
2. `clip_fraction`
3. `anchor_kl_discard`
4. final `avg_rank`
5. final `win_rate`

secondary 指標:

1. `imitation_eval -> cycle_00 -> final` の変化
2. 後半 5 cycle 平均
3. `policy_loss`

目安:

- stable mixed とみなす最低条件
  - `ratio_mean <= 1.2`
  - `clip_fraction <= 0.35`
  - `anchor_kl_discard <= 0.2`
  - final eval が imitation 直後から大崩れしない

- unstable とみなす条件
  - `ratio_mean` が大きく発散
  - `anchor_kl_discard` が 1 を大きく超える
  - final eval が imitation 直後から大幅悪化

## 10. 期待する読み方

### ケース 1

- N1 だけ崩れる
- S1 だけ安定する

解釈:

- `low_lr` が主因
- 最小有効条件は `M3 + low_lr`

### ケース 2

- N1 / N2 が崩れる
- S1 / S2 は単独では不十分

解釈:

- `low_lr + low_clip` の組が必要

### ケース 3

- N1 / N2 / N3 がどれも大きくは崩れない
- S1 / S2 / S3 も単独では効かない

解釈:

- 3 ノブ相互作用の可能性が高い
- M4 は有効だが最小条件はまだ未確定

## 11. 実行方法

### Driver

```bash
./.venv/bin/python scripts/local/stage2/exp_005_driver.py
```

### 管理ファイル

- `configs/stage2a_core_minimal_mixed_minimality_baseline.yaml`
- `scripts/local/stage2/exp_005_driver.py`
- `experiments/Stage02_CallUnlock/exp_005/run_map.json`

### 実行時間の目安

- 1 run あたり約 45-60 分想定
- 6 本で約 4.5-6 時間想定

夜間にまとめて流す前提でも収まる見込みとする。

## 12. 期待するアウトプット

- `experiments/Stage02_CallUnlock/exp_005/run_map.json`
- report に転記する主要数値
- M3 / M4 / N1 / N2 / N3 / S1 / S2 / S3 の比較表

## 13. 次アクション判定

### 最小有効条件が見つかった場合

- その条件を mixed baseline 候補とする
- 2-3 seed で再確認する
- 必要なら `exp_003` で見た feature 比較条件と合流させる

### 最小有効条件が見つからなかった場合

- M4 は引き続き有効候補として保持する
- ただし次は単純ハイパラではなく、
  - branch 別 mixed 方針
  - baseline sample の扱い
  - discard 専用 anchor / diagnostics
 など、mixed 設計側の見直しを検討する
