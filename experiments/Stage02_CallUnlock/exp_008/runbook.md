# Experiment Runbook: exp_008

作成日: 2026-03-30  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_007/report.md`
- `experiments/Stage02_CallUnlock/exp_006/report.md`
- `experiments/Stage02_CallUnlock/exp_005/report.md`

## 1. 背景

`exp_005` により、Stage02a mixed PPO の最小有効条件は S1 であると分かった。

S1 条件:

- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=1.0`
- `lr=1e-4`
- `clip_epsilon=0.15`
- `max_grad_norm=0.50`

`exp_007` では、S1 から 1 ノブだけを動かした比較を行い、
**`policy_anchor.coef=0.75` の R3 が最も有望**という結果になった。

R3 の要点:

- stable に 20 cycle 完走
- imitation 直後より final eval が改善
- `ratio_mean`, `clip_fraction`, `anchor_kl_discard` も健全域に留まった

さらに、R3 source の branch-swap eval では、
**discard 側の改善はかなり見える一方、optional 側の改善はまだ弱い**ことが分かった。

ここまでで、

- mixed PPO は stable に回る
- discard 側では少なくとも改善の気配がある

ところまでは確認できた。

次に必要なのは、
**この改善が 20 cycle 限定の偶然ではなく、より長い PPO と複数 seed でも再現するか**
を確認することである。

## 2. 問い

R3 条件を 50 cycle まで延長し、seed を 3 本に増やしたとき、

1. mixed PPO は最後まで stable に回るか
2. imitation 直後より final eval の改善が再現するか
3. その改善は discard 改善として一貫して出ると期待できるか
4. 今後の Stage02 baseline として R3 を採用してよいか

## 3. 基本方針

この実験では、**R3 をそのまま長く回して再現性を見る**。

今回は新しいノブ探索はしない。
目的は頂点性能ではなく、

- `20 cycle` で見えた改善が
- `50 cycle` でも崩れず
- `3 seeds` でも概ね再現するか

を確認することである。

## 4. Baseline

基準条件は R3 とする。

R3 条件:

- mixed PPO
- A `core_minimal`
- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=0.75`
- `lr=1e-4`
- `clip_epsilon=0.15`
- `max_grad_norm=0.50`

参照 run:

- `exp_007` の `R3_lower_anchor_075`（対応は `experiments/Stage02_CallUnlock/exp_007/run_map.json` を参照）

20 cycle 時点の参考:

- imitation 直後: `avg_rank=2.315`, `win_rate=0.2394`
- final: `avg_rank=2.235`, `win_rate=0.2523`
- final PPO diagnostics:
  - `ratio_mean=1.0163`
  - `clip_fraction=0.2265`
  - `anchor_kl_discard=0.0223`

## 5. 今回の比較条件

今回は hyperparameter sweep ではなく、**同一条件を 3 seeds** 回す。

### Seed A

- `experiment.global_seed=42`

### Seed B

- `experiment.global_seed=43`

### Seed C

- `experiment.global_seed=44`

## 6. 固定条件

全 seed 共通:

- A `core_minimal`
- `training.rule_mix_learner.ppo_mode = "mixed"`
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.policy_anchor.coef = 0.75`
- `training.lr = 1e-4`
- `training.clip_epsilon = 0.15`
- `training.max_grad_norm = 0.50`
- `training.multi_cycle.num_cycles = 50`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`

評価条件は、これまでの Stage02 実験と同じく rotation eval を使う。

## 7. 目的に対する評価観点

### 7.1 安定性

- 50 cycle 完走できるか
- learner loss が終盤で吹き上がらないか
- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- eval が後半で崩壊しないか

### 7.2 改善の有無

- imitation 直後 eval と final eval の差
- `cycle_00` から `cycle_49` までの推移
- `avg_rank`
- `win_rate`
- `deal_in_rate`
- 可能なら tail-10 average

### 7.3 再現性

- 3 seeds のうち何本で `final > imitation_eval` が成立するか
- 3 seeds のうち何本で `ratio_mean` と `anchor_kl_discard` が正常域を維持するか
- 1 seed の偶然ではなく、方向性が揃っているか

## 8. 成功判定

この実験での成功は、少なくとも以下を満たすこととする。

1. 3 seeds すべて、または少なくとも 2/3 seeds で 50 cycle stable に完走する
2. 3 seeds のうち少なくとも 2 本で、`final eval` が `imitation_eval` より明確に良い
3. `ratio_mean` が大きく崩れず、`anchor_kl_discard` も健全域に留まる
4. 後半 cycle で eval が崩れず、改善が保持される

ここでの主目的は、

- 「最高性能の追求」ではなく
- **改善が長時間・複数 seed で安定して出る regime の確認**

である。

## 9. 読み方

### ケース 1: 3 seeds とも stable、かつ final 改善が再現

解釈:

- R3 は Stage02 mixed baseline としてかなり強い
- この条件を持って partial / ルール拡張へ進む根拠として十分

### ケース 2: stable だが改善は seed によって揺れる

解釈:

- stable mixed はできている
- ただし改善量はまだ小さく、ノイズに近い
- ここで頂点探索はせず、より realistic なルール帯へ進んでから再評価するのが自然

### ケース 3: 後半 cycle で again drift する

解釈:

- R3 は 20 cycle では良かったが、50 cycle ではまだ弱い
- `anchor` か `policy_ratio` の再調整、または stopping rule の検討が必要

### ケース 4: seed ごとの branch 気配がかなり違う

解釈:

- discard 改善の気配はあるが、reproducibility がまだ弱い
- 完全麻雀へ進む前に、branch attribution をもう少しだけ確認する価値がある

## 10. 実装方針

この runbook では、新しい学習実装は前提にしない。

既存の S1 baseline config を土台に、override で R3 条件へ寄せ、
さらに `num_cycles=50` と `seed` だけを変えて実行する。

想定管理ファイル:

- `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `scripts/local/stage2/exp_008_driver.py`
- `experiments/Stage02_CallUnlock/exp_008/run_map.json`

想定 override 差分:

- `training.policy_anchor.coef=0.75`
- `training.multi_cycle.num_cycles=50`
- `experiment.global_seed in {42, 43, 44}`

## 11. 実行方針

想定実行は 3 本連続 driver とする。

想定ラベル:

- `R3_seed42_mc50`
- `R3_seed43_mc50`
- `R3_seed44_mc50`

想定所要時間:

- 1 run あたり約 2〜3 時間
- 3 run 合計で約 6〜9 時間

長時間実験なので、夜間バッチ向きである。

## 12. 次アクション判定

### 良い結果だった場合

- R3 を Stage02 mixed baseline として採用する
- 次は partial observation またはルール拡張へ進む
- optional branch の改善不足は、完全麻雀側で再評価する

### stable だが改善が弱い場合

- R3 は safe baseline として保持する
- ただし「改善が安定して強い」とはまだ言わない
- その状態で realism 側を進める

### again unstable だった場合

- R3 を baseline 候補から外す
- S1 に戻すか、anchor を再度強める
- 長時間 mixed への適性を別途再検討する
