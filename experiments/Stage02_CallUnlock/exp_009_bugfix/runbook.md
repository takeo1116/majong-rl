# Experiment Runbook: exp_009

作成日: 2026-03-31  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_008/report.md`
- `experiments/Stage02_CallUnlock/exp_007/report.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

## 1. 背景

`exp_008` により、R3 条件

- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=0.75`
- `lr=1e-4`
- `clip_epsilon=0.15`
- `max_grad_norm=0.50`

は、50 cycle・3 seeds でも stable に回ることが確認できた。

一方で、改善量は大きくなかった。

- imitation 直後より final が少し良い seed はある
- ただし後半で改善が積み上がる感じは弱い
- branch-swap でも discard 改善は見えたが、optional 側はまだ弱い

ここから読むと、次に触るべきなのは PPO サイクル数ではなく、**policy が判断に使える semantic signal を増やすこと**である。

そのため `CQ-0256` で、

- `value_head`
- `terminal_head`
- `yaku_head`

を持つ semantic auxiliary trunk が実装された。  
今回の実験では、これが実際に学習改善へつながるかを最初に見る。

## 2. 問い

今回の runbook で答えたい問いは次の 3 つである。

1. semantic auxiliary trunk を有効にすると、R3 と同程度の安定性を保てるか
2. semantic auxiliary trunk により、`imitation_eval -> final eval` の改善量は大きくなるか
3. aux loss の強さが結果に影響する場合、default 係数と lighter 係数のどちらが良いか

## 3. 基本方針

今回は新しい reward shaping や新しい PPO ハイパラ探索は行わない。

比較するのは、

- 現行の R3 control
- semantic aux default
- semantic aux light

の 3 条件だけとし、差分を **semantic auxiliary trunk の有無と loss 係数**に限定する。

また、最初の実験なので seed はまず `42` の 1 本に限定する。  
ここで promising な傾向が出た条件だけを、後続で multi-seed 化する。

## 4. 比較条件

### C0: R3 control

- semantic aux 無効
- 現行 mixed baseline を、現在のコードベース上で再確認するための control

### A1: semantic aux default

- `model.semantic_aux.enabled = true`
- `training.semantic_aux.enabled = true`
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.2`
- `training.semantic_aux.yaku_loss_coef = 0.1`

### A2: semantic aux light

- semantic aux は有効
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`

## 5. 共通固定条件

全条件共通:

- A `core_minimal`
- `full observation`
- `training.rule_mix_learner.ppo_mode = "mixed"`
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.policy_anchor.coef = 0.75`
- `training.lr = 1e-4`
- `training.clip_epsilon = 0.15`
- `training.max_grad_norm = 0.50`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`
- `experiment.global_seed = 42`

意図:

- `exp_008` までに stable と確認できた R3 を土台にする
- semantic aux の効果以外の差分を増やさない
- まずは 20 cycle で、semantic signal が early に効くかを見る

## 6. 実験の読み方

### 6.1 安定性

以下が R3 と同程度に健全かを見る。

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- learner loss の暴れ方
- 20 cycle 完走可否

### 6.2 semantic 学習

semantic aux 条件では、少なくとも以下を追う。

- imitation の `terminal_loss`
- imitation の `yaku_loss`
- PPO の `terminal_loss`
- PPO の `yaku_loss`

見たいこと:

- finite であること
- imitation で下がること
- PPO 中も極端に悪化しないこと

### 6.3 性能

主に次を比較する。

- `imitation_eval`
- `cycle_00`
- `final`
- 可能なら `tail-5 average`

特に見る差分:

- `final - imitation_eval`
- `cycle_00 - imitation_eval`

## 7. 成功判定

今回の first experiment の成功は、少なくとも次を満たすこととする。

1. `A1` または `A2` が `C0` と同程度に stable
2. semantic aux loss が学習中に有限で、少なくとも imitation で低下傾向を示す
3. `A1` または `A2` の `final eval` が、`C0` よりも `imitation_eval` からの改善量で上回る

ここでの主目的は、

- semantic trunk が学べること
- その情報が policy 改善に使われる兆しがあること

を確認することである。

## 8. 典型的な解釈

### ケース 1: A1 が stable で改善も大きい

解釈:

- semantic auxiliary は有望
- 次は A1 を 3 seeds に広げる
- その後 branch-swap で discard / optional の寄与を再確認する

### ケース 2: A1 は微妙だが A2 が良い

解釈:

- semantic の方向は正しい
- ただし aux loss が強すぎる
- lighter 係数を baseline 候補にする

### ケース 3: loss は下がるが性能差がない

解釈:

- semantic target 自体は学べている
- しかし policy 利用がまだ弱い
- 次は summary の使い方、detach、あるいは branch 別診断を見直す

### ケース 4: semantic 条件で不安定化する

解釈:

- summary の注入や aux 重みが強すぎる
- より軽い係数、あるいは projection 次元の縮小が必要

## 9. 実装・実行方針

ベース config は R3 系の mixed baseline を使い、override で semantic aux 条件を切る。

想定管理ファイル:

- `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `experiments/Stage02_CallUnlock/exp_009/run_map.json`
- `experiments/Stage02_CallUnlock/exp_009/report.md`

今回は driver 実行を前提とする。

想定ラベル:

- `C0_r3_control`
- `A1_semaux_default`
- `A2_semaux_light`

## 10. 実行方式

想定実行は 3 条件連続 driver とする。

想定所要時間:

- 1 run あたり約 1〜2 時間
- 3 run 合計で約 3〜6 時間

長すぎる overnight 実験ではなく、まずは日中に回して結果を見られるサイズにする。

実行コマンドは driver 作成後に追記する。

実行コマンド:

```bash
./.venv/bin/python scripts/local/stage2/exp_009_driver.py
```

1 条件だけ試す場合:

```bash
EXP009_ONLY=A1_semaux_default ./.venv/bin/python scripts/local/stage2/exp_009_driver.py
```

## 11. 今回やらないこと

- branch-swap
- multi-seed
- reward shaping
- han / fu auxiliary
- non-detach summary
- partial observation

## 12. 次アクション判定

### 良い結果だった場合

- 良かった条件を `3 seeds` に広げる
- その後 branch-swap で optional 改善が立つかを見る

### 中立だった場合

- semantic loss は下がるかを確認する
- 下がるなら policy 利用の問題として次段を考える

### 悪かった場合

- aux loss 係数か projection 次元を軽くする
- それでも駄目なら reward 側の再検討に戻る
