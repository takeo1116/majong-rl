# Experiment Runbook: exp_007

作成日: 2026-03-30  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_006/report.md`
- `experiments/Stage02_CallUnlock/exp_005/report.md`
- `experiments/Stage02_CallUnlock/exp_004/report.md`

## 1. 背景

`exp_005` により、Stage02a mixed PPO の最小有効条件は S1 であると分かった。

S1 条件:

- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=1.0`
- `lr=1e-4`
- `clip_epsilon=0.15`
- `max_grad_norm=0.50`

`exp_006` では、この S1 run に対して branch-swap eval を実施した。
結果として、mixed PPO 自体は stable である一方、
500 match rotation eval でも discard / optional の branch 単位改善はかなり小さく、
明確には分離できなかった。

したがって次の課題は、

- 頂点性能を探すことではなく
- **安定を壊さずに PPO の改善量を少し強めること**

になる。

## 2. 問い

S1 を baseline としたとき、

1. PPO の update を少し強めても stable を維持できるか
2. imitation 直後より final eval がより明確に改善する条件はあるか
3. その改善は branch-swap へ進む価値がある程度には大きいか

## 3. 基本方針

この実験では、**S1 から 1 ノブだけを動かす**。

理由:

- `exp_005` で `low_lr` が主因と分かったため、まずはそこを中心に少しだけ戻すのが筋が良い
- `exp_006` で branch 改善が弱かったため、まず end-to-end で改善量を大きくできるかを見るべき
- 同時に複数ノブを動かすと、何が効いたか分からなくなる

## 4. Baseline

基準条件は S1 とする。

- mixed PPO
- A `core_minimal`
- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=1.0`
- `lr=1e-4`
- `clip_epsilon=0.15`
- `max_grad_norm=0.50`

基準 run:

- `exp_005` の `S1_low_lr_only`（対応は `experiments/Stage02_CallUnlock/exp_005/run_map.json` を参照）

参考:

- S1 imitation 直後 eval: `avg_rank=2.315`, `win_rate=0.2394`
- S1 final eval: `avg_rank=2.325`, `win_rate=0.2568`

## 5. 比較条件

### R1 `mid_lr_1p5e4`

S1 から `lr` だけ少し上げる。

- `lr=1.5e-4`

意図:

- `3e-4` は強すぎた
- `1e-4` は安定だが、効きが弱い可能性がある
- 中間値で改善量が増えるかを見る

### R2 `mid_lr_2e4`

S1 から `lr` をもう少し上げる。

- `lr=2.0e-4`

意図:

- `1.5e-4` よりさらに強めたとき、まだ stable を保てるかを見る
- `3e-4` に戻す前に、安定域の上限を探る

### R3 `lower_anchor_075`

S1 から anchor だけ少し弱める。

- `policy_anchor.coef=0.75`

意図:

- 現在 imitation への拘束が強すぎて policy が動きにくい可能性を確認する
- `lr` 側だけではなく、anchor 側にも改善余地があるかを見る

## 6. 固定条件

全条件で共通:

- A `core_minimal`
- `training.rule_mix_learner.ppo_mode = "mixed"`
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.clip_epsilon = 0.15`
- `training.max_grad_norm = 0.50`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`

## 7. 評価観点

### 安定性

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- learner loss の終盤挙動
- hard collapse / NaN / eval 崩壊の有無

### 改善量

- imitation 直後 eval と final eval の差
- tail-5 eval の安定性
- `avg_rank`
- `win_rate`
- `deal_in_rate`

## 8. 成功判定

この実験での成功は、少なくとも以下を満たす条件を 1 つ見つけることである。

1. 20 cycle stable に完走する
2. `ratio_mean` が概ね正常域に留まる
3. `anchor_kl_discard` が大きく発散しない
4. imitation 直後より final eval が、S1 よりも明確に良い

ここでの主目的は、

- 「最高性能」ではなく
- **改善が出る stable mixed regime を作ること**

である。

## 9. 読み方

### ケース 1: R1 が成功、R2 は崩れる

解釈:

- `lr=1e-4` はやや保守的
- しかし上げすぎると again unstable
- `1.5e-4` 付近が良い妥協点の可能性が高い

### ケース 2: R1 / R2 とも成功し、改善量も増える

解釈:

- mixed PPO の stable region は S1 より広い
- 次は `2e-4` 近辺でさらに branch 改善を確認する価値がある

### ケース 3: R1 / R2 は悪化、R3 だけ改善

解釈:

- update size より anchor が効きのボトルネックだった可能性がある

### ケース 4: どれも S1 と大差ない

解釈:

- この簡易ルール帯では PPO 改善量自体が小さい
- 完全麻雀拡張へ進みつつ、より realistic 条件で再評価するのが自然

## 10. 実装方針

この runbook では、新しい実装は前提にしない。

既存の mixed baseline config を土台に、override で 1 ノブずつ変えて比較する。

想定管理ファイル:

- `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `scripts/local/stage2/exp_007_driver.py`
- `experiments/Stage02_CallUnlock/exp_007/run_map.json`

実行コマンド:

```bash
./.venv/bin/python scripts/local/stage2/exp_007_driver.py
```

1 条件だけ試す場合:

```bash
EXP007_ONLY=R1 ./.venv/bin/python scripts/local/stage2/exp_007_driver.py
```

想定所要時間:

- 1 run あたり約 45〜60 分
- 3 run 合計で約 2.5〜3.5 時間

## 11. 次アクション判定

### 改善量が明確に出る stable 条件が見つかった場合

- その条件を新しい mixed baseline 候補にする
- その条件で branch-swap eval を再実施する
- 問題なければ partial / ルール拡張へ進む

### stable だが改善量が増えない場合

- この簡易ルール帯では mixed PPO の伸びしろが小さい可能性を受け入れる
- 性能の頂点は追わず、完全麻雀実装へ進む

### again unstable になる場合

- S1 を baseline のまま維持する
- 以後の mixed 実験は S1 を安全条件として扱う
