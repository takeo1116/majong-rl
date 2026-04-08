# Experiment Report: exp_009

作成日: 2026-03-31  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_009/runbook.md`
- `experiments/Stage02_CallUnlock/exp_008/report.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

## 1. 目的

この実験の目的は、`CQ-0256` で実装した semantic auxiliary trunk が、

- R3 と同程度の安定性を保てるか
- `imitation_eval -> final eval` の改善量を大きくできるか
- aux loss の係数感度が強いか

を最初に確認することである。

比較した条件は次の 3 つ。

- `C0_r3_control`
- `A1_semaux_default`
- `A2_semaux_light`

## 2. 条件

共通固定:

- A `core_minimal`
- `full observation`
- mixed PPO
- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=0.75`
- `lr=1e-4`
- `clip_epsilon=0.15`
- `max_grad_norm=0.50`
- `num_cycles=20`
- `imitation_eval.enabled=true`
- `eval_each_cycle=true`
- seed `42`

差分:

### C0: R3 control

- `model.semantic_aux.enabled=false`
- `training.semantic_aux.enabled=false`

### A1: semantic aux default

- `model.semantic_aux.enabled=true`
- `training.semantic_aux.enabled=true`
- `policy_projection_dim=16`
- `terminal_loss_coef=0.2`
- `yaku_loss_coef=0.1`

### A2: semantic aux light

- `model.semantic_aux.enabled=true`
- `training.semantic_aux.enabled=true`
- `policy_projection_dim=16`
- `terminal_loss_coef=0.1`
- `yaku_loss_coef=0.05`

## 3. 結果

### 3.1 imitation 直後と final

#### C0

- imitation: `avg_rank=2.315`, `win_rate=0.2394`
- final: `avg_rank=2.245`, `win_rate=0.2529`, `deal_in_rate=0.1699`

#### A1

- imitation: `avg_rank=2.495`, `win_rate=0.2341`
- final: `avg_rank=2.295`, `win_rate=0.2544`, `deal_in_rate=0.1554`

#### A2

- imitation: `avg_rank=2.400`, `win_rate=0.2316`
- final: `avg_rank=2.320`, `win_rate=0.2473`, `deal_in_rate=0.1687`

### 3.2 PPO 安定性

#### C0

- `ratio_mean=1.0328`
- `clip_fraction=0.2536`
- `anchor_kl_discard=0.0230`

#### A1

- `ratio_mean=1.0078`
- `clip_fraction=0.2405`
- `anchor_kl_discard=0.0206`

#### A2

- `ratio_mean=1.1178`
- `clip_fraction=0.2737`
- `anchor_kl_discard=0.0483`

### 3.3 補助観測

best cycle:

- `C0`: cycle 19, `avg_rank=2.245`, `win_rate=0.2529`
- `A1`: cycle 19, `avg_rank=2.295`, `win_rate=0.2544`
- `A2`: cycle 9, `avg_rank=2.265`, `win_rate=0.2335`

tail-5 average:

- `C0`: `avg_rank=2.401`, `win_rate=0.2364`, `deal_in_rate=0.1713`
- `A1`: `avg_rank=2.449`, `win_rate=0.2390`, `deal_in_rate=0.1714`
- `A2`: `avg_rank=2.453`, `win_rate=0.2304`, `deal_in_rate=0.1740`

semantic aux loss:

- `A1` cycle 0: `terminal_loss=1.6144`, `yaku_loss=0.2394`
- `A1` cycle 19: `terminal_loss=1.6149`, `yaku_loss=0.2317`
- `A2` cycle 0: `terminal_loss=1.6208`, `yaku_loss=0.2416`
- `A2` cycle 19: `terminal_loss=1.6087`, `yaku_loss=0.2391`

## 4. 読み取り

### 4.1 semantic aux を入れても stable

`A1` は control と同程度、むしろややきれいな PPO diagnostics を示した。

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`

はいずれも健全域にあり、semantic auxiliary の追加で mixed PPO が壊れる様子は見えなかった。

`A2` も完走はしたが、`A1` と比べると PPO 指標は一段悪い。

### 4.2 default の方が light より良い

`A2` は

- final performance
- PPO stability

の両方で `A1` を下回った。

少なくともこの 1 seed では、semantic aux の loss を軽くした方が良い、という evidence は出ていない。

### 4.3 A1 は imitation を即改善するというより、PPO で活きる可能性がある

`A1` は imitation 直後の値だけ見ると `C0` より悪い。

しかし final では

- `win_rate` は `C0` と同等以上
- `deal_in_rate` は明確に良い

まで戻している。

この挙動は、semantic auxiliary が

- imitation 教師の即時模倣性能を直接押し上げる

というより、

- PPO 中の表現学習や policy 更新に効いている

可能性を示唆する。

### 4.4 ただし、現時点では改善はまだ小さい

`A1` は promising だが、`C0` を全面的に圧倒したわけではない。

特に tail-5 average では優位は明確でなく、

- final 1 点は良い
- しかし後半全体で大きく積み上がるとはまだ言いにくい

という段階である。

## 5. 結論

今回の 1 seed・3条件比較では、

- **本命は `A1_semaux_default`**
- `A2_semaux_light` は採用理由が弱い
- semantic auxiliary trunk の方向は、少なくとも捨てなくてよい

という結論になった。

一方で、今の結果だけでは

- semantic trunk 自体が本当に meaningful な terminal / yaku 予測を学習できているか

はまだ直接確認できていない。

そのため次にやるべきことは、

1. semantic head の診断を追加する
2. そのうえで `A1` を multi-seed 化する

の順が自然である。

## 6. 次アクション

1. semantic trunk の直接診断用 CQ を切る
2. terminal / yaku の予測品質を checkpoint ごとに測る
3. その結果が良ければ `A1` を 3 seeds に広げる
