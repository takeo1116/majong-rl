# Experiment Report: exp_015

作成日: 2026-04-09  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_015/runbook.md`
- `experiments/Stage02_CallUnlock/exp_014/report.md`
- `experiments/Stage02_CallUnlock/exp_015/semantic_eval_A2_semaux_light_vhalf_tenpaifix_prnorm_cycle19/semantic_eval_imitation_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_015/semantic_eval_A2_semaux_light_vhalf_tenpaifix_prnorm_cycle19/semantic_eval_final_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_015/semantic_eval_A1_semaux_default_vhalf_tenpaifix_prnorm_cycle19/semantic_eval_imitation_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_015/semantic_eval_A1_semaux_default_vhalf_tenpaifix_prnorm_cycle19/semantic_eval_cycle10_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_015/semantic_eval_A1_semaux_default_vhalf_tenpaifix_prnorm_cycle19/semantic_eval_final_cycle19_summary.md`

## 1. 要約

`exp_015` は、`exp_014` の `tenpaifix` 条件を維持したまま、`CQ-0268` の **terminal semantic loss の player-round 正規化** を加えた再実験である。

今回の結論は次の通り。

- **`CQ-0268` は当たりだった**
- 現在の practical baseline は **`A2_semaux_light_vhalf_tenpaifix_prnorm`** でよい
- `exp_014` で見えた
  - `draw_tenpai` は上がる
  - その代わり `win_called / win_menzen` が崩れる
  という trade-off は、今回かなり緩和された
- policy でも `A2` はさらに改善した
- `A1` も retain は改善したが、最終的なバランスは依然として `A2` の方が良い
- ただし **`deal_in` は依然として top-1 class としてはほぼ立っていない**

したがって `exp_015` は、

- terminal duplicated-label bias の補正が有効であること
- terminal 5-class の学習安定化に、player-round 正規化が効くこと
- 今後の基準は `A2 + tenpaifix + prnorm` でよいこと

を確認した実験と整理できる。

## 2. 背景

`exp_014` までで、次の構造整理は反映済みだった。

- `CQ-0265`
  - `shanten_hint` / `discard_ukeire_hint` を shared trunk から外し、discard direct hint に戻した
- `CQ-0266`
  - terminal semantic を 5-class に整理した
  - `win_menzen / win_called / draw_tenpai / deal_in / other_non_dealin`
- `CQ-0267`
  - `self_tenpai_flag` / `remaining_draws_norm` を追加した

その結果 `exp_014` では、

- policy 性能は改善した
- `draw_tenpai` は狙いどおり改善した
- しかし `win_called / win_menzen` が大きく悪化した

という、terminal class 間の綱引きがかなり強く出た。

この原因として、

- 同じ `player-round` の全 decision に同じ terminal label を貼っている
- その結果、長い局ほど terminal loss の総量が大きくなる
- row ベースの empirical 分布が歪み、class 間のバランスを崩している

という仮説を立て、`CQ-0268` を実装した。

## 3. 今回の修正 (`CQ-0268`)

`CQ-0268` では、**terminal semantic loss のみ**に対して player-round 正規化を導入した。

- group key: `episode_id / round_id / player_id`
- 同じ group に属する row 数を `n` としたとき、各 row の terminal weight は `1/n`
- 同じ player-round に属する terminal loss の総量を `1.0` にそろえる

重要:

- 今回は **正規化のみ**
- 巡目による progress weighting は入れていない
- `yaku` / `policy` / `value` / reward / GAE は据え置き

狙いは、

- `draw_tenpai` を完全には失わず
- `win_called / win_menzen` を戻し
- terminal class 間の学習を穏やかにする

ことである。

## 4. 条件

比較条件:

### A2: semantic aux light + vhalf + tenpaifix + prnorm

- `terminal_loss_coef = 0.1`
- `yaku_loss_coef = 0.05`
- `value_loss_coef = 0.125`
- terminal semantic loss に player-round 正規化あり

### A1: semantic aux default + vhalf + tenpaifix + prnorm

- `terminal_loss_coef = 0.2`
- `yaku_loss_coef = 0.1`
- `value_loss_coef = 0.125`
- terminal semantic loss に player-round 正規化あり

固定:

- direct hint branch (`CQ-0265`)
- terminal 5-class (`CQ-0266`)
- `self_tenpai_flag` / `remaining_draws_norm` (`CQ-0267`)
- seed `42`
- `num_cycles = 20`

## 5. 主結果

### 5.1 imitation と final

#### A2

- imitation: `avg_rank=2.355`, `win_rate=0.2476`, `deal_in_rate=0.1838`
- final: `avg_rank=2.345`, `win_rate=0.2540`, `deal_in_rate=0.1781`

#### A1

- imitation: `avg_rank=2.410`, `win_rate=0.2350`, `deal_in_rate=0.1846`
- final: `avg_rank=2.415`, `win_rate=0.2505`, `deal_in_rate=0.1934`

読み:

- `A2` は imitation より final がさらに良く、今回の実験で最良
- `A1` も `win_rate` は伸びているが、`avg_rank` は imitation よりわずかに悪化
- policy 面の practical baseline は明確に `A2`

### 5.2 `exp_014` との比較

`A2` final:

- `avg_rank`: `2.385 -> 2.345`
- `win_rate`: `0.2508 -> 0.2540`
- `deal_in_rate`: `0.1876 -> 0.1781`

読み:

- `CQ-0268` は policy の観点でも素直に前進
- 少なくとも「terminal 正規化で policy が壊れる」方向には出ていない

### 5.3 cycle の形

driver log ベースの概観:

#### A2

- imitation: `avg_rank ≈ 2.36`
- best cycle: `cycle 0`, `avg_rank ≈ 2.29`
- final: `avg_rank ≈ 2.42`
- tail-5 average: `≈ 2.438`

#### A1

- imitation: `avg_rank ≈ 2.41`
- best cycle: `cycle 0 / 10` 近辺, `avg_rank ≈ 2.33`
- final: `avg_rank ≈ 2.42`
- tail-5 average: `≈ 2.450`

読み:

- `A1` は `exp_013` までと比べると retain がかなり改善している
- ただし最終的な安定感と水準は `A2` の方が良い
- `A2` を今後の標準条件にしてよい

### 5.4 PPO 安定性

#### A2

- `ratio_mean = 0.9921`
- `clip_fraction = 0.0789`
- `anchor_kl_discard = 0.00524`

#### A1

- `ratio_mean = 1.0025`
- `clip_fraction = 0.0825`
- `anchor_kl_discard = 0.00566`

読み:

- どちらも PPO 指標としては健全
- 今回の差は不安定化ではなく、terminal 学習のバランス改善として解釈してよい

## 6. semantic diagnostics

評価 shard:

- `cycle_19/selfplay`

対象 checkpoint:

### A2

- `checkpoint_imitation.pt`
- `checkpoint_learner.pt`

### A1

- `checkpoint_imitation.pt`
- `checkpoint_cycle_10.pt`
- `checkpoint_learner.pt`

## 7. terminal の変化

### 7.1 A2: `exp_014` の trade-off はかなり改善した

`exp_014 A2 final` → `exp_015 A2 final`

- `win_menzen`
  - recall/top1: `0.0025 -> 0.0884`
  - `mean_p: 0.0703 -> 0.1452`
- `win_called`
  - recall/top1: `0.0004 -> 0.0779`
  - `mean_p: 0.1214 -> 0.1759`
- `draw_tenpai`
  - recall/top1: `0.0382 -> 0.0166`
  - `mean_p: 0.0694 -> 0.0594`
- `deal_in`
  - recall/top1: `0.0000 -> 0.0000`
  - `mean_p: 0.2119 -> 0.1766`

読み:

- `draw_tenpai` はやや後退した
- しかしその代わりに `win_called / win_menzen` がかなり戻った
- `exp_014` の「ある class を立てると他が死ぬ」崩れ方が大きく緩和された

これは `CQ-0268` の狙いにかなり合致している。

### 7.2 A2: imitation から final で 3 class とも改善している

A2 imitation → final:

- `win_menzen`: recall `0.0197 -> 0.0884`
- `win_called`: recall `0.0702 -> 0.0779`
- `draw_tenpai`: recall `0.0013 -> 0.0166`

読み:

- 以前のような terminal collapse ではなく、**複数 class が一応同時に伸びる**形になっている
- 特に `draw_tenpai` を完全には失っていないのが重要

### 7.3 A2 final の terminal 位置づけ

A2 final:

- `win_menzen`
  - precision `0.3820`
  - recall `0.0884`
  - `mean_p = 0.1452`
- `win_called`
  - precision `0.3815`
  - recall `0.0779`
  - `mean_p = 0.1759`
- `draw_tenpai`
  - precision `0.3827`
  - recall `0.0166`
  - `mean_p = 0.0594`
- `deal_in`
  - precision `0.0000`
  - recall `0.0000`
  - `mean_p = 0.1766`
- `other_non_dealin`
  - precision `0.6620`
  - recall `0.9832`
  - `mean_p = 0.6816`

読み:

- 依然として `other_non_dealin` 優勢ではある
- ただし `win_called / win_menzen` は以前より明らかに健全
- `draw_tenpai` は弱いがゼロではない
- `deal_in` はまだ top-1 class としては成立していない

### 7.4 A1: retain は改善したが、final のバランスは A2 に劣る

A1 imitation → cycle10 → final:

- `win_menzen` recall: `0.0712 -> 0.1321 -> 0.1303`
- `win_called` recall: `0.0761 -> 0.0837 -> 0.0579`
- `draw_tenpai` recall: `0.0000 -> 0.0153 -> 0.0027`

読み:

- `cycle10` はかなり良い
- final での崩れは以前より小さい
- ただし `draw_tenpai` は final でかなり弱い
- `win_called` も final では A2 に劣る

したがって A1 は「完全に捨てるほど悪くはない」が、現時点で基準にする理由は薄い。

### 7.5 `deal_in` は引き続き別評価が必要

A2 / A1 ともに:

- `deal_in recall = 0.0`
- `deal_in top1_hit = 0.0`
- ただし `mean_p(deal_in)` 自体は一定量ある
  - A2 final: `0.1766`
  - A1 final: `0.1628`

ここは `win_called` と同じ指標では見ない方が自然である。

`deal_in` は

- 最頻終局 class というより
- 危険度の確率信号

として見るべきで、今後は top-1 ではなく

- `mean_p(deal_in | y=deal_in)`
- `mean_p(deal_in | y!=deal_in)`
- `PR-AUC`
- 危険局面 subset での上昇

のような診断を追加するのが妥当である。

## 8. yaku

### A2 final

- micro F1 `0.5007`
- macro F1 `0.0965`
- exact match `0.4339`

### A1 final

- micro F1 `0.5947`
- macro F1 `0.1318`
- exact match `0.4508`

読み:

- yaku は依然として `A1` が強い
- ただし policy / terminal の総合バランスでは `A2` が優勢
- したがって現段階では「yaku 単体最適」より「policy + terminal の実用基準」を優先して `A2` を採る方が良い

### 8.1 last-3 winner decision 集計

役ごとの難易度差を切り分けるため、`exp_015` final checkpoint について
**winning player-round の最後の 3 decision のみ**に絞った one-off 集計を行った。

出力:

- `experiments/Stage02_CallUnlock/exp_015/yaku_last3_eval/yaku_last3_eval_summary.md`
- `experiments/Stage02_CallUnlock/exp_015/yaku_last3_eval/yaku_last3_eval.json`

注意:

- ここでいう last-3 は「最後の 3 hand-turn」ではなく、**最後の 3 decision**
- `discard` と `call` の両 branch を含む

#### A2 final

- overall winner-only
  - micro F1 `0.5007`
  - macro F1 `0.0965`
  - exact match `0.4339`
- last-3 winner decisions only
  - micro F1 `0.5367`
  - macro F1 `0.1067`
  - exact match `0.4348`

役ごとの recall 変化:

- `Riichi`: `0.3918 -> 0.5475`
- `Yakuhai`: `0.9121 -> 0.9459`
- `Tanyao`: `0.0 -> 0.0`
- `MenzenTsumo`: `0.0 -> 0.0`
- `Ippatsu`: `0.0 -> 0.0`
- `Pinfu`: `0.0 -> 0.0`
- `Toitoi`: `0.0 -> 0.0`

#### A1 final

- overall winner-only
  - micro F1 `0.5947`
  - macro F1 `0.1318`
  - exact match `0.4508`
- last-3 winner decisions only
  - micro F1 `0.6399`
  - macro F1 `0.1648`
  - exact match `0.4826`

役ごとの recall 変化:

- `Riichi`: `0.8113 -> 0.8662`
- `Yakuhai`: `0.7603 -> 0.8698`
- `Ippatsu`: `0.0047 -> 0.0160`
- `Pinfu`: `0.0010 -> 0.0034`
- `Tanyao`: `0.0 -> 0.0`
- `MenzenTsumo`: `0.0 -> 0.0`
- `Toitoi`: `0.0 -> 0.0`

読み:

- last-3 に絞ると `Riichi` / `Yakuhai` はさらに良くなる
- しかし `Tanyao` / `MenzenTsumo` / `Pinfu` / `Toitoi` は、**最後の 3 decision に限ってもほぼ立たない**
- したがって、yaku 評価の unfairness だけでは説明できず、
  **late-structural な役自体がまだほとんど学習できていない**
  と見てよい

## 9. 結論

`exp_015` の結論は次の通り。

1. `CQ-0268` は有効だった
- terminal duplicated-label bias の補正として、player-round 正規化はかなり筋が良い

2. `exp_014` の trade-off はかなり改善した
- `draw_tenpai` を完全には失わず
- `win_called / win_menzen` を戻せた

3. 現時点の新しい practical baseline は `A2_semaux_light_vhalf_tenpaifix_prnorm`
- policy が最良
- terminal も最もバランスが良い

4. `A1` は retain 改善が見えたが、基準条件としてはまだ A2 に劣る

5. yaku については、`Riichi` / `Yakuhai` は見えているが、
   `Tanyao` / `Pinfu` / `MenzenTsumo` などの late-structural な役は終盤でも依然として弱い

6. 次に見るべき本命は `deal_in`
- ただし `deal_in` は top-1 class としてではなく、risk signal として評価する方が自然

## 10. 次の手

次は次の順で進めるのが自然である。

1. `deal_in` 用の risk-oriented diagnostics を追加する
- `mean_p_pos / mean_p_neg`
- `p50 / p90`
- `roc_auc / pr_auc`
- `late_and_noten`
- `early_and_tenpai`

2. `A2_semaux_light_vhalf_tenpaifix_prnorm` を当面の基準条件とする

3. `deal_in` を top-1 class として押し上げるべきかは、追加診断を見てから判断する

現時点では、無理に `deal_in` を top-1 にしようとするより、
**危険局面で `p(deal_in)` が適切に上がっているか** を先に見るべきである。
