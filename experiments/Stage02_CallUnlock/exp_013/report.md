# Experiment Report: exp_013

作成日: 2026-04-03  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_013/runbook.md`
- `experiments/Stage02_CallUnlock/exp_012/report.md`
- `experiments/Stage02_CallUnlock/exp_013/semantic_eval_a2_cycle19_confidence/semantic_eval_imitation_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_013/semantic_eval_a2_cycle19_confidence/semantic_eval_cycle08_on_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_013/semantic_eval_a2_cycle19_confidence/semantic_eval_final_cycle19_summary.md`

## 1. 要約

`exp_013` は、`exp_012` で新基準となった `value_loss_coef=0.125` を固定した上で、
さらに次の構造整理を反映して `C0 / A1 / A2` を再比較した実験である。

- `CQ-0265`
  - `shanten_hint` / `discard_ukeire_hint` を shared trunk から外し、
    Stage01 と同じ discard direct hint branch に戻した
- `CQ-0266`
  - terminal semantic を 8-class から 5-class に整理した
  - `win_menzen / win_called / draw_tenpai / deal_in / other_non_dealin`

結論は次の通り。

- **今回の実用基準は `A2_semaux_light_vhalf_structfix`**
- `A2` は imitation から final までほぼ落ちず、3 条件で最も安定した
- `A1` は best cycle が非常に強い一方、final まで保持できていない
- `C0` は今回も PPO で大きく悪化しており、semantic aux なしでは保持が弱い
- A2 の semantic diagnostics では、**terminal 5-class 化後に `win_called` が実際に top-1 で取れるようになった**
- 一方で `draw_tenpai` と `deal_in` はまだ弱く、yaku head も PPO で劣化する

したがって `exp_013` は、

- direct hint 切り離しと terminal 5-class 化は正しい方向である
- 特に `win_called` の terminal 学習は前進している
- ただし semantic 学習全体としてはまだ未完成であり、次は `draw_tenpai` / `deal_in` / yaku 保持をどう改善するかを考える段階

と整理するのが自然である。

## 2. 背景

`exp_011` では feature 修正後に `A1/A2` が再び `C0` を上回ったが、
semantic diagnostics ではまだ terminal head の collapse が残っていた。

続く `exp_012` では `value_loss_coef` sweep を行い、

- `value_loss_coef=0.125`

が新しい基準として有望であることを確認した。

その一方で、`exp_012` の深掘りでは

- relevant な `win_called` 局面でも `p(win_called)` がほとんど上がらない
- open 状態なのに `p(win_menzen)` が不自然に残る

という問題があり、semantic 側にはまだ

- discard 専用 local hints が shared trunk に流れている
- terminal class が細かすぎて、意味の薄い終局差まで学ばせている

という構造上のノイズが残っていると考えられた。

そのため `exp_013` では、まず

- direct hint の切り離し
- terminal target の整理

を優先し、その上で `C0 / A1 / A2` を比較した。

## 3. 条件

共通固定:

- `value_loss_coef = 0.125`
- `full observation`
- actor-relative full observation
- full path `riichi` / `menzen`
- `Stage2a` discard direct hints
- terminal semantic 5-class
- seed `42`
- `num_cycles = 20`

比較条件:

### C0: control + vhalf + structfix

- semantic aux 無効

### A1: semantic aux default + vhalf + structfix

- semantic aux 有効
- `policy_projection_dim = 16`
- `terminal_loss_coef = 0.2`
- `yaku_loss_coef = 0.1`

### A2: semantic aux light + vhalf + structfix

- semantic aux 有効
- `policy_projection_dim = 16`
- `terminal_loss_coef = 0.1`
- `yaku_loss_coef = 0.05`

## 4. 主結果

### 4.1 imitation と final

#### C0

- imitation: `avg_rank=2.370`, `win_rate=0.2490`, `deal_in_rate=0.1822`
- final: `avg_rank=2.595`, `win_rate=0.2301`, `deal_in_rate=0.2050`

#### A1

- imitation: `avg_rank=2.455`, `win_rate=0.2425`, `deal_in_rate=0.2022`
- final: `avg_rank=2.545`, `win_rate=0.2327`, `deal_in_rate=0.2100`

#### A2

- imitation: `avg_rank=2.410`, `win_rate=0.2482`, `deal_in_rate=0.1935`
- final: `avg_rank=2.410`, `win_rate=0.2432`, `deal_in_rate=0.2011`

読み:

- `A2` は imitation から final までほぼ落ちていない
- `A1` は imitation より final が悪く、保持に失敗している
- `C0` は imitation から final で大きく悪化しており、今回も最も不安定

### 4.2 best cycle

- `C0`: cycle `9`, `avg_rank=2.325`, `win_rate=0.2311`, `deal_in=0.1957`
- `A1`: cycle `4`, `avg_rank=2.305`, `win_rate=0.2537`, `deal_in=0.1986`
- `A2`: cycle `8`, `avg_rank=2.345`, `win_rate=0.2595`, `deal_in=0.1841`

読み:

- best cycle だけ見ると `A1` が最良
- ただし `A1` はその良さを final まで保持できていない
- `A2` は peak で `A1` に少し負けるが、保持まで含めると総合で優位

### 4.3 tail-5 average

- `C0`: `avg_rank=2.527`, `win_rate=0.2345`, `deal_in=0.1977`
- `A1`: `avg_rank=2.454`, `win_rate=0.2357`, `deal_in=0.1944`
- `A2`: `avg_rank=2.451`, `win_rate=0.2397`, `deal_in=0.1913`

読み:

- tail-5 では `A1/A2` が `C0` より明確に良い
- `A1` と `A2` はほぼ並ぶが、`A2` がわずかに良い
- final と tail-5 の両方で安定しているのは `A2`

### 4.4 PPO 安定性

#### C0

- `ratio_mean=1.0019`
- `clip_fraction=0.0718`
- `anchor_kl_discard=0.0044`

#### A1

- `ratio_mean=1.0035`
- `clip_fraction=0.0866`
- `anchor_kl_discard=0.0052`

#### A2

- `ratio_mean=1.0147`
- `clip_fraction=0.1113`
- `anchor_kl_discard=0.0054`

読み:

- 3 条件とも PPO 指標自体は健全
- 今回の問題は「不安定に発散する」ことではなく、
  **学べたものを保持できるか** にある
- その点で `A2` は、多少動きつつも performance を維持できている

## 5. 主要な解釈

今回の実験で一番大きいのは、

- `CQ-0265`: discard 専用 local hint を semantic/value から切り離した
- `CQ-0266`: terminal semantic を policy に意味のある 5-class に整理した

という 2 つを入れた後でも、

- `A2` は明確に強い
- `A1` は依然として retain が弱い

という差が残ったことである。

これは、

- 構造整理そのものは正しい
- ただし default semantic aux はまだ少し強く、保持コストが高い
- 現時点では light setting の方が PPO と整合しやすい

という読みを支持する。

## 6. semantic diagnostics (A2)

今回の deep dive は `A2_semaux_light_vhalf_structfix` に対して行った。

対象 checkpoint:

- `checkpoint_imitation.pt`
- `checkpoint_cycle_08.pt`
- `checkpoint_learner.pt`

評価 shard:

- `cycle_19/selfplay`

出力先:

- `experiments/Stage02_CallUnlock/exp_013/semantic_eval_a2_cycle19_confidence/`

補足:

- 既存の `semantic_head_eval.py` は `CQ-0265` の direct hint 再構築に未追従だったため、
  今回は runner と同じ再構築ロジックで one-off 評価を行った
- 結果ファイル自体は保存済みであり、診断値は再利用可能

### 6.1 terminal の推移

#### `win_called`

- imitation
  - recall `0.0021`
  - `mean_p = 0.1512`
  - `top1_hit = 0.0021`
  - `top3_hit = 0.9888`
- cycle 08
  - recall `0.0521`
  - `mean_p = 0.2315`
  - `top1_hit = 0.0521`
  - `top3_hit = 0.9980`
- final
  - recall `0.2155`
  - `mean_p = 0.2445`
  - `top1_hit = 0.2155`
  - `top3_hit = 0.9978`

読み:

- これはかなり大きい改善である
- 少なくとも `win_called` は、以前の 8-class terminal のような
  「ほぼ認識できていない」状態から抜けている
- final では **`win_called` が実際に top-1 で選ばれる** ようになった

#### `win_menzen`

- imitation
  - recall `0.0011`
  - `mean_p = 0.0556`
- cycle 08
  - recall `0.2259`
  - `mean_p = 0.2180`
- final
  - recall `0.0473`
  - `mean_p = 0.1383`

読み:

- `win_menzen` は一度かなり持ち上がるが、final では再び落ちる
- `A2` の terminal 改善は主に `win_called` 側で起きている

#### `draw_tenpai`

- imitation
  - recall `0.0000`
  - `mean_p = 0.0438`
- cycle 08
  - recall `0.0000`
  - `mean_p = 0.0542`
- final
  - recall `0.0000`
  - `mean_p = 0.0483`

読み:

- 5-class 化後も `draw_tenpai` はまだ学べていない
- terminal の次の改善対象としてかなり明確

#### `deal_in`

- imitation
  - recall `0.0000`
  - `mean_p = 0.1880`
  - `top3_hit = 0.9858`
- cycle 08
  - recall `0.0000`
  - `mean_p = 0.1063`
  - `top3_hit = 0.2760`
- final
  - recall `0.0000`
  - `mean_p = 0.1695`
  - `top3_hit = 0.6195`

読み:

- `deal_in` は probability mass 自体は持てている
- ただし top-1 で選ぶところまで行っていない
- `win_called` に比べると、まだ semantic target として弱い

#### `other_non_dealin`

- imitation
  - recall `0.9988`
  - `mean_p = 0.6112`
- final
  - recall `0.7478`
  - `mean_p = 0.4089`

読み:

- imitation の時点では `other_non_dealin` 一辺倒に近い
- final ではそこから離れ、`win_called` / `win_menzen` / `deal_in` に確率を配れるようになっている
- この点でも terminal 5-class 化の効果は出ている

### 6.2 yaku

#### imitation

- micro F1 `0.4322`
- macro F1 `0.0961`
- exact match `0.3429`

#### cycle 08

- micro F1 `0.3020`
- macro F1 `0.0585`
- exact match `0.1678`

#### final

- micro F1 `0.3030`
- macro F1 `0.0641`
- exact match `0.1566`

読み:

- yaku は imitation から PPO で落ちる傾向が残っている
- terminal と違い、yaku は今回まだ「改善が明確」とは言いにくい
- 特に `Riichi` と `Yakuhai` 以外はかなり弱い

## 7. 総合判断

今回の総合判断は次の通り。

### 7.1 何が改善したか

- semantic に不要な discard local hint を切り離せた
- terminal target が policy に意味のある 5-class に整理された
- その結果、**A2 では `win_called` が実際に terminal として学ばれ始めた**

### 7.2 何がまだ弱いか

- `A1` の保持問題
- `draw_tenpai`
- `deal_in`
- yaku head の PPO 中劣化

### 7.3 現時点の実用判断

- **新しい基準条件は `A2_semaux_light_vhalf_structfix`**
- `A1` は best cycle の強さから見るとまだ捨てる必要はない
- ただし実用上は、保持まで含めると現状 `A2` の方が良い

## 8. 次の一手

今回の結果を踏まえると、次はこの順が自然である。

1. `exp_013/report.md` を基準レポートとして固定する
2. 必要なら `A1` の `cycle_04` を同じ 5-class terminal で診断し、A2 と比較する
3. その上で、次の改善対象を絞る
   - `draw_tenpai` / `deal_in` を学ばせる sampling / weighting
   - yaku head の保持
   - A1 retain 問題の再検討

現時点では、**構造整理だけではもう十分ではなく、次は semantic 学習設計そのものを詰める段階**に入ったと考えてよい。
