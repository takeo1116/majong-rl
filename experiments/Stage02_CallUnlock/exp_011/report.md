# Experiment Report: exp_011

作成日: 2026-04-02  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_011/runbook.md`
- `experiments/Stage02_CallUnlock/exp_010/report.md`
- `experiments/Stage02_CallUnlock/exp_009_bugfix/bug_report.md`
- `experiments/Stage02_CallUnlock/exp_011/semantic_eval_a1_cycle19_confidence/semantic_eval_imitation_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_011/semantic_eval_a1_cycle19_confidence/semantic_eval_cycle05_on_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_011/semantic_eval_a1_cycle19_confidence/semantic_eval_final_cycle19_summary.md`

## 1. 要約

`exp_011` は、`exp_010` と同じ学習条件を維持したまま、feature 側に入れた

- actor-relative full observation (`CQ-0263`)
- full path の actor-relative `riichi` / `menzen` (`CQ-0264`)

の効果を確認する再実験である。

結論は次の通り。

- **feature 修正はかなり効いている**
- `C0` より `A1` / `A2` が明確に良く、`exp_010` より semantic aux が活きやすくなった
- 現時点の本命は **`A1_semaux_default_featurefix`**
  - final `avg_rank` が最良
  - imitation から final への改善も最も自然
  - PPO 安定性も良好
- `A2_semaux_light_featurefix` も強いが、imitation が既に強く、PPO での上積みは弱い
- semantic diagnostics では、**A1 の imitation checkpoint で `win_called` の confidence が `exp_010` より改善**した
- ただし `checkpoint_cycle_05` と final では、terminal head は再び `ron_bystander` 偏重に collapse しており、**semantic head の PPO 中 collapse はまだ未解決**である

したがって、`exp_011` は

- feature 欠損が semantic aux の有効性をかなり邪魔していた
- その欠損を埋めると policy performance は改善する
- ただし semantic head 自体の保持にはまだ別の対策が必要

と整理するのが自然である。

## 2. 実験目的

`exp_010` では、open-hand bugfix 後に semantic auxiliary 比較を取り直したが、

- `win_called` support は復活した
- しかし `A1` の優位はかなり弱まった
- semantic diagnostics では `win_called` をほぼ学べていなかった

という結果だった。

その後の特徴量レビューで、full observation にはまだ

- 4 家 block が seat-fixed
- `riichi` が 0 埋め
- `menzen` が未入力

という欠損があることが分かった。

そこで `exp_011` では、学習条件はそのままに、

- actor-relative full observation
- actor-relative `riichi`
- actor-relative `menzen`

を導入した上で、`C0 / A1 / A2` を再比較した。

## 3. 条件

共通固定:

- `core_minimal`
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

### C0: feature-fix 後 R3 control

- `model.semantic_aux.enabled=false`
- `training.semantic_aux.enabled=false`

### A1: feature-fix 後 semantic aux default

- `model.semantic_aux.enabled=true`
- `training.semantic_aux.enabled=true`
- `policy_projection_dim=16`
- `terminal_loss_coef=0.2`
- `yaku_loss_coef=0.1`

### A2: feature-fix 後 semantic aux light

- `model.semantic_aux.enabled=true`
- `training.semantic_aux.enabled=true`
- `policy_projection_dim=16`
- `terminal_loss_coef=0.1`
- `yaku_loss_coef=0.05`

## 4. 主結果

### 4.1 imitation 直後と final

#### C0

- imitation: `avg_rank=2.515`, `win_rate=0.2321`, `deal_in_rate=0.1856`
- final: `avg_rank=2.555`, `win_rate=0.2288`, `deal_in_rate=0.1890`

#### A1

- imitation: `avg_rank=2.500`, `win_rate=0.2333`, `deal_in_rate=0.1961`
- final: `avg_rank=2.430`, `win_rate=0.2428`, `deal_in_rate=0.1938`

#### A2

- imitation: `avg_rank=2.365`, `win_rate=0.2396`, `deal_in_rate=0.1943`
- final: `avg_rank=2.440`, `win_rate=0.2439`, `deal_in_rate=0.1894`

読み:

- `C0` は今回も PPO で改善しきれていない
- `A1` は imitation から final へ素直に改善している
- `A2` は imitation 時点でかなり強いが、PPO での上積みは小さい

### 4.2 PPO 安定性

#### C0

- `ratio_mean=1.0148`
- `clip_fraction=0.2281`
- `anchor_kl_discard=0.0223`

#### A1

- `ratio_mean=1.0069`
- `clip_fraction=0.2314`
- `anchor_kl_discard=0.0192`

#### A2

- `ratio_mean=1.0183`
- `clip_fraction=0.2301`
- `anchor_kl_discard=0.0193`

読み:

- 3 条件とも安定している
- `exp_010` と比べると、特に `A1/A2` の PPO 指標はかなり素直になった

### 4.3 best cycle

- `C0`: cycle 8, `avg_rank=2.350`, `win_rate=0.2390`, `deal_in_rate=0.1830`
- `A1`: cycle 5, `avg_rank=2.350`, `win_rate=0.2499`, `deal_in_rate=0.1857`
- `A2`: cycle 8, `avg_rank=2.300`, `win_rate=0.2437`, `deal_in_rate=0.1780`

### 4.4 tail-5 average

- `C0`: `avg_rank=2.493`, `win_rate=0.2377`, `deal_in_rate=0.1908`
- `A1`: `avg_rank=2.454`, `win_rate=0.2402`, `deal_in_rate=0.1962`
- `A2`: `avg_rank=2.457`, `win_rate=0.2362`, `deal_in_rate=0.1883`

読み:

- tail-5 では `A1/A2` が `C0` より明確に良い
- `A1` と `A2` はほぼ並ぶが、`A1` の方が final と tail-5 の一貫性が少し高い

## 5. `exp_010` との比較

`exp_010` では、feature 欠損を含んだ状態で

- `A1` final: `avg_rank=2.545`
- `A2` final: `avg_rank=2.485`
- `C0` final: `avg_rank=2.600`

だった。

`exp_011` では

- `A1` final: `avg_rank=2.430`
- `A2` final: `avg_rank=2.440`
- `C0` final: `avg_rank=2.555`

となった。

特に大きいのは `A1` で、

- imitation から final への改善が `exp_010` より明確
- PPO 安定性も `exp_010` よりかなり良い

したがって、**actor-relative + `riichi/menzen` 追加は semantic aux の実効性を高めた**と読むのが自然である。

## 6. `win_called` support

今回の feature 修正は teacher semantics 自体は変えないため、imitation 側の `win_called` support は `exp_010` と同様に維持されている。

### imitation data

全条件共通:

- `overall_win_called = 167,377`
- `call_win_called = 37,062`

### cycle_19 selfplay

- `C0`: `overall_win_called = 11,210`, `call_win_called = 2,466`
- `A1`: `overall_win_called = 11,079`, `call_win_called = 2,427`
- `A2`: `overall_win_called = 11,535`, `call_win_called = 2,552`

したがって、`win_called` support は今回も十分存在している。

## 7. semantic diagnostics (A1)

`A1_semaux_default_featurefix` について、同じ `cycle_19/selfplay` shard 上で

- `checkpoint_imitation.pt`
- `checkpoint_cycle_05.pt`
- `checkpoint_learner.pt`

を評価した。

対象:
- `experiments/Stage02_CallUnlock/exp_011/semantic_eval_a1_cycle19_confidence/`

### 7.1 imitation checkpoint

#### terminal

- accuracy: `0.4141`
- `win_menzen` recall: `0.1103`
- `win_called` recall: `0.0060`
- `ron_bystander` recall: `0.9120`

`win_called` confidence:

- `mean_p = 0.1153`
- `p50 = 0.1097`
- `p90 = 0.1755`
- `top1_hit_rate = 0.0060`
- `top3_hit_rate = 0.1763`
- `mean_rank = 3.8`

#### yaku

- micro F1: `0.3874`
- macro F1: `0.0884`
- exact match: `0.2570`

主な positive-conditioned confidence:

- `Riichi`: `mean_p=0.5822`, `hit@0.5=0.6037`
- `Yakuhai`: `mean_p=0.4451`, `hit@0.5=0.3803`
- `Tanyao`: `mean_p=0.1515`, `top3=0.7522`

読み:

- `exp_010` の imitation より、**`win_called` の terminal confidence は改善**している
- 特に
  - `mean_p: 0.0994 -> 0.1153`
  - `top3_hit_rate: 0.1073 -> 0.1763`
  - `mean_rank: 4.01 -> 3.8`
  と良化した
- したがって、feature 修正は少なくとも imitation semantic には効いている

### 7.2 cycle_05 checkpoint

#### terminal

- accuracy: `0.4323`
- `win_menzen` recall: `0.0625`
- `win_called` recall: `0.0000`
- `ron_bystander` recall: `0.9979`

`win_called` confidence:

- `mean_p = 0.0795`
- `top1_hit_rate = 0.0000`
- `top3_hit_rate = 0.0033`
- `mean_rank = 4.1`

#### yaku

- micro F1: `0.3175`
- macro F1: `0.0689`
- exact match: `0.1710`

主な positive-conditioned confidence:

- `Riichi`: `mean_p=0.5639`, `hit@0.5=0.7566`
- `Yakuhai`: `mean_p=0.3162`, `hit@0.5=0.0724`
- `Tanyao`: `mean_p=0.1472`, `top3=0.9024`

読み:

- cycle 5 の policy performance は良いが、semantic head はすでにかなり崩れている
- terminal は `ron_bystander` 1 強に近く、`win_called` は imitation より大きく悪化している

### 7.3 final checkpoint

#### terminal

- accuracy: `0.4321`
- `win_menzen` recall: `0.0590`
- `win_called` recall: `0.0000`
- `ron_bystander` recall: `0.9978`

`win_called` confidence:

- `mean_p = 0.0930`
- `top1_hit_rate = 0.0000`
- `top3_hit_rate = 0.0060`
- `mean_rank = 4.0`

#### yaku

- micro F1: `0.2159`
- macro F1: `0.0457`
- exact match: `0.1057`

主な positive-conditioned confidence:

- `Riichi`: `mean_p=0.4780`, `hit@0.5=0.3188`
- `Yakuhai`: `mean_p=0.3857`, `hit@0.5=0.1093`
- `Tanyao`: `mean_p=0.2397`, `hit@0.2=0.9512`, `top3=0.9964`

読み:

- final でも terminal collapse は解消していない
- `win_called` は imitation では少し見えているが、PPO 中に維持できていない
- yaku も、imitation から cycle_05、さらに final に向けて全体として悪化する

## 8. 解釈

### 8.1 feature 修正の効果は本物

今回もっとも重要なのは、`exp_010` と比べて

- `A1/A2` の final performance が改善
- PPO 安定性も改善
- `A1` の imitation semantic で `win_called` confidence が改善

したことである。

これは、これまでの弱さに

- actor-relative でない full observation
- `riichi` / `menzen` 欠損

がかなり効いていたことを支持する。

### 8.2 ただし semantic head の PPO collapse は残る

一方で、`checkpoint_cycle_05` と final の診断を見ると、

- terminal head は `ron_bystander` へ強く寄る
- `win_called` recall は 0 に戻る
- yaku head も imitation より悪化する

という問題はまだ残っている。

したがって、今の改善は

- semantic head 自体が完全に健全化した

というより、

- **feature 修正により trunk / policy 側は改善したが、aux head / loss の保持はまだ弱い**

と読む方が自然である。

### 8.3 A1 と A2 の位置づけ

- `A1`
  - final `avg_rank` 最良
  - imitation から final への改善が自然
  - PPO 安定性も良好
- `A2`
  - imitation は最良
  - final `win_rate` は最良
  - ただし PPO での上積みは小さい

このため、現時点では **本線は A1、A2 は有力な対抗** と整理する。

## 9. まとめ

`exp_011` の時点で言えることは次の通り。

1. actor-relative + `riichi/menzen` 追加は有効だった
2. `A1/A2` は `C0` より明確に良く、semantic aux は再び有望な方向に戻った
3. `A1` の imitation semantic では `win_called` confidence が改善した
4. ただし PPO 中に semantic head が collapse する問題は残っている

したがって次の焦点は、

- `A1` を multi-seed で確かめるか
- 先に semantic head の collapse を抑えるか

の 2 択になる。

現時点では、**policy performance の観点では `A1` を 3 seeds に広げる価値があり、同時に semantic head の保持設計は follow-up で見る価値が高い**。
