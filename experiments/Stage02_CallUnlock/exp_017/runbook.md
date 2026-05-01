# Experiment Runbook: exp_017

作成日: 2026-04-09  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_015/report.md`
- `experiments/Stage02_CallUnlock/exp_016/report.md`
- `experiments/Stage02_CallUnlock/exp_016/runbook.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

## 1. 背景

`exp_015` では、terminal player-round 正規化 (`CQ-0268`) を入れた結果、

- `A2_semaux_light_vhalf_tenpaifix_prnorm` が practical baseline になった
- terminal の綱引きが `exp_014` より安定した
- `draw_tenpai` は少し後退したが、`win_called / win_menzen` は回復した

という状態になった。

その後 `exp_016` では、`CQ-0270` により self tile-presence flags を共有 encoder 入力に追加した。

追加した特徴:

- `self_has_honor`
- `self_has_terminal`
- `self_has_simple`
- `self_has_man`
- `self_has_pin`
- `self_has_sou`

狙い:

- `Tanyao` のような「字牌 / 么九牌が存在しないこと」が条件になる役を、
  34次元 counts だけよりも MLP が扱いやすくする

しかし `exp_016` の結果は、

- policy は `exp_015` baseline より悪化
- `Tanyao` は改善しない
- 一方で `Riichi` や一部 terminal class は強まる

というものだった。

したがって現時点では、

- **特徴量アイデア自体を棄却するには早い**
- しかし **shared input にそのまま足すだけでは baseline を更新できない**

と整理している。

## 2. 今回の問い

`exp_017` の問いは次の 2 点である。

1. `yakuflags` の効果が悪かった主因は、`value / terminal / yaku` 側の表現容量不足か
2. `value_hidden_dims` を広げることで、
   - `yakuflags` の悪影響が消えるか
   - `Tanyao` や yaku macro F1 が改善するか

言い換えると、今回は

- `tile_presence_flags` 自体の意味
- それを **現在の trunk 幅で受け止められるか**

を切り分ける実験である。

## 3. 実験方針

今回は `A2` のみを固定し、

- `tile_presence_flags`: `off / on`
- `value_hidden_dims`: `narrow / wide`

の 2x2 factorial にする。

理由:

- 問いたいのは `semantic/value` 側の capacity 仮説
- `C0` を入れるより、A2 の 4 条件をきれいに比較した方が情報効率が高い
- `CQ-0272` により `tile_presence_flags` を config で on/off できるようになったため、
  同一コードベースで比較できる

## 4. 比較条件

すべて `A2_semaux_light_vhalf_tenpaifix_prnorm` 系で固定する。

### 4.1 off + narrow

ラベル:

- `A2_semaux_light_vhalf_tenpaifix_prnorm_off_narrow`

設定:

- `feature_encoder.tile_presence_flags = false`
- `model.value_hidden_dims = [128, 64]`

意味:

- 実質 `exp_015` baseline の再現条件

### 4.2 off + wide

ラベル:

- `A2_semaux_light_vhalf_tenpaifix_prnorm_off_widevalue`

設定:

- `feature_encoder.tile_presence_flags = false`
- `model.value_hidden_dims = [256, 128]`

意味:

- `yakuflags` なしで `value / terminal / yaku` 側だけ容量を増やした条件

### 4.3 on + narrow

ラベル:

- `A2_semaux_light_vhalf_tenpaifix_prnorm_on_narrow`

設定:

- `feature_encoder.tile_presence_flags = true`
- `model.value_hidden_dims = [128, 64]`

意味:

- 実質 `exp_016` A2 条件の再現

### 4.4 on + wide

ラベル:

- `A2_semaux_light_vhalf_tenpaifix_prnorm_on_widevalue`

設定:

- `feature_encoder.tile_presence_flags = true`
- `model.value_hidden_dims = [256, 128]`

意味:

- 今回の本命条件
- `yakuflags` が capacity 拡張で初めて活きるかを直接見る

## 5. 共通固定条件

全条件共通:

- `training.value_loss_coef = 0.125`
- `training.policy_anchor.coef = 0.75`
- `training.multi_cycle.num_cycles = 20`
- `experiment.global_seed = 42`
- semantic aux 有効
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`

前提として保持するもの:

- latest direct hint branch (`CQ-0265`)
- terminal 5-class (`CQ-0266`)
- `self_tenpai_flag` / `remaining_draws_norm` (`CQ-0267`)
- terminal player-round normalization (`CQ-0268`)
- `deal_in` risk diagnostics (`CQ-0269`)
- illegal discard snapshot fix (`CQ-0271`)

## 6. 必須観測

### 6.1 通常性能

- imitation eval
- final
- best cycle
- tail-5 average

特に見る差分:

- `off_narrow` vs `off_wide`
- `on_narrow` vs `on_wide`
- `off_wide` vs `on_wide`

### 6.2 PPO 安定性

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- learner loss の暴れ
- final までの保持

### 6.3 terminal diagnostics

少なくとも `off_narrow`, `on_narrow`, `on_wide` について取る。

見るもの:

- terminal accuracy
- `win_menzen`
- `win_called`
- `draw_tenpai`
- `deal_in`

特に、

- `on_narrow` で悪化したものが `on_wide` で戻るか

を確認する。

### 6.4 yaku diagnostics

今回の本命はここである。

最低限見るもの:

- micro F1
- macro F1
- exact match
- `Riichi`
- `Yakuhai`
- `Tanyao`

余力があれば:

- `Pinfu`
- `MenzenTsumo`

加えて、

- winner player-round の最後の 3 decision に限定した yaku 集計

も取る。

### 6.5 `deal_in` risk diagnostics

`CQ-0269` で追加した次を使う。

- overall `mean_p_pos / mean_p_neg`
- overall `roc_auc / pr_auc`
- `late_and_noten`
- `early_and_tenpai`

## 7. 成功判定

今回の成功条件は、次のいずれかを満たすこととする。

1. `on_wide` が `on_narrow` より policy を明確に改善する
2. `on_wide` で `Tanyao` の `mean_p` か recall が動く
3. `on_wide` で yaku macro F1 が改善する
4. `on_wide` が `off_wide` と比較して、policy を大きく損なわず yaku を改善する

特に理想形は、

- `on_narrow` ではダメ
- `on_wide` で改善

が出ること。

この場合、`yakuflags` は「悪い特徴量」ではなく、
**現在の semantic/value 側容量では活かしきれていなかった**
と解釈できる。

## 8. 失敗時の読み方

もし `on_wide` でも

- `Tanyao` が動かない
- policy が戻らない
- yaku macro も改善しない

なら、問題は容量不足よりも

- shared input に入れる位置
- yaku 用 loss の掛け方
- yaku 専用 branch / head 設計

にある可能性が高い。

その場合は、

- `yakuflags` を shared encoder feature として常時使う方針は保留
- yaku 側だけに限定して入れる案
- yaku loss 側の再設計

を次の候補にする。

## 9. 実行メモ

driver:

```bash
./.venv/bin/python scripts/local/stage2/exp_017_driver.py
```

1 条件だけ回す:

```bash
EXP017_ONLY=A2_semaux_light_vhalf_tenpaifix_prnorm_on_widevalue \
  ./.venv/bin/python scripts/local/stage2/exp_017_driver.py
```

期待される成果物:

- `experiments/Stage02_CallUnlock/exp_017/run_map.json`
- `experiments/Stage02_CallUnlock/exp_017/driver_logs/*.log`
- 各 run の `summary.json`

