# Experiment Runbook: exp_018

作成日: 2026-04-10  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_015/report.md`
- `experiments/Stage02_CallUnlock/exp_016/report.md`
- `experiments/Stage02_CallUnlock/exp_017/report.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

## 1. 背景

`exp_017` では、`tile_presence_flags` の有無と `value_hidden_dims` の幅を
`A2` 固定で 2x2 に切って比較した。

結果として、

- `yakuflags on + narrow` は明確に悪い
- `yakuflags on + wide` では policy がかなり回復
- 特に `Tanyao` の `mean_p / hit@0.2` は大きく改善

という傾向が見えた。

したがって、

- 追加した flags 自体が完全に悪いわけではない
- ただし raw feature を policy trunk まで直接入れるのは重い

という整理になった。

その後 `CQ-0273` により、

- encoder は従来どおり `tile_presence_flags` を出す
- `value / terminal / yaku` 側には raw のまま入れる
- `discard / optional` の raw policy 入力からは除外する

という `semantic_only` routing が実装された。

## 2. 今回の問い

`exp_018` の問いは次の 2 点である。

1. `tile_presence_flags` を **semantic/value 側だけに raw 入力**すれば、
   `exp_017 on_wide` で見えた `Tanyao` 信号を保ちつつ policy 悪化を減らせるか
2. `semantic_only` にしたうえで、まだ `wide` が必要なのか

言い換えると、今回は

- `flags を入れるかどうか`
ではなく
- `flags をどこに入れるべきか`

を切る実験である。

## 3. 実験方針

今回は **新規 2 条件のみ**を回す。

比較アンカーとしては `exp_017` の既存結果を使い回す。

再利用するアンカー:

- `exp_017 / A2_semaux_light_vhalf_tenpaifix_prnorm_off_narrow`
- `exp_017 / A2_semaux_light_vhalf_tenpaifix_prnorm_on_widevalue`

理由:

- `CQ-0273` は `tile_presence_flags_semantic_only=false` なら現行挙動を保つ
- よって `off_narrow` と `on_wide` を再実行しなくても、
  今回の問いには十分比較できる
- 新規実行コストを 2 条件に絞れる

## 4. 新規比較条件

すべて `A2_semaux_light_vhalf_tenpaifix_prnorm` 系で固定する。

### 4.1 on + semantic_only + narrow

ラベル:

- `A2_semaux_light_vhalf_tenpaifix_prnorm_on_semonly_narrow`

設定:

- `feature_encoder.tile_presence_flags = true`
- `model.semantic_aux.tile_presence_flags_semantic_only = true`
- `model.value_hidden_dims = [128, 64]`

意味:

- `semantic_only` routing 単独で改善するかを見る

### 4.2 on + semantic_only + wide

ラベル:

- `A2_semaux_light_vhalf_tenpaifix_prnorm_on_semonly_widevalue`

設定:

- `feature_encoder.tile_presence_flags = true`
- `model.semantic_aux.tile_presence_flags_semantic_only = true`
- `model.value_hidden_dims = [256, 128]`

意味:

- 今回の本命条件
- `exp_017 on_wide` と比べて、
  raw flag を policy trunk に直接入れないことが効くかを見る

## 5. 比較の読み方

### 5.1 baseline 比較

比較対象:

- `exp_017 off_narrow`
- `exp_018 on_semonly_narrow`
- `exp_018 on_semonly_wide`

見ること:

- `semantic_only` routing が baseline を上回れるか

### 5.2 routing 比較

比較対象:

- `exp_017 on_wide`
- `exp_018 on_semonly_wide`

見ること:

- 同じ `tile_presence_flags=true + wide` で
  - all trunks raw input
  - semantic/value only raw input
  の差を切る

### 5.3 capacity 比較

比較対象:

- `exp_018 on_semonly_narrow`
- `exp_018 on_semonly_wide`

見ること:

- `semantic_only` にしたあとも `wide` が必要か

## 6. 共通固定条件

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

- direct hint branch (`CQ-0265`)
- terminal 5-class (`CQ-0266`)
- `self_tenpai_flag` / `remaining_draws_norm` (`CQ-0267`)
- terminal player-round normalization (`CQ-0268`)
- `deal_in` risk diagnostics (`CQ-0269`)
- illegal discard snapshot fix (`CQ-0271`)
- `tile_presence_flags` on/off flag (`CQ-0272`)
- `tile_presence_flags` semantic-only routing (`CQ-0273`)

## 7. 必須観測

### 7.1 通常性能

- imitation eval
- final
- best cycle
- tail-5 average

特に見る比較:

- `exp_017 on_wide` vs `exp_018 on_semonly_wide`
- `exp_017 off_narrow` vs `exp_018 on_semonly_wide`

### 7.2 terminal diagnostics

少なくとも新規 2 条件について取る。

見るもの:

- terminal accuracy
- `win_menzen`
- `win_called`
- `draw_tenpai`
- `deal_in risk`

### 7.3 yaku diagnostics

今回の本命はここである。

見るもの:

- micro F1
- macro F1
- exact match
- `Tanyao`
- `Riichi`
- `Yakuhai`

特に `Tanyao` は:

- recall
- mean_p
- hit@0.2
- last-3 winner-only

を比較する。

## 8. 成功判定

最低限、以下のどれかが欲しい。

1. `on_semonly_wide` の policy が `exp_017 on_wide` より改善
2. `on_semonly_wide` の `Tanyao mean_p / hit@0.2` が `exp_017 on_wide` 並み
3. `on_semonly_narrow` が `on_wide` よりかなり良ければ、
   問題の主因が raw policy 入力だったと判断できる
4. `on_semonly_wide` が `exp_017 off_narrow` に近づく、または超える

## 9. 失敗時の解釈

### 9.1 `on_semonly_wide` でも改善しない

この場合は、

- 問題は routing より loss 側
- あるいは yaku/value 側でも接続位置がまだ広すぎる

と考える。

次候補:

- yaku head 限定入力
- yaku loss の mild reweight

### 9.2 `on_semonly_narrow` は良いが `wide` で悪い

この場合は、

- `semantic_only` routing 自体は正しい
- ただし `wide` 化で別の shortcut が増えている

可能性がある。

## 10. 実行コマンド

全条件:

```bash
./.venv/bin/python scripts/local/stage2/exp_018_driver.py
```

1 条件のみ:

```bash
EXP018_ONLY=A2_semaux_light_vhalf_tenpaifix_prnorm_on_semonly_widevalue \
  ./.venv/bin/python scripts/local/stage2/exp_018_driver.py
```

## 11. 期待アウトプット

- `experiments/Stage02_CallUnlock/exp_018/run_map.json`
- `experiments/Stage02_CallUnlock/exp_018/driver_logs/*.log`
- 新規 2 run の `summary.json`
- 必要に応じて semantic/yaku diagnostics

