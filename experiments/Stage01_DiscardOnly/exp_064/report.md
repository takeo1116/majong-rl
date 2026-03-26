# Experiment Report: exp_064

作成日: 2026-03-21  
対象: bugfix 後の新モデルを前提に、`rule-only PPO + policy_anchor(coef=0.5) + clip_epsilon=0.15` を固定し、`gamma` と `gae_lambda` を再 sweep した。

条件:
- A: `g050_gae000`
- B: `g050_gae030`
- C: `g050_gae060`
- D: `g075_gae000`
- E: `g075_gae030`
- F: `g075_gae060`

参照:
- `experiments/exp_064/runbook.md`
- `experiments/exp_064/run_map.json`

補足:
- 実行時の `batch_summary.json` や `runs/` 配下のローカル成果物は、VCS に載せない前提のため本 report からは参照しない

補足:
- E `g075_gae030` は一度 Windows 自動再起動で中断したため、途中 batch ではなく **rerun 完了 batch** を採用している

## 1. 結論

今回の sweep から強く言えるのは次の 4 点である。

1. **`gae_lambda=0.3` が有望**
   - `gamma=0.50` でも `0.75` でも `gae=0.0` より final `avg_score` が上がった
   - しかも `best -> final` drawdown も改善している

2. **`gae_lambda=0.6` は悪い**
   - 両 gamma で final `avg_score` が大きく悪化した
   - peak は作れても保持できていない

3. **最良条件は `gamma=0.75, gae_lambda=0.3`**
   - final `avg_score = 2297.0` で全条件中トップ
   - drawdown も小さい

4. **ただし `gamma` の優劣は単純ではない**
   - `gae=0.0` では `gamma=0.50` がやや優位
   - `gae=0.3` では `gamma=0.75` が優位
   - `gae=0.6` では両方悪いが、`gamma=0.75` の方がさらに悪い

したがって、post-bugfix baseline の暫定更新先としては

- **旧 baseline**: `gamma=0.50, gae_lambda=0.0`
- **新 baseline 候補**: `gamma=0.75, gae_lambda=0.3`

が自然である。

## 2. 実験条件

共通:
- 新モデル (`policy_direct_hints + context_gate`)
- `training.imitation_loss_mode=tie_aware_best_set`
- imitation `1000 matches x 3 chunks`
- PPO `200 matches x 30 cycles`
- `policy_anchor.coef=0.5`
- `clip_epsilon=0.15`
- `reward.shaping.shanten_delta.scale=0.003`
- `training.rule_mix.policy_ratio=0.0`
- seeds `42,43,44`

差分:
- A: `gamma=0.50, gae_lambda=0.0`
- B: `gamma=0.50, gae_lambda=0.3`
- C: `gamma=0.50, gae_lambda=0.6`
- D: `gamma=0.75, gae_lambda=0.0`
- E: `gamma=0.75, gae_lambda=0.3`
- F: `gamma=0.75, gae_lambda=0.6`

補足:
- A `g050_gae000` は current baseline と同一条件の既存 batch を再利用した
- `gamma` / `gae` は PPO 側の horizon を変えるノブであり、imitation 設定そのものは共通である

## 3. Warmstart は共通

6 条件とも imitation warmstart 設定は同一である。  
したがって今回の差は基本的に **PPO 中の `gamma / gae` 差** と見てよい。

注意:
- `cycle0.eval_before.avg_score` の条件差は、warmstart が異なるというより **eval のばらつき** を含む
- したがって今回の比較は
  - final 指標
  - peak
  - drawdown
  - learner 診断
を中心に読むのが自然である

## 4. 最終結果

| 条件 | gamma | gae | final avg_rank | final avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|---:|---:|
| A `g050_gae000` | `0.50` | `0.0` | `2.3583` | `1943.0` | `0.2914` | `0.4539` |
| B `g050_gae030` | `0.50` | `0.3` | `2.3283` | `2072.5` | `0.2913` | `0.4542` |
| C `g050_gae060` | `0.50` | `0.6` | `2.3575` | `1695.3` | `0.2904` | `0.4544` |
| D `g075_gae000` | `0.75` | `0.0` | `2.3758` | `1856.0` | `0.2927` | `0.4520` |
| E `g075_gae030` | `0.75` | `0.3` | `2.3317` | `2297.0` | `0.2928` | `0.4497` |
| F `g075_gae060` | `0.75` | `0.6` | `2.4075` | `1271.9` | `0.2902` | `0.4593` |

読み方:
- **score 最良は E `g075_gae030`**
- **rank 最良は B `g050_gae030`**
  - ただし E との差はかなり小さい
- `gae=0.3` の 2 条件が総合上位
- `gae=0.6` の 2 条件は明確に下位

ここからまず安全に言えるのは、

- **`gae=0.3` は current baseline より良い**
- **`gae=0.6` は current baseline より悪い**

である。

## 5. Peak と保持

| 条件 | best cycle mean | best avg_score mean | final avg_score | best→final drawdown |
|---|---:|---:|---:|---:|
| A `g050_gae000` | `6.0` | `2982.0` | `1943.0` | `1039.0` |
| B `g050_gae030` | `4.0` | `2790.6` | `2072.5` | `718.1` |
| C `g050_gae060` | `9.3` | `3150.7` | `1695.3` | `1455.3` |
| D `g075_gae000` | `17.3` | `2902.2` | `1856.0` | `1046.2` |
| E `g075_gae030` | `13.0` | `3036.3` | `2297.0` | `739.3` |
| F `g075_gae060` | `8.7` | `3191.1` | `1271.9` | `1919.2` |

ここが今回かなり重要である。

### `gae=0.3`
- B/E とも drawdown が小さい
- **peak を作るだけでなく、ある程度保持できている**

### `gae=0.0`
- A/D は peak 後に 1000 点規模で落ちる
- current baseline の弱点は依然として残っている

### `gae=0.6`
- C/F は peak 自体は高い
- しかし final までに大きく落ちる
- 特に F は drawdown が `1919` と最大

したがって、

**`gae=0.3` は「改善と保持のバランス」が良く、`gae=0.6` は「改善するが保持できない」**

と読むのが自然である。

## 6. gamma ごとの比較

### `gae=0.0`
- A `g050_gae000`: `1943.0`
- D `g075_gae000`: `1856.0`

この帯では **`gamma=0.50` がやや良い**。

### `gae=0.3`
- B `g050_gae030`: `2072.5`
- E `g075_gae030`: `2297.0`

この帯では **`gamma=0.75` が明確に良い**。

### `gae=0.6`
- C `g050_gae060`: `1695.3`
- F `g075_gae060`: `1271.9`

この帯では **両方悪いが、`gamma=0.75` がさらに悪い**。

つまり今回の `gamma` については、

**「高い方が常に良い / 低い方が常に良い」ではなく、`gae=0.3` のときにだけ `0.75` が効いている**

という整理になる。

## 7. learner 診断

final learner 診断の代表値:

| 条件 | best_set_hit_after | clip_fraction | ratio_std | late.value_error |
|---|---:|---:|---:|---:|
| A `g050_gae000` | `0.9096` | `0.1954` | `0.1265` | `0.00299` |
| B `g050_gae030` | `0.9098` | `0.1922` | `0.1252` | `0.00308` |
| C `g050_gae060` | `0.9102` | `0.1743` | `0.1182` | `0.00186` |
| D `g075_gae000` | `0.9113` | `0.1964` | `0.1268` | `0.00320` |
| E `g075_gae030` | `0.9107` | `0.1815` | `0.1207` | `0.00378` |
| F `g075_gae060` | `0.9111` | `0.1645` | `0.1142` | `0.00374` |

ここから読み取れること:

1. **teacher safety rail はどの条件でも大きくは崩れていない**
   - `best_set_hit_after` は全条件ほぼ `0.91`
   - したがって今回の差は、teacher rail の維持率より **その先の改善の質** にありそう

2. **数値的に穏やかな条件が、そのまま強いわけではない**
   - `gae=0.6` では `clip_fraction` / `ratio_std` はむしろ低い
   - それでも score は悪い

この点はかなり大事で、

**今回の差は単なる PPO の update 爆発ではなく、horizon が actor を押す方向そのものに効いている**

と読む方が自然である。

## 8. shanten advantage の癖

final の `shanten_diag.advantage.mean`:

| 条件 | same | improve | worsen |
|---|---:|---:|---:|
| A `g050_gae000` | `+0.0382` | `-0.1209` | `-0.0112` |
| B `g050_gae030` | `+0.0406` | `-0.1283` | `-0.0121` |
| C `g050_gae060` | `+0.0426` | `-0.1394` | `-0.0012` |
| D `g075_gae000` | `+0.0396` | `-0.1209` | `-0.0226` |
| E `g075_gae030` | `+0.0440` | `-0.1375` | `-0.0175` |
| F `g075_gae060` | `+0.0477` | `-0.1505` | `-0.0153` |

今回も以前からの違和感は残っている。

- `same` は一貫して正
- `improve` は一貫して強く負
- `worsen` は負だが、`improve` ほど悪くない

しかも `gae` を上げるほど
- `same` はさらに正に寄り
- `improve` はさらに負に寄る
傾向がある。

したがって、

**`gae=0.3` は性能面では改善するが、advantage の順位づけの違和感そのものを消してはいない**

と言える。

これは次段の論点として残る。

## 9. 解釈

今回の結果を一番自然に言い換えると、こうなる。

### 1. post-bugfix baseline では `gae=0.0` が短すぎた可能性が高い
- current baseline は悪くなかった
- ただし `gae=0.3` にすると final score と drawdown が両方改善した

### 2. しかし `gae=0.6` は長すぎる
- peak は高い
- でも保持できない
- 数値的には穏やかでも、改善方向が悪い

### 3. `gamma=0.75` は単独で常に良いわけではない
- `gae=0.3` と組み合わせたときにだけ本領を発揮している

つまり今回の sweep は、

**「もっと長く見るべきか」ではなく、「少しだけ長く見るのが良い」**

という結果に近い。

## 10. 暫定判断

現時点の暫定判断は次のとおりである。

1. **current baseline (`gamma=0.50, gae=0.0`) は更新候補**
2. 次の暫定 PPO baseline は
   - **`gamma=0.75, gae_lambda=0.3`**
   が第一候補
3. 第二候補は
   - **`gamma=0.50, gae_lambda=0.3`**
4. `gae=0.6` は当面切ってよい

実務的には、

**次の実験からは `gamma=0.75, gae=0.3` を新 baseline として使う**

のが一番自然である。

## 11. 次にやること

今回の結果で horizon 周りはかなり整理できた。

自然な次の選択肢は次の 2 つである。

1. **baseline config を `gamma=0.75, gae=0.3` に更新する**
   - 以後の rule-only PPO 比較の土台を更新する
2. **その baseline 上で次の本題に進む**
   - `value_loss_coef`
   - `policy_ratio`
   - sample weighting / advantage quality

いまの流れでは、まず 1 をやってから 2 に進むのがきれいである。

## 12. 参考: `g090_gae030` の単発確認

`gamma=0.90, gae_lambda=0.3` については、pre-bugfix では高 gamma を一度棄却していたが、bugfix 後の環境で挙動が変わっている可能性を見て **1 seed の exploratory run** を追加で確認した。

補足:
- `g090_gae030` の単発 run は `runs/` 配下のローカル成果物として確認した

結果:
- final `avg_rank = 2.3925`
- final `avg_score = 1389.25`
- best cycle `= 1`
- best `avg_score = 2831.0`
- drawdown `= 1441.75`

比較対象として `g075_gae030` の seed 42 は:
- final `avg_score = 2385.5`
- best cycle `= 8`
- drawdown `= 301.25`

この 1 seed から読む限り、
- `gamma=0.90` は序盤で一度強く伸びる
- しかし peak が極端に早く、その後の保持に失敗する
- `clip_fraction` / `ratio_std` はむしろ穏やかなので、数値爆発というより **長い horizon が policy を押す方向そのものがあまり良くない** 可能性が高い

したがって現時点では、
- `g090_gae030` は **参考観測に留める**
- main baseline 候補は引き続き **`g075_gae030`**

と整理するのが妥当である。
