# Experiment Report: exp_063

作成日: 2026-03-19  
対象: bugfix 後の新モデルを前提に、`rule-only PPO + policy_anchor(coef=0.5)` を固定し、`clip_epsilon` を `0.10 / 0.15 / 0.20` で比較した。

- A: `clip010`
- B: `clip015`
- C: `clip020`

参照:
- `experiments/exp_063/runbook.md`
- `experiments/exp_063/run_map.json`

補足:
- 実行時の `batch_summary.json` や `runs/` 配下のローカル成果物は、VCS に載せない前提のため本 report からは参照しない

## 1. 結論

今回の sweep から強く言えるのは次の 2 点である。

1. **`clip_epsilon=0.20` は悪い**
   - final `avg_score` が明確に低く
   - `best -> final` drawdown も最大
   - 今の系では update を緩める方向は悪化に繋がる

2. **`0.10` と `0.15` はかなり近い**
   - final `avg_score` は `1943.0` vs `1891.0` で近い
   - ただし性格は異なる
     - `0.10`: peak は遅いが保持が良い
     - `0.15`: peak は高いが戻りやすい

したがって今の暫定結論は、

- `0.20` は切ってよい
- 本命は **`0.10` vs `0.15`**
- そして今回の 3 seeds だけでは、**`0.15` が明確に上とまでは言い切らない**

である。

## 2. 実験条件

共通:
- 新モデル (`policy_direct_hints + context_gate`)
- `training.imitation_loss_mode=tie_aware_best_set`
- imitation `1000 matches x 3 chunks`
- PPO `200 matches x 30 cycles`
- `policy_anchor.coef=0.5`
- `gamma=0.50`
- `gae_lambda=0.0`
- `reward.shaping.shanten_delta.scale=0.003`
- `training.rule_mix.policy_ratio=0.0`
- seeds `42,43,44`

差分:
- A: `training.clip_epsilon=0.10`
- B: `training.clip_epsilon=0.15`
- C: `training.clip_epsilon=0.20`

補足:
- B `clip015` は `experiments/exp_062/report.md` で扱った `anchor050` と同一 batch を再利用した

## 3. Warmstart は共通

3 条件とも imitation warmstart は同一設定であり、今回の差は基本的に **PPO 中の `clip_epsilon` 差** とみてよい。

共通 warmstart 指標:
- `teacher_top1_match_rate ≈ 0.4334`
- `teacher_best_set_hit_rate ≈ 0.9436`
- `value_loss ≈ 0.00633`

## 4. 最終結果

| 条件 | final avg_rank | final avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A `clip010` | `2.3533` | `1891.0` | `0.2953` | `0.4521` |
| B `clip015` | `2.3583` | `1943.0` | `0.2914` | `0.4539` |
| C `clip020` | `2.4100` | `1234.3` | `0.2875` | `0.4580` |

読み方:
- **score 最良は B `clip015`**
- ただし **A `clip010` はかなり近い**
  - 差は `+52` 点しかない
- `avg_rank` はむしろ A がわずかに良い
- C `clip020` は score / rank / deal-in の総合で明確に悪い

したがって、
- `0.20` は外れ
- `0.10` と `0.15` が競っている
という構図である。

## 5. Peak と保持

| 条件 | best cycle mean | best avg_score mean | final avg_score | best→final drawdown |
|---|---:|---:|---:|---:|
| A `clip010` | `19` | `2581.6` | `1891.0` | `690.6` |
| B `clip015` | `8` | `2803.6` | `1943.0` | `860.6` |
| C `clip020` | `5` | `2631.3` | `1234.3` | `1397.0` |

ここが今回の一番重要な比較である。

### A: `clip010`
- peak がかなり遅い
- その代わり drawdown は最小
- つまり **ゆっくり改善して、比較的よく保持する**

### B: `clip015`
- peak は最も高い
- ただし A より戻りは大きい
- つまり **改善力は高いが、保持は A ほどではない**

### C: `clip020`
- peak がかなり早い
- その後の戻りが非常に大きい
- つまり **update を許しすぎて壊している**

この比較から、`clip_epsilon` の役割はかなり明確である。

- 小さくするほど
  - peak は遅くなる
  - しかし保持は良くなる
- 大きくするほど
  - peak は早い
  - しかし drift が悪化する

## 6. learner 診断

final learner 診断:

| 条件 | clip_fraction | ratio_std | action_match_after | best_set_hit_after |
|---|---:|---:|---:|---:|
| A `clip010` | `0.2662` | `0.1067` | `0.4113` | `0.9143` |
| B `clip015` | `0.1954` | `0.1265` | `0.4070` | `0.9096` |
| C `clip020` | `0.1504` | `0.1450` | `0.4041` | `0.9068` |

重要な注意:
- **`clip_fraction` は threshold 依存** なので、条件間で単純比較しにくい
- `0.10` の方が `clip_fraction` が高いのは、
  - update が荒いからというより
  - **閾値が厳しいから当たりやすい**
  面がある

そのため、条件間で step の大きさを見るには **`ratio_std` の方が素直** である。

`ratio_std` は
- A `0.1067`
- B `0.1265`
- C `0.1450`

と綺麗に並んでおり、

**`clip_epsilon` を下げるほど、実際に policy step は小さくなっている**

と読める。

さらに teacher 診断でも
- `best_set_hit_after`: `A > B > C`

なので、`0.10` は safety rail の保持も強い。

## 7. seed ごとの様子

final `avg_score`:
- A: `1689.0`, `1849.5`, `2134.5`
- B: `2166.0`, `2217.75`, `1445.25`
- C: `1184.25`, `1608.0`, `910.75`

読み方:
- C は 3 seeds すべてで弱い
- A と B は seed によって勝ち負けが入れ替わる

つまり、
- `0.20` はかなり自信を持って切れる
- しかし `0.10` と `0.15` の優劣は、まだ少し保留が妥当

## 8. 解釈

今回の結果を一番自然に言い換えると、こうなる。

### 1. `0.20` は update が大きすぎる
- peak は早い
- final までの崩れが大きい
- 現在の rule-only + anchor 系では、緩める方向は悪化する

### 2. `0.10` は保持寄り
- peak は遅い
- drawdown は最小
- teacher らしさの保持も良い

### 3. `0.15` は改善寄り
- peak は最も高い
- final score もわずかに最良
- ただし A よりは戻る

つまり、今回の sweep は

**「`clip_epsilon` を上げるべきか」ではなく、`0.10` と `0.15` のどちらの性格を採るか**

という段階まで論点を絞れた、と言える。

## 9. 暫定判断

現時点では次のように整理するのが妥当である。

1. **`clip020` は採用しない**
2. 暫定基準は **`clip015` のままでもよい**
   - final `avg_score` は最良
3. ただし、
   - peak 保持
   - teacher 保持
   の観点では **`clip010` もかなり有力**

したがって今の safest な言い方は、

**`0.15` が暫定 best だが、`0.10` は keep 候補として十分残る**

である。

## 10. 次にやること

今回の結果で `clip_epsilon` についてはかなり絞れた。

自然な次の選択肢は次の 2 つである。

1. **`clip010` vs `clip015` を追加 seeds で詰める**
   - 近い 2 条件の優劣を確認する
2. **`clip015` を固定して別ノブへ進む**
   - `value_loss_coef`
   - `policy_ratio`
   - あるいは sample weighting / advantage quality

今の evidence だけなら、
- `0.20` は切る
- `0.10` と `0.15` の 2 候補に絞る

という判断が一番きれいである。
