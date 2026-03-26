# Experiment Report: exp_066

作成日: 2026-03-22  
対象: CQ-0210 / CQ-0211 修正後の new semantics 前提で、`gamma` / `gae_lambda` を短い側に寄せた horizon re-centering pilot を 1 seed で確認する。

条件:
- A: `g050_gae000`
- B: `g050_gae030`
- C: `g065_gae000`
- D: `g065_gae030`
- E: `g075_gae030`（修正後参照点）

参照:
- `experiments/exp_066/runbook.md`
- `experiments/exp_066/run_map.json`
- `experiments/exp_065_bugfix/bug_report.md`

補足:
- 実行時の `batch_summary.json` や `runs/` 配下のローカル成果物は、VCS に載せない前提のため本 report からは参照しない
- 本 report は **途中経過** である
- `A/B` は完了しているが、`C/D` は途中で中断したため、本 report では **A/B と修正後参照点 E の比較** のみを扱う

## 1. 結論

A/B までの結果から、現時点で安全に言えるのは次の 4 点である。

1. **修正後 semantics で `gamma=0.50` は短すぎる寄りに見える**
   - `A/B` とも imitation 直後の score は高い
   - しかし final は imitation 直後を大きく下回る

2. **`gae_lambda=0.3` は `0.0` より少し良い**
   - `B` は `A` より final `avg_score` が上
   - `final - imitation_initial` の悪化幅も `B` の方が小さい
   - ただし差は限定的で、`gamma=0.50` 自体を救うほどではない

3. **学習信号の向きは引き続き自然**
   - `A/B` でも `improve advantage > 0`, `same advantage < 0` を維持できている
   - CQ-0210 / CQ-0211 修正後の signal 健全化は崩れていない

4. **次の本命は `gamma=0.65` 以上の帯**
   - `0.50` まで戻すのは戻しすぎに見える
   - `C/D` を見切るか、必要なら `0.90` を単発で見る方が情報量が高い

## 2. 実験条件

共通:
- 新モデル (`policy_direct_hints + context_gate`)
- `rule-only PPO`
- `policy_anchor.coef=0.5`
- `clip_epsilon=0.15`
- `value_loss_coef=0.25`
- imitation `1000 matches x 3 chunks`
- PPO `200 matches x 30 cycles`
- seed `42`

今回扱う条件:
- A: `gamma=0.50, gae_lambda=0.0`
- B: `gamma=0.50, gae_lambda=0.3`
- E: `gamma=0.75, gae_lambda=0.3`（修正後の参照点）

未完了:
- C: `gamma=0.65, gae_lambda=0.0`
- D: `gamma=0.65, gae_lambda=0.3`

## 3. 結果

| 条件 | gamma | gae | init avg_score | best cycle | best avg_score | final avg_score | final avg_rank | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A `g050_gae000` | `0.50` | `0.0` | `2748.75` | `14` | `2793.5` | `1216.75` | `2.4175` | `0.2801` | `0.4659` |
| B `g050_gae030` | `0.50` | `0.3` | `2600.25` | `14` | `2962.5` | `1312.75` | `2.4125` | `0.2798` | `0.4622` |
| E `g075_gae030` | `0.75` | `0.3` | `2175.75` | `24` | `2547.25` | `1249.5` | `2.4250` | `0.2799` | `0.4618` |

補足:
- `init avg_score` は `cycle0.eval_before.avg_score` を指す
- `E` は `exp_066` の新規条件ではなく、**修正後 baseline の参照点**である

## 4. A/B の比較

### final 指標
- A `g050_gae000`: `1216.75`
- B `g050_gae030`: `1312.75`

`B > A` であり、`gae=0.3` の方が少し良い。

### imitation 直後との差
- A: `1216.75 - 2748.75 = -1532.0`
- B: `1312.75 - 2600.25 = -1287.5`

どちらも **final が imitation 直後を大きく下回る**。  
したがって、修正後 semantics 前提では `gamma=0.50` は短すぎる寄りと読むのが自然である。

### peak と保持
- A drawdown: `1576.75`
- B drawdown: `1649.75`

`B` は final では少し良いが、drawdown 自体は小さくない。  
つまり、`gae=0.3` は `0.50` 帯を少し改善するが、保持問題そのものを解消してはいない。

## 5. 参照点 E との比較

修正後参照点 `E g075_gae030` と比べると:

- A final: `1216.75`
- B final: `1312.75`
- E final: `1249.5`

この 1 seed だけを見ると、`B` は `E` より少し上、`A` は `E` より少し下である。  
ただしここで大事なのは absolute final よりも **挙動の形** である。

- `A/B` は imitation 直後が高いが、その後の悪化が大きい
- `E` は imitation 直後がやや低い一方、`final - init` の悪化幅は比較的小さい

よって、いまの論点は

- `0.50` が強いか

ではなく、

- **修正後 semantics では `0.50` は短すぎて、長めの horizon の方が安定するのではないか**

である。

## 6. メトリクスの自然さ

final `learner_diag` の代表値:

| 条件 | clip_fraction | ratio_std | value_error_mean | best_set_hit_after |
|---|---:|---:|---:|---:|
| A `g050_gae000` | `0.1938` | `0.1295` | `0.00960` | `0.9028` |
| B `g050_gae030` | `0.1885` | `0.1258` | `0.01124` | `0.9011` |
| E `g075_gae030` | `0.1912` | `0.1236` | `0.00806` | `0.9003` |

この範囲では、
- clip / ratio が爆発しているわけではない
- teacher rail も大きく壊れていない

したがって、A/B の弱さは **数値的不安定** というより、**horizon が短すぎることによる target の短期化** と解釈する方が自然である。

## 7. shanten signal の向き

final `shanten_diag.advantage.mean`:

| 条件 | improve | same | worsen |
|---|---:|---:|---:|
| A `g050_gae000` | `+0.0965` | `-0.0308` | `+0.0050` |
| B `g050_gae030` | `+0.0945` | `-0.0307` | `+0.0098` |
| E `g075_gae030` | `+0.1050` | `-0.0318` | `-0.0087` |

ここはかなり重要である。

- `improve > 0`
- `same < 0`

は A/B でも維持されている。  
つまり、CQ-0210 / CQ-0211 修正後の **signal 健全化は崩れていない**。

一方で、`A/B` では `worsen advantage` がわずかに正で、`E` より不自然である。  
これは、`gamma=0.50` が短すぎて **worsen を十分に罰せていない** 可能性を示唆する。

## 8. 現時点の解釈

A/B だけからの暫定解釈は次のとおり。

1. 修正後 semantics では、`gamma=0.50` は短すぎる寄り
2. `gae=0.3` は `0.0` より少し良いが、主因ではない
3. `improve/same/worsen` の向きは自然なまま保たれている
4. 次に見るべきは、`0.50` より上の帯
   - 本来は `C/D` の `gamma=0.65`
   - もしくは単発 exploratory として `gamma=0.90`

## 9. 次の方針

今回の pilot は途中中断のため、exp_066 から確定結論を出す段階ではない。  
ただし、A/B の途中結果だけでも次の優先順位はかなり明確である。

1. `C/D` を最後まで見て `gamma=0.65` 帯を判断する
2. もしくは、今夜の exploratory として
   - `gamma=0.90`
   - `anchor=0.75`
   - `policy_ratio=0.05`
   - `shanten_delta.scale=0.001`
   のような単発条件を広く見る

現時点では、**「修正後 semantics で `gamma=0.50` に戻す」のは本命ではない** という理解でよい。
