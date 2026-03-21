# Experiment Report: exp_062

作成日: 2026-03-18  
対象: bugfix 後の新モデルを前提に、`rule-only PPO + policy_anchor` を固定し、`policy_anchor.coef` を `0.25 / 0.50 / 0.75` で比較した。

- A: `anchor025`
- B: `anchor050`
- C: `anchor075`

参照:
- `experiments/exp_062/runbook.md`
- `experiments/exp_062/run_map.json`

補足:
- 実行時の `batch_summary.json` や `runs/` 配下のローカル成果物は、VCS に載せない前提のため本 report からは参照しない

## 1. 結論

一番自然な読みは、**`policy_anchor.coef=0.50` が現時点では最もバランスが良い** である。

- `0.25` は peak 自体は高いが、後半の戻りが大きい
- `0.75` は teacher らしさの保持は最も強いが、`avg_score` の着地は `0.50` に届かない
- `0.50` は
  - peak が十分高く
  - 戻り幅が最も小さく
  - final `avg_score` が最良
  だった

したがって今の暫定結論は、

1. `0.25` は drift 抑制として弱い  
2. `0.75` は少し縛りすぎている可能性がある  
3. **`0.50` が、改善と保持のトレードオフ上もっとも良い暫定点**

である。

## 2. 実験条件

共通:
- 新モデル (`policy_direct_hints + context_gate`)
- `training.imitation_loss_mode=tie_aware_best_set`
- imitation `1000 matches x 3 chunks`
- PPO `200 matches x 30 cycles`
- `gamma=0.50`
- `gae_lambda=0.0`
- `reward.shaping.shanten_delta.scale=0.003`
- `training.rule_mix.policy_ratio=0.0`
- seeds `42,43,44`

差分:
- A: `policy_anchor.coef=0.25`
- B: `policy_anchor.coef=0.50`
- C: `policy_anchor.coef=0.75`

## 3. Warmstart は共通

3 条件とも imitation は同一設定で、指標も一致している。

- `teacher_top1_match_rate ≈ 0.4334`
- `teacher_best_set_hit_rate ≈ 0.9436`
- `value_loss ≈ 0.00633`

つまり今回の差は、ほぼ **PPO 中の anchor 強度差** とみてよい。

## 4. 最終結果

| 条件 | final avg_rank | final avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A `anchor025` | `2.4125` | `1283.8` | `0.2874` | `0.4569` |
| B `anchor050` | `2.3583` | `1943.0` | `0.2914` | `0.4539` |
| C `anchor075` | `2.3558` | `1742.3` | `0.2944` | `0.4552` |

読み方:
- **`avg_score` は B が最良**
- `avg_rank` と `win_rate` は B/C がほぼ同格
- `deal_in_rate` は B が最良

したがって、final の総合バランスは **B `anchor050` が最も良い** と読むのが自然である。

ただし B と C の差は極端には大きくない。
特に `avg_rank` は

- B: `2.3583`
- C: `2.3558`

でほぼ同格であり、**`0.75` が完全に悪いとは言わない**。  
今回強く言えるのは、`0.25` が明確に弱いことと、`0.50` が score 観点で最良だったことである。

## 5. Peak と保持

| 条件 | best cycle mean | best avg_score mean | final avg_score | best→final drawdown |
|---|---:|---:|---:|---:|
| A `anchor025` | `11.0` | `2987.5` | `1283.8` | `1703.8` |
| B `anchor050` | `6.0` | `2982.0` | `1943.0` | `1039.0` |
| C `anchor075` | `8.7` | `2902.3` | `1742.3` | `1160.0` |

ここが今回の一番重要な比較である。

### A: `anchor025`
- peak は高い
- しかし final までの落ち幅が最も大きい
- `best→final = -1703.8`

つまり、**改善する力はあるが保持できない**。

### B: `anchor050`
- peak は A とほぼ同等
- そのうえで戻り幅が最小
- `best→final = -1039.0`

つまり、**peak を高く作りつつ、最もよく保持できている**。

### C: `anchor075`
- 戻り幅は A より明らかに小さい
- ただし B よりは少し大きい
- best も B よりわずかに低い

つまり、**保持は強いが、そのぶん改善も少し止めている** ように見える。

## 6. 初期値からの改善幅

aggregate の `eval_before.avg_score` と final を比べると、

| 条件 | cycle0 before avg_score | final avg_score | net |
|---|---:|---:|---:|
| A `anchor025` | `1151.0` | `1283.8` | `+132.8` |
| B `anchor050` | `1916.1` | `1943.0` | `+26.9` |
| C `anchor075` | `1755.8` | `1742.3` | `-13.5` |

この表だけを見ると B/C の差は小さく見えるが、これは **B/C ともに一度大きく伸びて、その後少し戻っている** からである。

重要なのは net というより、

- どこまで伸びたか
- そこからどれだけ戻ったか

の方であり、その観点では B が最もバランスが良い。

## 7. teacher 診断

cycle 0 after:

| 条件 | action_match_after | best_set_hit_after |
|---|---:|---:|
| A `anchor025` | `0.4279` | `0.9356` |
| B `anchor050` | `0.4283` | `0.9360` |
| C `anchor075` | `0.4276` | `0.9362` |

cycle 29 after:

| 条件 | action_match_after | best_set_hit_after |
|---|---:|---:|
| A `anchor025` | `0.4094` | `0.8999` |
| B `anchor050` | `0.4070` | `0.9096` |
| C `anchor075` | `0.4123` | `0.9169` |

読み方:
- anchor を強くするほど、teacher らしさの保持は強い
- 特に `best_set_hit_after` は `C > B > A`

ただし、**teacher らしさの保持がそのまま score 最良にはなっていない**。

ここはかなり示唆的で、

- `0.25`: teacher から離れすぎる
- `0.75`: teacher を守りすぎる
- `0.50`: その中間で、実戦スコアのバランスが最も良い

という見え方になる。

## 8. learner 診断

aggregate learner 診断:

| 条件 | ratio_std | clip_fraction | value_error_mean |
|---|---:|---:|---:|
| A `anchor025` | `0.1408` | `0.2286` | `0.00245` |
| B `anchor050` | `0.1265` | `0.1954` | `0.00249` |
| C `anchor075` | `0.1172` | `0.1735` | `0.00425` |

読み方:
- anchor を強くするほど `ratio_std` と `clip_fraction` は穏やかになる
- つまり **数値的 drift 抑制** の効果は確かにある

しかし、
- 最も穏やかな C が最良 score ではない

ため、単純に「より強く縛るほど良い」ではない。

ここからも、
**`0.75` は stabilizer としては強いが、改善まで少し止めている**
という解釈が自然である。

## 9. seed ごとの様子

final `avg_score`:

- A: `793.5`, `1784.0`, `1273.75`
- B: `2166.0`, `2217.75`, `1445.25`
- C: `1144.0`, `1614.75`, `2468.25`

見え方:
- A は 3 seeds とも B に負けている
- B は 2/3 seeds で最良
- C は seed 44 では最良だが、seed 42 でかなり低い

つまり、
- `0.75` は刺さる seed もある
- ただし安定感まで含めると、**まだ `0.50` の方が採用しやすい**

と言える。

## 10. 解釈

今回の sweep から一番自然な暫定結論は次の通り。

### 1. `0.25` は弱い
- peak は作れても保持できない
- drift 抑制として不足している

### 2. `0.75` は少し強い
- teacher らしさは最も保つ
- しかし score 最良にはならない
- 改善まで少し縛っている可能性が高い

### 3. `0.50` は現時点の最良バランス
- peak が十分高い
- 戻り幅が最小
- final `avg_score` が最良
- `deal_in_rate` も最良

したがって、**次の rule-only PPO 系の基準値は `policy_anchor.coef=0.50` を維持する** のが妥当である。

## 11. 次にやること

今回の結果で、anchor 強度そのものはかなり絞れた。

次の優先順位は、

1. `policy_anchor.coef=0.50` を固定して続ける
2. その上で、**rule baseline サンプルの中で何を強く学ぶか** を見る
   - 例: `advantage > 0` の baseline sample のみ使う
   - 例: baseline sample を `relu(advantage)` で重み付けする
3. `batch_size` は二次要因として、その後に見る

今回の結果からは、
**当面は `rule-only + anchor(0.50)` を主診断系にして、sample weighting 側へ進む** のが自然である。
