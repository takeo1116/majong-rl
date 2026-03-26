# Experiment Report: exp_060

作成日: 2026-03-18  
対象: `experiments/exp_060/runbook.md`  
目的: CQ-0208 修正後の `full` 観測補助特徴バグが解消された条件で、`imitation 1000 x 10 chunks` の短尺 A/B 比較を行い、旧モデルと `policy_direct_hints + context_gate` 新モデルの性能差を再確認する

## 1. 実験概要

- 条件: 2条件
- seeds: `42`（1 seed）
- imitation:
  - `training.multi_chunk_imitation.enabled=true`
  - `num_chunks=10`
  - `imitation_matches_per_chunk=1000`
  - total imitation matches = `10000`
- eval: `rotation, num_matches=100`
- phases:
  - `["imitation","selfplay","eval"]`
  - `selfplay.num_matches=0`

条件一覧:

| 条件 | モデル |
|---|---|
| A `old_model_multichunk1000x10_bugfix` | `policy_direct_hints.enabled=false` |
| B `new_model_direct_hints_multichunk1000x10_bugfix` | `policy_direct_hints.enabled=true` + `sources=["shanten_hint","discard_ukeire_hint"]` + `context_gate.enabled=true` |

補足:
- 今回は CQ-0208 修正後のコードで取得している。
- `feature_encoder.shanten_hint` / `discard_ukeire_hint` / `current_shanten` / `shape_hint` は、`observation_mode=full` でも `current_player` 基準で生成される。
- 新モデル条件では `shanten_hint` / `discard_ukeire_hint` は shared trunk から除外され、policy direct branch のみに入る。

## 2. 実行結果

- 成功条件数: `2/2`
- 失敗条件数: `0`

参照:
- 旧モデル: （ローカル成果物）
- 新モデル: （ローカル成果物）

## 3. 条件別結果

| 指標 | 旧モデル | 新モデル | 差分 (新-旧) |
|---|---:|---:|---:|
| `teacher_top1_match_rate` | `0.4954` | `0.7007` | `+0.2053` |
| `teacher_best_set_hit_rate` | `0.9938` | `1.0000` | `+0.0062` |
| `value_loss` | `0.03936` | `0.00546` | `-0.03390` |
| `avg_rank` | `2.5075` | `2.4775` | `-0.0300` |
| `avg_score` | `-274.00` | `383.25` | `+657.25` |
| `win_rate` | `0.2739` | `0.2789` | `+0.0050` |
| `deal_in_rate` | `0.4778` | `0.4607` | `-0.0171` |

所見:
- bugfix 後は旧モデルも大きく改善し、`avg_score` はほぼゼロ近辺まで回復した。
- その上でなお、新モデルは teacher 指標・eval 指標の両方で旧モデルを上回った。
- 特に `teacher_top1_match_rate` の差 `+0.2053` は大きい。

## 4. chunk 推移

### 旧モデル

- chunk 0:
  - `teacher_top1 = 0.4383`
  - `teacher_best_set = 0.9648`
- peak:
  - `teacher_top1`: chunk `8`, `0.5040`
  - `teacher_best_set`: chunk `8`, `0.9955`
- final:
  - `teacher_top1 = 0.4954`
  - `teacher_best_set = 0.9938`

### 新モデル

- chunk 0:
  - `teacher_top1 = 0.5320`
  - `teacher_best_set = 0.9999`
- peak:
  - `teacher_top1`: chunk `9`, `0.7007`
  - `teacher_best_set`: chunk `4`, `1.0000`
- final:
  - `teacher_top1 = 0.7007`
  - `teacher_best_set = 1.0000`

所見:
- bugfix 後の新モデルは、最初の chunk から旧モデルよりかなり高い位置にいる。
- `best_set` は新モデルでほぼ即時に取り切れており、その後は主に `top1` を詰める学習になっている。
- 旧モデルも改善はしているが、10 chunks 時点で `top1` と `deal_in_rate` にまだ差が残る。

## 5. bugfix 前との関係

bugfix 前の `exp_059` では、
- 新モデルでも `50 chunks` 後に `teacher_best_set_hit_rate ≈ 0.9377`
- `avg_score ≈ -6959`
に留まっていた。

今回の bugfix 後短尺 run では、
- 新モデル `10 chunks` で `teacher_best_set_hit_rate = 1.0`
- `avg_score = 383.25`
まで到達した。

この差は大きく、以前の imitation ceiling 議論のかなりの部分が
`full` 観測補助特徴バグの影響を受けていたと考えるのが自然である。

## 6. 解釈

今回かなり強く言えること:

1. CQ-0208 で修正した `current_player` 基準バグは、imitation 性能の本丸に近い不具合だった。  
2. bugfix 後は、旧モデルでも teacher 指標と eval が大きく改善する。  
3. その上でなお、新モデルは `teacher_top1_match_rate`、`avg_score`、`deal_in_rate` で明確に上回る。  
4. したがって architecture 改善の効果自体は bugfix 後も残っている。  
5. 以前の「ceiling が -7000 付近」という見え方は、teacher/objective の本質限界ではなく、まず bug に強く支配されていたと整理すべきである。  

## 7. 結論

1. bugfix 後の短尺 A/B でも、新モデルは旧モデルを上回った。  
2. 旧モデルとの差は bugfix で縮まったが、消えてはいない。  
3. したがって `policy_direct_hints + context_gate` は、修正後条件でも有効な architecture 改善と見てよい。  
4. 今後の imitation / PPO 検討は、CQ-0208 修正後の新しい baseline を基準にやり直すべきである。  
