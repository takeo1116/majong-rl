# Experiment Report: exp_058

作成日: 2026-03-17  
対象: `experiments/exp_058/runbook.md`  
目的: `imitation-only` 条件で旧モデルと `policy_direct_hints + context_gate` 新モデルを `3 seeds` で直接比較し、architecture 変更そのものが teacher 模倣と対戦性能にどれだけ効くかを確認する

## 1. 実験概要

- 条件: 2条件
- seeds: `42..44`（3 seeds）
- imitation matches: `10000`
- eval: `rotation, num_matches=100`
- 共通条件:
  - `experiment.phases=["imitation","selfplay","eval"]`
  - `selfplay.num_matches=0`
  - `feature_encoder.shanten_hint.enabled=true`
  - `feature_encoder.discard_ukeire_hint.enabled=true`
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `training.imitation_epochs=8`

条件一覧:

| 条件 | モデル |
|---|---|
| A `old_model_imitation10000` | `policy_direct_hints.enabled=false` |
| B `new_model_direct_hints_imitation10000` | `policy_direct_hints.enabled=true` + `sources=["shanten_hint","discard_ukeire_hint"]` + `context_gate.enabled=true` |

補足:
- 新モデル条件では `shanten_hint` / `discard_ukeire_hint` は shared trunk から除外され、policy direct branch のみに入る。
- 今回は PPO を入れず、architecture の imitation 効果だけを clean に見ている。

## 2. 実行結果

- 成功条件数: `2/2`
- 失敗条件数: `0`

参照:
- 旧モデル: （ローカル成果物）
- 新モデル: （ローカル成果物）

## 3. seed ごとの結果

### 旧モデル

| seed | teacher_top1 | teacher_best_set | value_loss | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | `0.3625` | `0.8669` | `0.03963` | `3.1425` | `-9352.0` | `0.1356` | `0.5529` |
| 43 | `0.3603` | `0.8672` | `0.03965` | `3.0900` | `-9114.5` | `0.1418` | `0.5556` |
| 44 | `0.3634` | `0.8631` | `0.03956` | `3.0975` | `-8716.0` | `0.1343` | `0.5498` |

### 新モデル

| seed | teacher_top1 | teacher_best_set | value_loss | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | `0.4142` | `0.8977` | `0.03929` | `2.9225` | `-6723.0` | `0.1784` | `0.5279` |
| 43 | `0.4540` | `0.9028` | `0.03942` | `3.0500` | `-8432.5` | `0.1612` | `0.5442` |
| 44 | `0.4507` | `0.9087` | `0.03933` | `3.0100` | `-8087.5` | `0.1660` | `0.5405` |

## 4. 平均比較

| 指標 | 旧モデル | 新モデル | 差分 (新-旧) |
|---|---:|---:|---:|
| `teacher_top1_match_rate` | `0.3621` | `0.4397` | `+0.0776` |
| `teacher_best_set_hit_rate` | `0.8658` | `0.9031` | `+0.0373` |
| `value_loss` | `0.03961` | `0.03935` | `-0.00027` |
| `avg_rank` | `3.1100` | `2.9942` | `-0.1158` |
| `avg_score` | `-9060.8` | `-7747.7` | `+1313.2` |
| `win_rate` | `0.1372` | `0.1685` | `+0.0313` |
| `deal_in_rate` | `0.5528` | `0.5375` | `-0.0152` |

所見:
- 新モデルは **teacher 指標も eval 指標も両方改善**。
- `avg_score` の改善幅 `+1313` は十分大きい。
- `avg_rank`、`win_rate`、`deal_in_rate` もすべて良い方向。

## 5. 差の向き

新モデルが旧モデルより改善した seed 数:

| 指標 | 改善 seed 数 |
|---|---:|
| `teacher_top1_match_rate` | `3/3` |
| `teacher_best_set_hit_rate` | `3/3` |
| `avg_rank` | `3/3` |
| `avg_score` | `3/3` |
| `win_rate` | `3/3` |
| `deal_in_rate` | `3/3` |

所見:
- 単なる当たり seed ではなく、**差の向きが全指標で揃っている**。
- architecture 変更の効果は再現していると見てよい。

## 6. 単発確認との関係

`exp_058` 前の単発比較では、
- teacher 指標の差は小さい
- しかし eval はかなり良い
という見え方だった。

今回 `3 seeds x imitation_matches=10000` で見ると、
- teacher 指標も明確に改善
- eval も明確に改善
に変わった。

つまり前回の単発結果は
- 「architecture は効いているが teacher 指標までは見えにくい」
という暫定解釈だったが、
今回の結果により
- **architecture は imitation 段階の teacher 模倣そのものも改善している**
と整理できる。

## 7. 解釈

今回かなり強く言えること:

1. `shanten_hint` / `discard_ukeire_hint` を shared trunk に混ぜるより、policy logits 直前の牌別 branch で扱う方が良い。  
2. `policy_direct_hints + context_gate` は、少なくとも imitation 段階では明確な改善を出す。  
3. したがって、以前から疑っていた  
   - `shanten_hint` が trunk の中で潰れている  
   - 牌別ヒントが牌別 logits へ届きにくい  
   という仮説はかなり支持された。  
4. `value_loss` はほぼ同等で、改善は主に policy 側から来ていると見てよい。  

## 8. 結論

1. `exp_058` により、`policy_direct_hints + context_gate` 新モデルは imitation で **旧モデルを一貫して上回る** ことが確認できた。  
2. 改善は eval だけでなく、`teacher_top1_match_rate` と `teacher_best_set_hit_rate` にも現れている。  
3. したがって、今後の PPO / mixed PPO 比較は **新モデルを本命** として進める価値が高い。  
4. ここから先の主な問いは  
   - この強い imitation 初期値を PPO がどこまで活かせるか  
   - それでも残る plateau / 戻りは RL target 側の問題なのか  
   に移る。
