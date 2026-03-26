# Experiment Report: exp_059

作成日: 2026-03-17  
対象: `experiments/exp_059/runbook.md`  
目的: `multi_chunk_imitation` を用いて `1000 matches x 50 chunks x 1 seed` の長い imitation-only 学習を行い、旧モデルと `policy_direct_hints + context_gate` 新モデルの ceiling と学習曲線を比較する

## 1. 実験概要

- 条件: 2条件
- seed: `42`
- imitation: `1000 matches x 50 chunks = total 50000 matches`
- eval: `rotation, num_matches=100`
- 共通条件:
  - `experiment.phases=["imitation","selfplay","eval"]`
  - `training.multi_chunk_imitation.enabled=true`
  - `training.multi_chunk_imitation.num_chunks=50`
  - `training.multi_chunk_imitation.imitation_matches_per_chunk=1000`
  - `selfplay.num_matches=0`
  - `feature_encoder.shanten_hint.enabled=true`
  - `feature_encoder.discard_ukeire_hint.enabled=true`
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `training.imitation_epochs=8`

条件一覧:

| 条件 | モデル |
|---|---|
| A `old_model_multichunk1000x50` | `policy_direct_hints.enabled=false` |
| B `new_model_direct_hints_multichunk1000x50` | `policy_direct_hints.enabled=true` + `sources=["shanten_hint","discard_ukeire_hint"]` + `context_gate.enabled=true` |

補足:
- 新モデル条件では `shanten_hint` / `discard_ukeire_hint` は shared trunk から除外され、policy direct branch のみに入る。
- 今回は architecture の ceiling を clean に見るため、PPO は入れていない。

## 2. 実行結果

- 成功条件数: `2/2`
- 失敗条件数: `0`

参照:
- 旧モデル: （ローカル成果物）
- 新モデル: （ローカル成果物）

## 3. 最終結果

| 指標 | 旧モデル | 新モデル | 差分 (新-旧) |
|---|---:|---:|---:|
| `teacher_top1_match_rate` | `0.4193` | `0.4732` | `+0.0538` |
| `teacher_best_set_hit_rate` | `0.9214` | `0.9377` | `+0.0163` |
| `value_loss` | `0.04027` | `0.03976` | `-0.00051` |
| `avg_rank` | `3.0375` | `2.9775` | `-0.0600` |
| `avg_score` | `-7999.25` | `-6959.25` | `+1040.0` |
| `win_rate` | `0.1628` | `0.1724` | `+0.0097` |
| `deal_in_rate` | `0.5487` | `0.5281` | `-0.0206` |

所見:
- **50 chunk 後でも新モデルが旧モデルを上回る。**
- teacher 指標も eval 指標も両方改善している。
- したがって、`policy_direct_hints + context_gate` は long-run imitation でも ceiling を押し上げていると見てよい。

## 4. chunk 推移

節目の chunk で teacher 指標を抜くとこうなる。

### 旧モデル

| chunk | top1 | best_set_hit | value_loss |
|---|---:|---:|---:|
| 0 | `0.3060` | `0.8270` | `0.03251` |
| 9 | `0.3831` | `0.8952` | `0.03923` |
| 19 | `0.3868` | `0.9034` | `0.03871` |
| 29 | `0.4064` | `0.9136` | `0.03938` |
| 39 | `0.4087` | `0.9179` | `0.03983` |
| 49 | `0.4193` | `0.9214` | `0.04027` |

### 新モデル

| chunk | top1 | best_set_hit | value_loss |
|---|---:|---:|---:|
| 0 | `0.3020` | `0.8039` | `0.02904` |
| 9 | `0.4489` | `0.9208` | `0.03864` |
| 19 | `0.4784` | `0.9293` | `0.03840` |
| 29 | `0.4971` | `0.9351` | `0.03887` |
| 39 | `0.4827` | `0.9371` | `0.03918` |
| 49 | `0.4732` | `0.9377` | `0.03976` |

所見:
- 新モデルは立ち上がりがかなり速い。
  - `chunk 9` で既に `top1=0.4489`, `best_set_hit=0.9208`
  - 旧モデル final にかなり近い水準まで早く到達している
- 旧モデルは後半もじわじわ伸び続ける。
- 新モデルは `20〜30 chunk` 付近でかなり飽和に近づき、その後は伸びが鈍い。

## 5. peak と飽和

peak chunk:

| 条件 | `teacher_top1` peak | `teacher_best_set_hit` peak |
|---|---:|---:|
| 旧モデル | `chunk 32, 0.4227` | `chunk 49, 0.9214` |
| 新モデル | `chunk 37, 0.5006` | `chunk 47, 0.9385` |

所見:
- 新モデルは peak 自体が高い。
- ただし `teacher_top1_match_rate` は `chunk 37` で peak を打ち、その後はやや戻る。
- `teacher_best_set_hit_rate` は終盤まで高止まりしているが、`chunk 30` 以降の増分は小さい。

## 6. `exp_058` との比較

`exp_058` の `imitation_matches=10000` 単発 imitation と比較すると:

### 旧モデル: 10k -> 50k

- `avg_rank`: `3.1425 -> 3.0375` (`-0.1050`)
- `avg_score`: `-9352.0 -> -7999.25` (`+1352.75`)
- `teacher_top1`: `0.3625 -> 0.4193` (`+0.0568`)
- `teacher_best_set_hit`: `0.8669 -> 0.9214` (`+0.0545`)

### 新モデル: 10k -> 50k

- `avg_rank`: `2.9225 -> 2.9775` (`+0.0550`)
- `avg_score`: `-6723.0 -> -6959.25` (`-236.25`)
- `teacher_top1`: `0.4142 -> 0.4732` (`+0.0590`)
- `teacher_best_set_hit`: `0.8977 -> 0.9377` (`+0.0399`)

所見:
- 旧モデルは `10k -> 50k` でまだかなり伸びる。
- 新モデルは teacher 指標はまだ上がる一方で、eval は seed 42 ではほぼ頭打ち圏に入っている。
- つまり新モデルは
  - 「teacher を再現する能力」はまだ伸びる
  - しかしその伸びが実戦性能にはそのまま変換されない
という形になっている。

## 7. 解釈

今回かなり強く言えること:

1. `policy_direct_hints + context_gate` は long-run imitation でも旧モデルより高い ceiling を持つ。  
2. ただし、その改善幅は有限であり、`avg_score=0` には全然届かない。  
3. 特に重要なのは、**`teacher_best_set_hit_rate ≈ 0.94` まで行っても `avg_score ≈ -6959` に留まる**こと。  
4. これは、architecture 側のボトルネックは一部解けた一方で、  
   - `best_set` という教師定義  
   - `tie_aware_best_set` loss  
   - teacher 指標と実戦性能のギャップ  
   が依然として大きいことを示している。  

## 8. 結論

1. `exp_059` により、新モデルは imitation の long-run ceiling でも旧モデルを上回ることが確認できた。  
2. ただし新モデルも `20〜30 chunk` 付近でかなり飽和し、`avg_score=0` にはまだ大きく届かない。  
3. したがって、次の本丸は architecture そのものより  
   - teacher 定義の見直し  
   - imitation objective の見直し  
   - `best_set` 指標と実戦性能のズレの切り分け  
   に移っている。  
4. 少なくとも、「hint を policy logits 直前で牌別に扱う」方向自体は有効であり、今後の主系列モデルとして扱う価値は高い。  
