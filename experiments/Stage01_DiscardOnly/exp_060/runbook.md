# Experiment Runbook: exp_060

作成日: 2026-03-18  
目的: CQ-0208 修正後の `full` 観測補助特徴バグが解消された条件で、`imitation 1000 x 10 chunks` の短尺 A/B 比較を行い、旧モデルと `policy_direct_hints + context_gate` 新モデルの性能差を再確認する。

## 1. 背景

`exp_058` / `exp_059` の解釈中に、`observation_mode=full` の `FlatFeatureEncoder` が

- `shanten_hint`
- `discard_ukeire_hint`
- `current_shanten`
- `shape_hint`

を `current_player` ではなく `player 0` の手牌から生成していたことが判明した。  
この不具合は CQ-0208 で修正済みであり、少なくとも bugfix 後に imitation-only 比較を取り直す必要がある。

今回はまず軽量な再取得として、

- `1000 matches x 10 chunks`
- `1 seed`
- old/new 2条件

の A/B を行う。

## 2. 実験の問い

1. bugfix 後でも `policy_direct_hints + context_gate` 新モデルは旧モデルを上回るか  
2. 旧モデルも bugfix により大きく改善し、差が主に学習速度に縮むのか  
3. 10 chunks 時点で、A/B のどちらが次の長尺比較（`exp_059` 相当やり直し）に進む価値が高いか

## 3. 条件

- seed: `42`
- imitation:
  - `training.multi_chunk_imitation.enabled=true`
  - `training.multi_chunk_imitation.num_chunks=10`
  - `training.multi_chunk_imitation.imitation_matches_per_chunk=1000`
  - total imitation matches = `10000`
- eval:
  - `rotation`
  - `num_matches=100`
- phases:
  - `["imitation","selfplay","eval"]`
  - `selfplay.num_matches=0`

### 条件 A: 旧モデル

- `model.policy_direct_hints.enabled=false`

### 条件 B: 新モデル

- `model.policy_direct_hints.enabled=true`
- `model.policy_direct_hints.sources=["shanten_hint","discard_ukeire_hint"]`
- `model.policy_direct_hints.local_hidden_dim=16`
- `model.policy_direct_hints.tile_embedding_dim=4`
- `model.policy_direct_hints.context_gate.enabled=true`

## 4. 共通固定条件

- `observation_mode=full`
- `feature_encoder.shanten_hint.enabled=true`
- `feature_encoder.discard_ukeire_hint.enabled=true`
- `feature_encoder.current_shanten.enabled=true`
- `feature_encoder.shape_hint.enabled=true`
- `feature_encoder.turn_context.enabled=true`
- `training.imitation_loss_mode=tie_aware_best_set`
- `training.imitation_value_warmstart.enabled=true`
- `training.imitation_value_warmstart.coef=0.3`
- `training.exclude_post_riichi_discards.enabled=true`
- `training.value_loss.type=mse`
- `training.imitation_epochs=8`
- `selfplay.policy_ratio=1.0`
- `selfplay.save_baseline_actions=false`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

## 5. 主評価指標

- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

補助観点:

- `chunks[*].teacher_top1_match_rate`
- `chunks[*].teacher_best_set_hit_rate`
- `value_loss`

## 6. 期待する読み方

### ケース 1

- 新モデルが 10 chunks 時点でも明確に上回る  

解釈:
- bugfix 後でも architecture 改善は本物
- `exp_059` 相当の long-run 比較を新旧で再取得する価値が高い

### ケース 2

- 旧モデルが大きく回復して新モデルとの差がかなり縮む  

解釈:
- 以前見えていた差の一部は bug の影響だった
- 次は「速度差なのか ceiling 差なのか」を長尺実験で見ればよい

### ケース 3

- 旧モデルも新モデルも想定以上に強い  

解釈:
- bug が主要因だった可能性が高い
- imitation objective / long-run ceiling の議論を新しい baseline でやり直す

## 7. 成功条件

- 条件数 `2/2` 完走
- `failed == 0`
- 各 run で
  - `phase_status.imitation == "success"`
  - `phase_status.selfplay == "success"`
  - `phase_status.eval == "success"`
- 各 run で
  - `multi_chunk_imitation.enabled == true`
  - `num_chunks == 10`
  - `len(chunks) == 10`
  - `sum(chunks[*].num_matches) == 10000`

## 8. 実行後にやること

1. 条件 A の結果を取得する  
2. 条件 B と比較して `report.md` を作る  
3. 差が十分大きければ `exp_059` 相当の long-run 比較を bugfix 後条件で再実施する
