# exp_059 runbook

最終更新: 2026-03-17  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `multi_chunk_imitation` を用いて `1000 matches x 50 chunks x 1 seed` の長い imitation-only 学習を行い、旧モデルと `policy_direct_hints + context_gate` 新モデルの ceiling と学習曲線を比較する

---

## 0. 実験の位置づけ

- 背景:
  - `exp_058` では、`imitation_matches=10000` の単発 imitation-only A/B 比較で、新モデルが旧モデルを `3 seeds` で一貫して上回った。
  - これにより、`policy_direct_hints + context_gate` は少なくとも imitation 段階では本物の改善を出していると見てよい。
  - 次の問いは「その改善がどこまで伸びるか」、つまり **imitation の ceiling** である。
- 問題:
  - `selfplay.imitation_matches` を単発でさらに大きくすると GPU / メモリ負荷が高く、これ以上の拡張が難しい。
  - CQ-0206 / CQ-0207 で `multi_chunk_imitation` が入り、`データ生成 -> imitation 学習` を chunk 単位で繰り返せるようになった。
- 仮説:
  - 新モデルは旧モデルより高い ceiling を持ち、`10000 matches` よりさらに先でも teacher 指標・eval 指標の両方で優位を維持する。
  - もし新旧とも似た位置で頭打ちするなら、architecture 以外の上限要因も疑うべきである。
- 方針:
  - `1000 matches x 50 chunks = total 50000 imitation matches` を 1 seed で回し、旧モデルと新モデルを比較する。
  - 単なる最終値だけでなく、`chunks` に記録される chunk ごとの teacher 指標の推移も読む。

## 1. 条件

- 条件数: 2
- seed: `42`（1 seed）
- chunk 数: `50`
- chunk あたり imitation matches: `1000`
- total imitation matches: `50000`
- eval: `rotation, num_matches=100`

条件一覧:

- A: `old_model_multichunk1000x50`
  - `model.policy_direct_hints.enabled=false`
- B: `new_model_direct_hints_multichunk1000x50`
  - `model.policy_direct_hints.enabled=true`
  - `sources=["shanten_hint","discard_ukeire_hint"]`
  - `context_gate.enabled=true`
  - direct hint source は trunk から除外され、policy direct branch のみに入る

## 2. 共通固定（override）

- `experiment.phases=["imitation","selfplay","eval"]`
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
- `training.policy_anchor.enabled=false`
- `training.entropy_coef=0.0`
- `imitation.num_workers=10`
- `training.imitation_epochs=8`
- `training.multi_chunk_imitation.enabled=true`
- `training.multi_chunk_imitation.num_chunks=50`
- `training.multi_chunk_imitation.imitation_matches_per_chunk=1000`
- `selfplay.num_matches=0`
- `selfplay.num_workers=10`
- `selfplay.policy_ratio=1.0`
- `selfplay.save_baseline_actions=false`
- `evaluation.mode=rotation`
- `evaluation.rotation_seats=[0,1,2,3]`
- `evaluation.num_matches=100`
- `evaluation.num_workers=10`
- `model.hidden_dims=[512,256]`
- `model.policy_tower.enabled=true`
- `model.policy_tower.hidden_dim=128`
- `model.value_tower.enabled=true`
- `model.value_tower.hidden_dim=128`
- `model.value_features.current_shanten.enabled=true`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

条件ごとの追加 override:

- A:
  - `model.policy_direct_hints.enabled=false`
- B:
  - `model.policy_direct_hints.enabled=true`
  - `model.policy_direct_hints.sources=["shanten_hint","discard_ukeire_hint"]`
  - `model.policy_direct_hints.local_hidden_dim=16`
  - `model.policy_direct_hints.tile_embedding_dim=4`
  - `model.policy_direct_hints.context_gate.enabled=true`

## 3. 主評価

1. 最終 imitation 指標
   - `teacher_top1_match_rate`
   - `teacher_best_set_hit_rate`
   - `value_loss`
2. 最終 eval 指標
   - `avg_rank`
   - `avg_score`
   - `win_rate`
   - `deal_in_rate`
3. chunk 推移
   - `chunks[*].teacher_top1_match_rate`
   - `chunks[*].teacher_best_set_hit_rate`
   - `chunks[*].policy_loss`
   - どこで飽和するか
4. ceiling の見方
   - `chunk 10 / 20 / 30 / 40 / 49` の teacher 指標の増分
   - 最終 10 chunks でまだ伸びているか

## 4. 成功判定

- driver 全体で:
  - `completed == 2`
  - `failed == 0`
- 各条件の `batch_summary.json` で:
  - `success_count == 1`, `failure_count == 0`
- 各 run で:
  - `phase_stats.imitation` が存在する
  - `phase_stats.eval` が存在する
  - `phase_stats.imitation.multi_chunk_imitation.enabled == true`
  - `phase_stats.imitation.multi_chunk_imitation.num_chunks == 50`
  - `phase_stats.imitation.multi_chunk_imitation.imitation_matches_per_chunk == 1000`
  - `len(phase_stats.imitation.chunks) == 50`
  - `sum(phase_stats.imitation.chunks[*].num_matches) == 50000`
  - `phase_stats.imitation.num_workers == 10`
  - `phase_stats.imitation.shard_count > 0`
  - `selfplay.total_matches == 0`
- 新モデル条件では:
  - `summary.model_features.policy_direct_hints.enabled == true`
  - `summary.model_features.policy_direct_hints.sources == ["shanten_hint","discard_ukeire_hint"]`

## 5. 判定基準

- 新モデルが ceiling でも有望:
  - 最終 `avg_rank / avg_score` が旧モデルより明確に良い
  - teacher 指標も高い
  - chunk 後半でも旧モデルより高い位置で推移する
- 両者ともまだ伸びる:
  - `chunk 40 -> 49` でも teacher 指標が上がり続ける
  - ceiling 未到達なので、さらに chunk 数を増やす価値がある
- 両者とも飽和:
  - 後半 10 chunks で teacher 指標・eval 指標の改善がほぼ止まる
  - このとき新旧の飽和位置の差が architecture の ceiling 差と見なせる
- 旧モデルと新モデルが同じ位置に寄る:
  - architecture だけでは上限を押し上げ切れない
  - objective / teacher 定義 / model capacity の別要因を疑う

## 6. 見たい結論

この runbook で決めたいのは次の一点。

> `policy_direct_hints + context_gate` は、長い imitation（total 50000 matches）でも旧モデルより高い ceiling を持つか。

ここで `yes` なら、新モデルを本命として PPO に戻す価値がさらに強くなる。  
ここで `no` なら、architecture 改善の寄与は初期域に限られ、別の上限制約が強いと考えられる。

## 7. 想定所要時間

- 旧モデル 1 seed: `2.5〜4時間`
- 新モデル 1 seed: `3〜4.5時間`
- 合計: `5.5〜8.5時間`

補足:
- `50 chunks` なので chunk ごとの起動オーバーヘッドが積み上がる。
- 新モデルは direct hint branch 分だけ旧モデルより少し重い見込み。
- 夜間実行向き。

## 8. 実行方針

- まずは 1 seed で long-run ceiling の形を見る
- 新モデルが後半まで優位なら、その後に `3 seeds` へ広げる
- もし後半で差が縮まるなら、architecture だけでなく teacher 定義や imitation objective の見直しも検討する
