# exp_058 runbook

最終更新: 2026-03-17  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `imitation-only` 条件で旧モデルと `policy_direct_hints + context_gate` 新モデルを `3 seeds` で直接比較し、構造変更そのものが imitation 性能と対戦性能にどれだけ効くかを確認する

---

## 0. 実験の位置づけ

- 背景:
  - `exp_057` までの rule-only mixed PPO では、`gamma=0.50` 付近が有望だったが、長期ではなお戻りが残った。
  - 一方、新モデル `policy_direct_hints` を入れた単発確認では、mixed PPO では効果が限定的に見えた。
  - その後の `imitation-only` 単発比較では、新モデルが旧モデルより `avg_score` を大きく改善し、少なくとも branch 自体は効いている可能性が高まった。
- 仮説:
  - `shanten_hint` / `discard_ukeire_hint` を shared trunk ではなく policy logits 直前で扱う構造は、imitation では旧モデルより明確に有利である。
  - もし `3 seeds` でも差が再現するなら、PPO で伸びきらない原因は architecture より RL target 側にあると整理しやすくなる。
- 方針:
  - `imitation_matches=10000` に増やし、旧モデルと新モデルを `3 seeds` で直接 A/B 比較する。
  - PPO は入れず、`experiment.phases=["imitation","selfplay","eval"]` + `selfplay.num_matches=0` で imitation 後評価だけを見る。

## 1. 条件

- 条件数: 2
- seeds: `42,43,44`（3 seeds）
- imitation matches: `10000`
- eval: `rotation, num_matches=100`

条件一覧:

- A: `old_model_imitation10000`
  - `model.policy_direct_hints.enabled=false`
  - `shanten_hint` / `discard_ukeire_hint` は従来どおり trunk に入る
- B: `new_model_direct_hints_imitation10000`
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
- `selfplay.imitation_matches=10000`
- `training.imitation_epochs=8`
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

1. imitation 指標
   - `teacher_top1_match_rate`
   - `teacher_best_set_hit_rate`
   - `value_loss`
2. eval 指標
   - `avg_rank`
   - `avg_score`
   - `win_rate`
   - `deal_in_rate`
3. 条件差の解釈
   - teacher 指標がほぼ同じなのに eval が伸びるか
   - teacher 指標ごと明確に改善するか
   - 3 seeds で差の向きが揃うか

## 4. 成功判定

- 各条件の `batch_summary.json` で:
  - `success_count == 3`, `failure_count == 0`
- driver 全体で:
  - `completed == 2`, `failed == 0`
- 各 run で:
  - `phase_stats.imitation` が存在する
  - `phase_stats.eval` が存在する
  - `selfplay.num_matches == 0` にもかかわらず phase validation で落ちていない
- 新モデル条件では:
  - `summary.model_features.policy_direct_hints.enabled == true`
  - `summary.model_features.policy_direct_hints.sources == ["shanten_hint","discard_ukeire_hint"]`

## 5. 判定基準

- 新モデルが明確に有望:
  - `3 seeds` 平均で `avg_rank` / `avg_score` が旧モデルより改善
  - かつ差の向きが seed 間で概ね揃う
- teacher 模倣強化型の改善:
  - `teacher_top1_match_rate` / `teacher_best_set_hit_rate` も一緒に改善する
- 打牌品質改善型の改善:
  - teacher 指標差は小さいが、`avg_score` / `avg_rank` は明確に改善する
  - この場合、direct hint branch は「同じ best-set 内でより良い選択」を助けている可能性がある
- architecture 効果が弱い:
  - teacher 指標も eval も差が小さい
  - その場合は構造変更より objective / RL 側の問題が本命

## 6. 見たい結論

この runbook で決めたいのは次の一点。

> `policy_direct_hints + context_gate` は、imitation 段階で旧モデルより一貫して強いか。

ここで `yes` なら、次段の mixed PPO 比較は「効く model の上で RL がどこまで伸びるか」を見る実験として整理できる。

## 7. 想定所要時間

- 旧モデル 1条件 x 3 seeds: `70〜90分`
- 新モデル 1条件 x 3 seeds: `90〜120分`
- 合計: `160〜210分`
- 余裕込み: `3〜3.5時間`

補足:
- `imitation_matches=10000` のため、単発確認よりかなり長い。
- 新モデルは direct hint branch 分だけ旧モデルよりやや重い見込み。

## 8. 実行方針

- まずは architecture の効き方だけを clean に切り分ける
- ここで新モデルが勝てば、そのモデルを次の PPO 系比較の本命にする
- ここで差が再現しなければ、単発 seed の改善は偶然の可能性を考慮し、architecture 探索の優先度を下げる
