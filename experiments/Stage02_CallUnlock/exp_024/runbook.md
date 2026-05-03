# Experiment Runbook: exp_024

作成日: 2026-05-02  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_024` の目的は、以前いったん不採用にした **役ヒント特徴量 (`tile_presence_flags`)** を、現在の安定化済み Stage2a baseline 上で再評価することである。

現在の基準は `exp_023` の結果を受けた

```text
separated policy-only PPO + no-anchor + lr=1e-4 + clip=0.15
```

である。

`exp_016`〜`exp_019` では `tile_presence_flags` は practical performance を改善しなかった。しかし当時は、後に `exp_023` で主因と判断した **mixed PPO に baseline actor sample を PPO ratio 付きで混ぜる問題** が残っていた。

したがって、過去の `yakuflags` 不採用判断は、現行の separated PPO 環境で再検証する価値がある。

今回の問いは次である。

```text
役ヒント特徴量は、mixed PPO 問題を取り除いた現在の stable RL 条件でも、
policy performance / yaku semantic signal のどちらかを改善するか。
```

## 2. 背景

### 2.1 過去の yakuflags 実験

`CQ-0270` で、次の 6 つの self tile-presence flags を追加した。

| index | feature | 意味 |
|---:|---|---|
| 0 | `self_has_honor` | 字牌を 1 枚以上持つ |
| 1 | `self_has_terminal` | 1/9 牌を 1 枚以上持つ |
| 2 | `self_has_simple` | 2-8 数牌を 1 枚以上持つ |
| 3 | `self_has_man` | 萬子を 1 枚以上持つ |
| 4 | `self_has_pin` | 筒子を 1 枚以上持つ |
| 5 | `self_has_sou` | 索子を 1 枚以上持つ |

狙いは、特に `Tanyao` のような「么九牌・字牌がない」ことが重要な役を MLP が読みやすくすることだった。

過去結果:

- `exp_016`: shared input に常時追加すると policy が悪化
- `exp_017`: `value_hidden_dims=[256,128]` に wide 化すると `on_wide` はかなり回復し、`Tanyao` の弱い確率信号は改善
- `exp_018`: `semantic_only` routing は悪化
- `exp_019`: `on_wide` を 3seed 比較しても practical baseline には勝てず、不採用

ただし、これらはすべて mixed PPO 問題が残っていた時期の判断である。

### 2.2 exp_023 の新基準

`exp_023` では、baseline actor sample を PPO policy update から除外し、`actor_type="policy"` sample のみで PPO 更新する `separated policy-only PPO` を試した。

3seed 平均:

| condition | final | best | best5 | best10 | tail5 | tail10 | tail20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp022 mixed | 3.132 | 2.147 | 2.233 | 2.271 | 3.076 | 2.985 | 2.816 |
| exp023 separated | 2.167 | 2.040 | 2.156 | 2.182 | 2.176 | 2.199 | 2.200 |

`exp_023` で long-run collapse は解消され、今後の Stage2a baseline は separated policy-only PPO とするのが自然になった。

### 2.3 exp_023 semantic baseline

`exp_023` final cycle59 について、3seed の semantic eval を追加実行した。

参照:

- `experiments/Stage02_CallUnlock/exp_023/semantic_eval_seed42_final_cycle59/semantic_eval_final_cycle59_summary.md`
- `experiments/Stage02_CallUnlock/exp_023/semantic_eval_seed43_final_cycle59/semantic_eval_final_cycle59_summary.md`
- `experiments/Stage02_CallUnlock/exp_023/semantic_eval_seed44_final_cycle59/semantic_eval_final_cycle59_summary.md`

3seed 平均:

| metric | value |
|---|---:|
| terminal accuracy | 0.6208 |
| yaku micro F1 | 0.3931 |
| yaku macro F1 | 0.0753 |
| yaku exact match | 0.1571 |
| Tanyao mean_p | 0.2683 |
| Tanyao hit@0.2 | 0.7367 |
| Tanyao top3 | 0.8795 |
| Yakuhai mean_p | 0.3234 |
| Pinfu mean_p | 0.2054 |
| win_called recall | 0.0069 |
| win_called top3 | 0.6542 |
| deal_in ROC-AUC | 0.5282 |

読み:

- practical performance は良いが semantic head はまだ粗い
- `Tanyao` は threshold 0.5 ではほぼ立たないが、弱い信号はかなり存在する
- `deal_in` risk はまだ弱い
- yakuflags 再評価では、`avg_rank` だけでなく `Tanyao mean_p / hit@0.2 / top3` を必ず見る

## 3. 今回の問い

1. `tile_presence_flags=true` + `value_hidden_dims=[256,128]` は、現行 separated PPO でも policy performance を改善するか
2. practical performance が改善しない場合でも、`Tanyao` / `Pinfu` / `Yakuhai` の semantic signal は改善するか
3. 役ヒント特徴量は entropy / max_prob / clip_fraction を悪化させずに保持できるか
4. 過去の yakuflags 不採用判断を維持すべきか、再検討すべきか

## 4. 実験方針

### 4.1 比較対象

#### Reference: exp_023 separated baseline

再実行しない。`exp_023/report.md` と semantic eval 出力を参照する。

条件:

```text
feature_encoder.tile_presence_flags=false
model.value_hidden_dims=[128,64]
model.semantic_aux.tile_presence_flags_semantic_only=false
training.rule_mix_learner.ppo_mode="separated"
```

#### New: exp_024 yakuflags on_wide separated

新規に 3seed 実行する。

条件:

```text
feature_encoder.tile_presence_flags=true
model.value_hidden_dims=[256,128]
model.semantic_aux.tile_presence_flags_semantic_only=false
training.rule_mix_learner.ppo_mode="separated"
```

`semantic_only` は今回試さない。`exp_018` で悪化しており、再評価するなら別実験に分ける。

### 4.2 なぜ 3seed か

以前の `yakuflags` は seed 差が大きく、seed42 だけでは判断を誤る可能性があった。今回は user が長時間実行可能なタイミングなので、最初から seed42/43/44 の 3seed を取る。

### 4.3 交絡回避

固定するもの:

- separated policy-only PPO
- no-anchor
- lr / clip / value loss / semantic loss 係数
- selfplay / eval match 数
- rule_mix の policy/baseline 混合比
- baseline actor sample を PPO policy loss に混ぜない方針

変えるもの:

- `feature_encoder.tile_presence_flags`
- `model.value_hidden_dims`

注意:

`tile_presence_flags` と `value_hidden_dims` を同時に変えるため、純粋な feature 単独効果ではない。これは過去の `exp_017` で `on_narrow` が悪く、`on_wide` が最も有望だったためである。今回は「採用候補として一番勝ち筋がある yakuflags 条件」を再検証する。

## 5. 条件定義

全 seed 共通:

- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `training.multi_cycle.num_cycles = 60`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `training.policy_anchor.enabled = false`
- `training.policy_anchor.coef = 0.0`
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.15`
- `training.entropy_coef = 0.0`
- `training.value_loss_coef = 0.125`
- `training.rule_mix.enabled = true`
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix.save_baseline_actions = true`
- `training.rule_mix_learner.enabled = true`
- `training.rule_mix_learner.ppo_mode = "separated"`
- `training.rule_mix_learner.baseline_imitation_epochs = 0`
- `training.rule_mix_learner.policy_ppo_epochs = 1`
- `training.rule_mix_learner.allow_mixed_offpolicy_baseline = false`
- `model.semantic_aux.enabled = true`
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.enabled = true`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`
- `model.semantic_aux.tile_presence_flags_semantic_only = false`
- `feature_encoder.tile_presence_flags = true`
- `model.value_hidden_dims = [256,128]`
- `selfplay.temperature = 1.0`
- `evaluation.num_workers = 10`
- `training.imitation_eval.num_workers = 10`
- expected shard semantics: `sample_semantics_version = 3`

新規 run:

| label | seed | role |
|---|---:|---|
| `Y_onwide_separated_seed42` | 42 | exp023 seed42 と比較 |
| `Y_onwide_separated_seed43` | 43 | exp023 seed43 と比較 |
| `Y_onwide_separated_seed44` | 44 | exp023 seed44 と比較 |

## 6. 実行コマンド

3seed を連続実行する。失敗時に止めたい場合は `set -e` を付ける。

### seed42

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
  'experiment.name="stage2a_exp024_Y_onwide_separated_seed42"' \
  'experiment.global_seed=42' \
  'training.multi_cycle.num_cycles=60' \
  'training.multi_cycle.selfplay_matches_per_cycle=200' \
  'training.policy_anchor.enabled=false' \
  'training.policy_anchor.coef=0.0' \
  'training.lr=0.0001' \
  'training.clip_epsilon=0.15' \
  'training.entropy_coef=0.0' \
  'training.value_loss_coef=0.125' \
  'training.rule_mix.enabled=true' \
  'training.rule_mix.policy_ratio=0.50' \
  'training.rule_mix.save_baseline_actions=true' \
  'training.rule_mix_learner.enabled=true' \
  'training.rule_mix_learner.ppo_mode="separated"' \
  'training.rule_mix_learner.baseline_imitation_epochs=0' \
  'training.rule_mix_learner.policy_ppo_epochs=1' \
  'training.rule_mix_learner.allow_mixed_offpolicy_baseline=false' \
  'model.semantic_aux.enabled=true' \
  'model.semantic_aux.policy_projection_dim=16' \
  'model.semantic_aux.tile_presence_flags_semantic_only=false' \
  'model.value_hidden_dims=[256,128]' \
  'training.semantic_aux.enabled=true' \
  'training.semantic_aux.terminal_loss_coef=0.1' \
  'training.semantic_aux.yaku_loss_coef=0.05' \
  'feature_encoder.tile_presence_flags=true' \
  'selfplay.temperature=1.0' \
  'evaluation.num_workers=10' \
  'training.imitation_eval.num_workers=10'
```

### seed43

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
  'experiment.name="stage2a_exp024_Y_onwide_separated_seed43"' \
  'experiment.global_seed=43' \
  'training.multi_cycle.num_cycles=60' \
  'training.multi_cycle.selfplay_matches_per_cycle=200' \
  'training.policy_anchor.enabled=false' \
  'training.policy_anchor.coef=0.0' \
  'training.lr=0.0001' \
  'training.clip_epsilon=0.15' \
  'training.entropy_coef=0.0' \
  'training.value_loss_coef=0.125' \
  'training.rule_mix.enabled=true' \
  'training.rule_mix.policy_ratio=0.50' \
  'training.rule_mix.save_baseline_actions=true' \
  'training.rule_mix_learner.enabled=true' \
  'training.rule_mix_learner.ppo_mode="separated"' \
  'training.rule_mix_learner.baseline_imitation_epochs=0' \
  'training.rule_mix_learner.policy_ppo_epochs=1' \
  'training.rule_mix_learner.allow_mixed_offpolicy_baseline=false' \
  'model.semantic_aux.enabled=true' \
  'model.semantic_aux.policy_projection_dim=16' \
  'model.semantic_aux.tile_presence_flags_semantic_only=false' \
  'model.value_hidden_dims=[256,128]' \
  'training.semantic_aux.enabled=true' \
  'training.semantic_aux.terminal_loss_coef=0.1' \
  'training.semantic_aux.yaku_loss_coef=0.05' \
  'feature_encoder.tile_presence_flags=true' \
  'selfplay.temperature=1.0' \
  'evaluation.num_workers=10' \
  'training.imitation_eval.num_workers=10'
```

### seed44

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
  'experiment.name="stage2a_exp024_Y_onwide_separated_seed44"' \
  'experiment.global_seed=44' \
  'training.multi_cycle.num_cycles=60' \
  'training.multi_cycle.selfplay_matches_per_cycle=200' \
  'training.policy_anchor.enabled=false' \
  'training.policy_anchor.coef=0.0' \
  'training.lr=0.0001' \
  'training.clip_epsilon=0.15' \
  'training.entropy_coef=0.0' \
  'training.value_loss_coef=0.125' \
  'training.rule_mix.enabled=true' \
  'training.rule_mix.policy_ratio=0.50' \
  'training.rule_mix.save_baseline_actions=true' \
  'training.rule_mix_learner.enabled=true' \
  'training.rule_mix_learner.ppo_mode="separated"' \
  'training.rule_mix_learner.baseline_imitation_epochs=0' \
  'training.rule_mix_learner.policy_ppo_epochs=1' \
  'training.rule_mix_learner.allow_mixed_offpolicy_baseline=false' \
  'model.semantic_aux.enabled=true' \
  'model.semantic_aux.policy_projection_dim=16' \
  'model.semantic_aux.tile_presence_flags_semantic_only=false' \
  'model.value_hidden_dims=[256,128]' \
  'training.semantic_aux.enabled=true' \
  'training.semantic_aux.terminal_loss_coef=0.1' \
  'training.semantic_aux.yaku_loss_coef=0.05' \
  'feature_encoder.tile_presence_flags=true' \
  'selfplay.temperature=1.0' \
  'evaluation.num_workers=10' \
  'training.imitation_eval.num_workers=10'
```

## 7. 実行後に取る diagnostics

各 seed の final checkpoint と `cycle_59/selfplay` shard に対して semantic eval を実行する。

コマンド雛形:

```bash
./.venv/bin/python scripts/local/stage2/semantic_head_eval.py \
  --config <run_dir>/config.yaml \
  --checkpoint <run_dir>/checkpoints/checkpoint_learner.pt \
  --shard-dir <run_dir>/cycle_59/selfplay \
  --output-dir experiments/Stage02_CallUnlock/exp_024/semantic_eval_seed<seed>_final_cycle59 \
  --label final_cycle59 \
  --device cpu
```

`<run_dir>` は CLI 出力または `summary.json` から確認し、report に必要な数値だけ転記する。

## 8. 成功判定

### 8.1 実行成功

各 seed で以下を満たす。

- run が正常終了する
- `summary.json` が存在する
- `checkpoints/checkpoint_learner.pt` が存在する
- `cycle_59/selfplay` shard が存在する
- final eval metrics が存在する
- `ppo_diag` に `ppo_mode="separated"` 相当の記録がある
- `used_baseline_samples=0` かつ `excluded_baseline_samples>0` が確認できる

### 8.2 採用判定

主判定は practical performance とする。avg_rank は低いほど良い。

採用寄り:

- 3seed 平均 `final` または `tail10` が `exp_023` より明確に改善する
- 少なくとも悪化せず、かつ yaku semantic signal が大きく改善する
- entropy / max_prob / clip_fraction が悪化しない

保留:

- performance は同等だが、`Tanyao` / `Pinfu` / `Yakuhai` の signal が明確に改善する
- seed ごとの勝敗が割れる

見送り:

- 3seed 平均 avg_rank が悪化する
- deal_in_rate が悪化する
- entropy collapse / max_prob 上昇 / clip_fraction 上昇が見える
- semantic signal も改善しない

## 9. 主評価

`exp_023` との比較で、以下を優先する。

1. `tail10 avg_rank`
2. `final avg_rank`
3. `best10 avg_rank`
4. `win_rate`
5. `deal_in_rate`

`exp_023` 3seed 平均 reference:

| metric | value |
|---|---:|
| final avg_rank | 2.167 |
| best avg_rank | 2.040 |
| best5 avg_rank | 2.156 |
| best10 avg_rank | 2.182 |
| tail5 avg_rank | 2.176 |
| tail10 avg_rank | 2.199 |
| tail20 avg_rank | 2.200 |
| final win_rate | 0.2279 |
| final deal_in_rate | 0.1928 |

## 10. semantic 評価

`exp_023` 3seed 平均 reference:

| metric | value |
|---|---:|
| terminal accuracy | 0.6208 |
| yaku micro F1 | 0.3931 |
| yaku macro F1 | 0.0753 |
| yaku exact match | 0.1571 |
| Tanyao mean_p | 0.2683 |
| Tanyao hit@0.2 | 0.7367 |
| Tanyao top3 | 0.8795 |
| Yakuhai mean_p | 0.3234 |
| Pinfu mean_p | 0.2054 |
| win_called recall | 0.0069 |
| win_called top3 | 0.6542 |
| deal_in ROC-AUC | 0.5282 |

重点的に見るもの:

- `Tanyao mean_p`
- `Tanyao hit@0.2`
- `Tanyao top3`
- `Pinfu mean_p`
- `Yakuhai mean_p`
- `yaku macro F1`
- `terminal win_called top3`
- `deal_in ROC-AUC`

注意:

`sigmoid > 0.5` の recall だけでは、弱い yaku signal の改善を見落とす。`Tanyao` は特に `mean_p` / `hit@0.2` / `top3` を主に読む。

## 11. PPO diagnostics

`exp_023` の安定性を壊していないか確認する。

見るもの:

- `entropy_last`
- `clip_last`
- `log_ratio_p01_last`
- `ratio_max_last`
- `max_prob_mean_last`
- branch 別 `discard` / `call` diagnostics

`exp_023` 3seed 平均 reference:

| metric | value |
|---|---:|
| entropy_last | 0.2841 |
| clip_last | 0.0897 |
| log_ratio_p01_last | -0.4537 |
| ratio_max_last | 6.0594 |
| max_prob_mean_last | 0.8853 |

役ヒント特徴量が policy を尖らせすぎる場合、`max_prob_mean` 上昇や entropy 低下として出る可能性がある。

## 12. 想定リスク

### 12.1 eval worker crash

`CQ-0280` は未実装である。cycle eval worker crash が再発すると run が停止する可能性がある。

発症率は低そうなので今回の実行は許容するが、発生した場合は同一 checkpoint で eval だけ retry する暫定対応を検討する。

### 12.2 容量増加による単純な学習差

今回の差分は `tile_presence_flags=true` と `value_hidden_dims=[256,128]` の同時変更である。したがって、改善が出た場合は「役ヒント特徴量だけ」の効果とは断定しない。

ただし過去実験では `on_wide` が最有望だったため、今回は採用候補としての実用条件を優先する。

### 12.3 semantic improvement と practical performance の乖離

過去 `exp_017` / `exp_019` では、Tanyao の弱い signal 改善が practical performance には結びつかなかった。

今回も semantic 指標だけ改善し、avg_rank が改善しない可能性がある。その場合は、採用ではなく「yaku head 改善候補」として保留する。

## 13. report に必ず残すもの

- 3seed の run 対応
- `exp_023` reference との performance 比較
- `final / best / best5 / best10 / tail5 / tail10 / tail20`
- cycle window: `c00-c09`, `c10-c19`, `c20-c29`, `c30-c39`, `c40-c49`, `c50-c59`
- win_rate / deal_in_rate
- PPO diagnostics
- semantic eval 3seed 表
- `Tanyao` / `Pinfu` / `Yakuhai` の解釈
- 採用 / 保留 / 見送り判断

## 14. 次アクション判定

### 採用する場合

`exp_024` が `exp_023` を performance で上回り、diagnostics も悪化しない場合:

- `tile_presence_flags=true`
- `value_hidden_dims=[256,128]`

を Stage2a 新 baseline 候補にする。

### 保留する場合

performance は同等以下だが semantic signal が明確に改善した場合:

- yaku head 専用の補助 route
- yaku loss の重み調整
- yaku head 直前への限定入力

を別 CQ / 別実験で検討する。

### 見送る場合

performance も semantic も改善しない場合:

- `yakuflags` 系は現行 Stage2a では再び見送り
- 次は `policy_ratio` sweep または `target_kl early stop` に進む

