# Experiment Runbook: exp_002

作成日: 2026-03-29  
Stage: `Stage02_CallUnlock`  
参照:
- `docs/PROJECT.md`
- `docs/RL_SPEC.md`
- `experiments/Stage02_CallUnlock/exp_001/runbook.md`
- `experiments/Stage02_CallUnlock/exp_001/report.md`

## 1. 背景

`exp_001` は、Stage02a の最初の比較実験として

1. A `core_minimal`
2. B `stage1style_context`
3. C `stage1style_context_plus_danger`

を比較し、Stage02 の暫定 baseline を決めることを目的として始めた。

しかし実際には、

- imitation / selfplay / eval の correctness 問題
- parallel shard 読み込み不整合
- worker thread cap 欠落
- Stage02a learner の大きな速度問題
- mixed PPO の Stage1 parity 不足

が先に露出し、A/B/C の feature 差を見る前に、PPO 自体の安定性が主課題になった。

`exp_001` の long run では、

- A は終盤で eval が壊れた
- B は完走したが PPO 指標が大きく崩れた
- C は途中停止

となり、feature 比較の前に PPO が後半 cycle で崩れる問題を解決する必要があることが分かった。

その後、以下を mainline に反映した。

- Stage02a mixed PPO の baseline weighting
- advantage の全体一括正規化
- unsafe mixed 条件の guard
- optional imitation diagnostics の batched 化
- imitation chunk timing の可視化
- Stage02a learner の tensor path 化
- discard best-set loss のベクトル化
- optional candidate tensor 前計算
- `DecisionShardReader.read_as_tensors()` の direct parquet path

この結果、imitation throughput は大幅に改善し、correctness / throughput 側の準備は概ね整った。

次は、**どの PPO 条件なら Stage02a が long run で壊れないか** を切り分ける。

## 2. 問い

Stage02a の現行 scaffold で、PPO を安定して回せる条件はどれか。

今回の `exp_002` では、feature 比較をいったん止め、A `core_minimal` に固定して次を比較する。

1. `mixed` PPO は安定化済み条件でもまだ崩れるか
2. `separated` PPO に切り替えると安定するか
3. `policy_ratio=0.25` / `baseline_sample_weight=0.5` / `anchor(imitation_fixed)` の組み合わせで、後半 cycle の drift を抑えられるか

具体的には次を確認する。

1. cycle 後半でも `ratio_mean` / `clip_fraction` / `anchor_kl_discard` が暴れないか
2. learner loss が指数的に吹き上がらないか
3. eval `avg_rank` / `win_rate` が後半で一方的に崩れないか
4. `mixed` と `separated` のどちらが安定性の初期 baseline に向くか

## 3. この実験の位置づけ

この `exp_002` は、Stage02 の **PPO 安定化実験** である。

- feature 比較ではない
- Stage02a の current scaffold で PPO 条件を決める実験である
- ここで安定条件が決まるまで、A/B/C 比較は再開しない
- 以後の Stage02 比較実験は、この `exp_002` で決めた安定 PPO 条件を共通土台とする

## 4. 共通条件

### feature set

今回は A `core_minimal` に固定する。

有効:

- `feature_encoder.shanten_hint.enabled = true`
- `feature_encoder.discard_ukeire_hint.enabled = true`
- `feature_encoder.current_shanten.enabled = true`
- `feature_encoder.shape_hint.enabled = true`
- `feature_encoder.turn_context.enabled = true`

無効:

- `feature_encoder.opponent_current_shanten.enabled = false`
- `feature_encoder.opponent_tenpai_flag.enabled = false`
- `feature_encoder.danger_mask.enabled = false`

### model

- `model.discard_hidden_dims = [256, 128]`
- `model.optional_hidden_dims = [128, 64]`
- `model.value_hidden_dims = [128, 64]`
- `model.candidate_dim = 16`
- `model.optional_scorer_hidden = 32`

### imitation

- `selfplay.imitation_matches = 1000`
- `imitation.num_workers = 10`
- `training.imitation_epochs = 8`
- `training.imitation_loss_mode = "tie_aware_best_set"`
- `training.imitation_value_warmstart.enabled = true`
- `training.imitation_value_warmstart.coef = 0.3`
- `training.multi_chunk_imitation.enabled = true`
- `training.multi_chunk_imitation.num_chunks = 3`
- `training.multi_chunk_imitation.imitation_matches_per_chunk = 1000`

### PPO / learner

- `training.epochs = 1`
- `training.lr = 0.0003`
- `training.batch_size = 256`
- `training.clip_epsilon = 0.15`
- `training.value_loss_coef = 0.25`
- `training.gamma = 0.50`
- `training.gae_lambda = 0.0`
- `training.entropy_coef = 0.0`
- `training.max_grad_norm = 0.5`

### rule mix / anchor

- `training.rule_mix.enabled = true`
- `training.rule_mix.policy_ratio = 0.25`
- `training.rule_mix.save_baseline_actions = true`
- `training.rule_mix_learner.enabled = true`
- `training.rule_mix_learner.baseline_sample_weight = 0.5`
- `training.policy_anchor.enabled = true`
- `training.policy_anchor.type = "kl"`
- `training.policy_anchor.coef = 0.5`
- `training.policy_anchor.reference = "imitation_fixed"`

### multi-cycle

- `training.multi_cycle.enabled = true`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `training.multi_cycle.eval_each_cycle = true`

### worker / inference

- `selfplay.num_workers = 10`
- `selfplay.worker_num_threads = 1`
- `selfplay.inference_device = "cpu"`
- `evaluation.num_workers = 10`
- `evaluation.worker_num_threads = 1`
- `evaluation.inference_device = "cpu"`

### eval

- `evaluation.mode = "rotation"`
- `evaluation.num_matches = 50`
- `evaluation.seed_start = 200000`

### phases

- `experiment.phases = ["imitation", "selfplay", "learner", "eval"]`

## 5. 比較条件

### Run A1 `mixed`

- `training.rule_mix_learner.ppo_mode = "mixed"`

狙い:

- Stage1 parity に寄せた mixed PPO 修正後でも、まだ drift が起きるかを確認する
- `policy_ratio=0.25` / `baseline_sample_weight=0.5` でどこまで抑えられるかを見る

### Run A2 `separated`

- `training.rule_mix_learner.ppo_mode = "separated"`

狙い:

- mixed PPO 特有の不安定性を切り分ける
- `separated` なら long run で安定するかを確認する

## 6. 実行順

1. Run A1 `mixed`
2. Run A2 `separated`

理由:

- まず mainline に近い `mixed` を確認する
- それでも崩れる場合、`separated` で安定するかを見る
- この順なら、問題が `mixed` 固有か、PPO 全体かを切り分けやすい

## 7. 実行コマンド

実行前提:

- C++ 側 (`bindings/`, `src/engine/`, `src/rules/`, `src/core/`) を触った場合は、先に `.venv` を rebuild する

```bash
./.venv/bin/python -m pip install -e . --no-build-isolation
```

### Run A1 `mixed`

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_smoke_full.yaml \
  --base-dir runs \
  --override \
  'experiment.name="stage2a_rerun_A_core_minimal_seed42_mc20_mixed_pr025_bsw05_v2"' \
  'experiment.global_seed=42' \
  'experiment.phases=["imitation","selfplay","learner","eval"]' \
  'feature_encoder.shanten_hint.enabled=true' \
  'feature_encoder.discard_ukeire_hint.enabled=true' \
  'feature_encoder.current_shanten.enabled=true' \
  'feature_encoder.shape_hint.enabled=true' \
  'feature_encoder.turn_context.enabled=true' \
  'feature_encoder.opponent_current_shanten.enabled=false' \
  'feature_encoder.opponent_tenpai_flag.enabled=false' \
  'feature_encoder.danger_mask.enabled=false' \
  'model.discard_hidden_dims=[256,128]' \
  'model.optional_hidden_dims=[128,64]' \
  'model.value_hidden_dims=[128,64]' \
  'model.candidate_dim=16' \
  'model.optional_scorer_hidden=32' \
  'selfplay.imitation_matches=1000' \
  'selfplay.policy_ratio=1.0' \
  'selfplay.save_baseline_actions=false' \
  'imitation.num_workers=10' \
  'training.imitation_epochs=8' \
  'training.epochs=1' \
  'training.lr=0.0003' \
  'training.batch_size=256' \
  'training.clip_epsilon=0.15' \
  'training.value_loss_coef=0.25' \
  'training.gamma=0.50' \
  'training.gae_lambda=0.0' \
  'training.entropy_coef=0.0' \
  'training.max_grad_norm=0.5' \
  'training.imitation_loss_mode="tie_aware_best_set"' \
  'training.imitation_value_warmstart.enabled=true' \
  'training.imitation_value_warmstart.coef=0.3' \
  'training.multi_chunk_imitation.enabled=true' \
  'training.multi_chunk_imitation.num_chunks=3' \
  'training.multi_chunk_imitation.imitation_matches_per_chunk=1000' \
  'training.multi_cycle.enabled=true' \
  'training.multi_cycle.num_cycles=20' \
  'training.multi_cycle.selfplay_matches_per_cycle=200' \
  'training.multi_cycle.eval_each_cycle=true' \
  'training.rule_mix.enabled=true' \
  'training.rule_mix.policy_ratio=0.25' \
  'training.rule_mix.save_baseline_actions=true' \
  'training.rule_mix_learner.enabled=true' \
  'training.rule_mix_learner.ppo_mode="mixed"' \
  'training.rule_mix_learner.baseline_sample_weight=0.5' \
  'training.policy_anchor.enabled=true' \
  'training.policy_anchor.type="kl"' \
  'training.policy_anchor.coef=0.5' \
  'training.policy_anchor.reference="imitation_fixed"' \
  'selfplay.num_workers=10' \
  'selfplay.worker_num_threads=1' \
  'selfplay.inference_device="cpu"' \
  'evaluation.mode="rotation"' \
  'evaluation.num_matches=50' \
  'evaluation.num_workers=10' \
  'evaluation.worker_num_threads=1' \
  'evaluation.inference_device="cpu"' \
  'evaluation.seed_start=200000'
```

### Run A2 `separated`

`A1` と同条件で、以下だけ変更する。

- `training.rule_mix_learner.ppo_mode = "separated"`

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_smoke_full.yaml \
  --base-dir runs \
  --override \
  'experiment.name="stage2a_rerun_A_core_minimal_seed42_mc20_separated_pr025_bsw05_v1"' \
  'experiment.global_seed=42' \
  'experiment.phases=["imitation","selfplay","learner","eval"]' \
  'feature_encoder.shanten_hint.enabled=true' \
  'feature_encoder.discard_ukeire_hint.enabled=true' \
  'feature_encoder.current_shanten.enabled=true' \
  'feature_encoder.shape_hint.enabled=true' \
  'feature_encoder.turn_context.enabled=true' \
  'feature_encoder.opponent_current_shanten.enabled=false' \
  'feature_encoder.opponent_tenpai_flag.enabled=false' \
  'feature_encoder.danger_mask.enabled=false' \
  'model.discard_hidden_dims=[256,128]' \
  'model.optional_hidden_dims=[128,64]' \
  'model.value_hidden_dims=[128,64]' \
  'model.candidate_dim=16' \
  'model.optional_scorer_hidden=32' \
  'selfplay.imitation_matches=1000' \
  'selfplay.policy_ratio=1.0' \
  'selfplay.save_baseline_actions=false' \
  'imitation.num_workers=10' \
  'training.imitation_epochs=8' \
  'training.epochs=1' \
  'training.lr=0.0003' \
  'training.batch_size=256' \
  'training.clip_epsilon=0.15' \
  'training.value_loss_coef=0.25' \
  'training.gamma=0.50' \
  'training.gae_lambda=0.0' \
  'training.entropy_coef=0.0' \
  'training.max_grad_norm=0.5' \
  'training.imitation_loss_mode="tie_aware_best_set"' \
  'training.imitation_value_warmstart.enabled=true' \
  'training.imitation_value_warmstart.coef=0.3' \
  'training.multi_chunk_imitation.enabled=true' \
  'training.multi_chunk_imitation.num_chunks=3' \
  'training.multi_chunk_imitation.imitation_matches_per_chunk=1000' \
  'training.multi_cycle.enabled=true' \
  'training.multi_cycle.num_cycles=20' \
  'training.multi_cycle.selfplay_matches_per_cycle=200' \
  'training.multi_cycle.eval_each_cycle=true' \
  'training.rule_mix.enabled=true' \
  'training.rule_mix.policy_ratio=0.25' \
  'training.rule_mix.save_baseline_actions=true' \
  'training.rule_mix_learner.enabled=true' \
  'training.rule_mix_learner.ppo_mode="separated"' \
  'training.rule_mix_learner.baseline_sample_weight=0.5' \
  'training.policy_anchor.enabled=true' \
  'training.policy_anchor.type="kl"' \
  'training.policy_anchor.coef=0.5' \
  'training.policy_anchor.reference="imitation_fixed"' \
  'selfplay.num_workers=10' \
  'selfplay.worker_num_threads=1' \
  'selfplay.inference_device="cpu"' \
  'evaluation.mode="rotation"' \
  'evaluation.num_matches=50' \
  'evaluation.num_workers=10' \
  'evaluation.worker_num_threads=1' \
  'evaluation.inference_device="cpu"' \
  'evaluation.seed_start=200000'
```

## 8. 成功判定

最低条件:

1. 20 cycle を NaN / invalid action / hard crash なしで完走する
2. cycle 後半でも learner loss が指数的に吹き上がらない
3. `ratio_mean` が極端に外れない
4. `clip_fraction` が高止まりしない
5. `anchor_kl_discard` が後半で暴走しない

目安:

- `ratio_mean` が桁崩れしない
- `clip_fraction` が常時 0.4 以上に張り付かない
- `anchor_kl_discard` が後半で数倍に吹き上がらない
- eval `avg_rank` が後半で単調に悪化し続けない

## 9. 集計観点

各 run について、最低限次を確認する。

1. imitation chunk timing
   - `data_generation_sec`
   - `learner_sec`
   - `diagnostics_sec`
2. cycle ごとの
   - `learner loss`
   - `updates`
   - `avg_rank`
   - `win_rate`
3. final に report へ転記する
   - `policy_loss`
   - `value_loss`
   - `ratio_mean`
   - `clip_fraction`
   - `anchor_kl_discard`
   - `anchor_kl_optional`
   - mixed PPO diagnostics

## 10. 次アクション判定

### `mixed` が安定し、`separated` も不要な場合

- `mixed` を Stage02 暫定 baseline PPO 条件として採用する
- A/B/C feature 比較を再開する

### `mixed` が不安定で、`separated` が安定する場合

- `separated` を Stage02 暫定 baseline PPO 条件として採用する
- A/B/C feature 比較は `separated` 条件で再開する

### `mixed` / `separated` の両方が不安定な場合

- feature 比較には戻らない
- `lr`
- `clip_epsilon`
- `max_grad_norm`
- `value_loss_coef`
の tuning 実験へ移る

## 11. 注意

- `exp_001` の A/B/C long run は、feature 比較の正式結果としては採用しない
- `exp_002` の目的は「PPO 条件の決定」であり、feature 優劣の判断ではない
- report では、まず安定性の有無を主に書き、性能比較は二次的に扱う
