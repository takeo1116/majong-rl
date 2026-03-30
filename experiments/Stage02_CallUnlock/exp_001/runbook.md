# Experiment Runbook: exp_001

作成日: 2026-03-28  
Stage: `Stage02_CallUnlock`  
参照:
- `experiments/Stage01_DiscardOnly/README.md`
- `experiments/Stage02_CallUnlock/exp_001/report.md`
- `experiments/Stage02_CallUnlock/exp_001/run_map.json`

## 1. 背景

Stage01 では discard-only 条件で imitation / PPO 基盤を立ち上げ、
`context_plus_danger` を current best まで押し上げた。

Stage02a では、まず

- `Chi`
- `Pon`
- `Daiminkan`
- `Skip`

を学習対象に加えた `CallUnlock` 段階を立ち上げた。

実装レビューと短い pilot を通じて、

- Stage2a の imitation / selfplay / learner / eval が end-to-end で動くこと
- summary / manifest / future label / deterministic eval が概ね安定したこと
- Stage2a の imitation / selfplay / eval を multi-process で回せること
- Stage2a に
  - `multi_chunk_imitation`
  - `rule_mix` / `rule_mix_learner`
  - grouped GAE
  - `policy_anchor`
  - teacher-aware imitation
  - imitation value warmstart
  - `discard / optional / value` 3-branch model
  が入ったこと
- 1 seed pilot では `core_minimal` が `stage1style_context` より良さそうなこと

が見えている。

次は、Stage1 baseline にかなり近い外枠

- `multi_chunk_imitation`
- `imitation_value_warmstart`
- `rule_mix` / `rule_mix_learner`
- `policy_anchor(imitation_fixed)`
- cycle ごとの eval

を使って、Stage02 の最初の比較実験を行う。

## 2. 問い

Stage2a を長尺 multi-cycle で回したとき、次の 3 条件のうちどれを Stage02 の暫定 baseline に採用すべきか。

1. A `core_minimal`
2. B `stage1style_context`
3. C `stage1style_context_plus_danger`

具体的には次を確認する。

1. A/B/C のうち、どれが最も素直に学習して eval 成績が良いか
2. `opponent_current_shanten` / `opponent_tenpai_flag` が Stage2a でも効くか
3. `danger_mask` を call 解放段階にも入れる価値があるか

## 3. この実験の位置づけ

この `exp_001` は、Stage02 の最初の比較実験である。

- Stage02 の基準条件をここで決める
- 以後の Stage02 実験は、この `exp_001` の勝ち条件を参照点にする
- Stage01 current best をそのまま移植する実験ではなく、
  Stage02a の最新 scaffold 上で自然な特徴セットを見極める実験と位置づける
- ただし現時点でも
  - optional 側 best-set imitation
  - EMA anchor
  は未実装なので、`exp_001` は「現時点の Stage02a mainline scaffold での最初の比較実験」とみなす

## 4. 共通条件

### model

- `model.discard_hidden_dims = [256, 128]`
- `model.optional_hidden_dims = [128, 64]`
- `model.value_hidden_dims = [128, 64]`
- `model.candidate_dim = 16`
- `model.optional_scorer_hidden = 32`

補足:

- policy は `discard / optional` の 2 branch
- critic は独立 `value_trunk`
- optional decision では `response_context + optional_summary` を value に入れる

### imitation / PPO

- `selfplay.imitation_matches = 1000`
- `imitation.num_workers = 10`
- `training.imitation_epochs = 8`
- `training.epochs = 1`
- `training.lr = 0.0003`
- `training.batch_size = 256`
- `training.clip_epsilon = 0.15`
- `training.value_loss_coef = 0.25`
- `training.gamma = 0.50`
- `training.gae_lambda = 0.0`
- `training.entropy_coef = 0.0`
- `training.max_grad_norm = 0.5`
- `training.imitation_loss_mode = "tie_aware_best_set"`
- `training.imitation_value_warmstart.enabled = true`
- `training.imitation_value_warmstart.coef = 0.3`
- `training.multi_chunk_imitation.enabled = true`
- `training.multi_chunk_imitation.num_chunks = 3`
- `training.multi_chunk_imitation.imitation_matches_per_chunk = 1000`
- `selfplay.policy_ratio = 1.0`
- `selfplay.save_baseline_actions = false`

### multi-cycle

- `training.multi_cycle.enabled = true`
- `training.multi_cycle.num_cycles = 30`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `training.multi_cycle.eval_each_cycle = true`
- `training.rule_mix.enabled = true`
- `training.rule_mix.policy_ratio = 0.0`
- `training.rule_mix.save_baseline_actions = true`
- `training.rule_mix_learner.enabled = true`
- `training.rule_mix_learner.ppo_mode = "mixed"`
- `training.policy_anchor.enabled = true`
- `training.policy_anchor.type = "kl"`
- `training.policy_anchor.coef = 0.5`
- `training.policy_anchor.reference = "imitation_fixed"`

### worker / inference

- `selfplay.num_workers = 10`
- `evaluation.num_workers = 10`
- `selfplay.inference_device = "cpu"`
- `evaluation.inference_device = "cpu"`

補足:

- Stage02 の長尺比較では、まず Stage01 baseline と同様に **CPU inference + multi-process worker** を使う
- 学習本体は引き続き training device 側で行う
- 現時点では worker 間の GPU 推論共有は前提にしない

### eval

- `evaluation.mode = "rotation"`
- `evaluation.num_matches = 50`
- `evaluation.seed_start = 200000`

### phases

- `experiment.phases = ["imitation", "selfplay", "learner", "eval"]`

## 5. 比較条件

### A `core_minimal`

有効化:

- `feature_encoder.shanten_hint.enabled = true`
- `feature_encoder.discard_ukeire_hint.enabled = true`
- `feature_encoder.current_shanten.enabled = true`
- `feature_encoder.shape_hint.enabled = true`
- `feature_encoder.turn_context.enabled = true`

無効:

- `feature_encoder.opponent_current_shanten.enabled = false`
- `feature_encoder.opponent_tenpai_flag.enabled = false`
- `feature_encoder.danger_mask.enabled = false`

狙い:

- Stage02a の最小有望条件
- call 解放直後に必要な情報だけでどこまで行けるかを見る

### B `stage1style_context`

A に追加:

- `feature_encoder.opponent_current_shanten.enabled = true`
- `feature_encoder.opponent_tenpai_flag.enabled = true`

据え置き:

- `feature_encoder.danger_mask.enabled = false`

狙い:

- Stage01 で補助的に効いた opponent 文脈が Stage02a でも有効かを見る

### C `stage1style_context_plus_danger`

B に追加:

- `feature_encoder.danger_mask.enabled = true`

狙い:

- Stage01 current best の本命だった `danger_mask` を Stage02a に持ち込む価値があるかを見る
- ただし Stage02a では encoder が call 側にも流れるため、Stage01 と全く同じ意味ではないことに注意する

## 6. 実行順

1. A `core_minimal`
2. B `stage1style_context`
3. C `stage1style_context_plus_danger`

運用:

- まず A を単独で先行実行する
- A が正常完走したら、B/C は driver でまとめて実行する

理由:

- A は現時点で最も素直な暫定 baseline 候補
- B/C をまとめて流す前に、multi-cycle 長尺 run 自体の安定性をもう一度確認したい

注意:

- 先行して実行した `stage2a_long_A_core_minimal_seed42_mc30_c54ef433` は、
  `selfplay.imitation_matches` 指定前かつ parallel parity 前の exploratory run とみなし、
  **正式比較には使わない**
- `exp_001` の正式な A 条件は、以下の parallel 条件を含む rerun を採用する

## 7. 実行コマンド

実行前提:

- C++ 側 (`bindings/`, `src/engine/`, `src/rules/`, `src/core/`) を触った場合は、先に `.venv` を rebuild する

```bash
./.venv/bin/python -m pip install -e . --no-build-isolation
```

### A `core_minimal`

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_smoke_full.yaml \
  --base-dir runs \
  --override \
  'experiment.name="stage2a_long_A_core_minimal_seed42_mc30"' \
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
  'training.multi_cycle.num_cycles=30' \
  'training.multi_cycle.selfplay_matches_per_cycle=200' \
  'training.multi_cycle.eval_each_cycle=true' \
  'training.rule_mix.enabled=true' \
  'training.rule_mix.policy_ratio=0.0' \
  'training.rule_mix.save_baseline_actions=true' \
  'training.rule_mix_learner.enabled=true' \
  'training.rule_mix_learner.ppo_mode="mixed"' \
  'training.policy_anchor.enabled=true' \
  'training.policy_anchor.type="kl"' \
  'training.policy_anchor.coef=0.5' \
  'training.policy_anchor.reference="imitation_fixed"' \
  'selfplay.num_workers=10' \
  'selfplay.inference_device="cpu"' \
  'evaluation.mode="rotation"' \
  'evaluation.num_matches=50' \
  'evaluation.num_workers=10' \
  'evaluation.inference_device="cpu"' \
  'evaluation.seed_start=200000'
```

### B `stage1style_context`

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_smoke_full.yaml \
  --base-dir runs \
  --override \
  'experiment.name="stage2a_long_B_stage1style_seed42_mc30"' \
  'experiment.global_seed=42' \
  'experiment.phases=["imitation","selfplay","learner","eval"]' \
  'feature_encoder.shanten_hint.enabled=true' \
  'feature_encoder.discard_ukeire_hint.enabled=true' \
  'feature_encoder.current_shanten.enabled=true' \
  'feature_encoder.shape_hint.enabled=true' \
  'feature_encoder.turn_context.enabled=true' \
  'feature_encoder.opponent_current_shanten.enabled=true' \
  'feature_encoder.opponent_tenpai_flag.enabled=true' \
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
  'training.multi_cycle.num_cycles=30' \
  'training.multi_cycle.selfplay_matches_per_cycle=200' \
  'training.multi_cycle.eval_each_cycle=true' \
  'training.rule_mix.enabled=true' \
  'training.rule_mix.policy_ratio=0.0' \
  'training.rule_mix.save_baseline_actions=true' \
  'training.rule_mix_learner.enabled=true' \
  'training.rule_mix_learner.ppo_mode="mixed"' \
  'training.policy_anchor.enabled=true' \
  'training.policy_anchor.type="kl"' \
  'training.policy_anchor.coef=0.5' \
  'training.policy_anchor.reference="imitation_fixed"' \
  'selfplay.num_workers=10' \
  'selfplay.inference_device="cpu"' \
  'evaluation.mode="rotation"' \
  'evaluation.num_matches=50' \
  'evaluation.num_workers=10' \
  'evaluation.inference_device="cpu"' \
  'evaluation.seed_start=200000'
```

### C `stage1style_context_plus_danger`

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_smoke_full.yaml \
  --base-dir runs \
  --override \
  'experiment.name="stage2a_long_C_stage1style_plus_danger_seed42_mc30"' \
  'experiment.global_seed=42' \
  'experiment.phases=["imitation","selfplay","learner","eval"]' \
  'feature_encoder.shanten_hint.enabled=true' \
  'feature_encoder.discard_ukeire_hint.enabled=true' \
  'feature_encoder.current_shanten.enabled=true' \
  'feature_encoder.shape_hint.enabled=true' \
  'feature_encoder.turn_context.enabled=true' \
  'feature_encoder.opponent_current_shanten.enabled=true' \
  'feature_encoder.opponent_tenpai_flag.enabled=true' \
  'feature_encoder.danger_mask.enabled=true' \
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
  'training.multi_cycle.num_cycles=30' \
  'training.multi_cycle.selfplay_matches_per_cycle=200' \
  'training.multi_cycle.eval_each_cycle=true' \
  'training.rule_mix.enabled=true' \
  'training.rule_mix.policy_ratio=0.0' \
  'training.rule_mix.save_baseline_actions=true' \
  'training.rule_mix_learner.enabled=true' \
  'training.rule_mix_learner.ppo_mode="mixed"' \
  'training.policy_anchor.enabled=true' \
  'training.policy_anchor.type="kl"' \
  'training.policy_anchor.coef=0.5' \
  'training.policy_anchor.reference="imitation_fixed"' \
  'selfplay.num_workers=10' \
  'selfplay.inference_device="cpu"' \
  'evaluation.mode="rotation"' \
  'evaluation.num_matches=50' \
  'evaluation.num_workers=10' \
  'evaluation.inference_device="cpu"' \
  'evaluation.seed_start=200000'
```

## 8. 観測したい指標

主に次を見る。

1. `phase_stats.cycles[*].eval.avg_rank`
2. `phase_stats.cycles[*].eval.avg_score`
3. `phase_stats.cycles[*].eval.win_rate`
4. `phase_stats.cycles[*].eval.deal_in_rate`
5. `phase_stats.cycles[*].selfplay.call_count`
6. `phase_stats.cycles[*].selfplay.policy_wins`
7. `phase_stats.cycles[*].selfplay.policy_deal_ins`
8. `phase_stats.cycles[*].learner.policy_loss`
9. `phase_stats.cycles[*].learner.value_loss`
10. `phase_stats.imitation.teacher_top1_match_rate_discard`
11. `phase_stats.imitation.teacher_best_set_hit_rate_discard`
12. `phase_stats.imitation.teacher_top1_match_rate_optional`
13. `phase_stats.imitation.imitation_value_warmstart`
14. `phase_stats.imitation.value_loss`
15. `phase_stats.cycles[*].learner_diag.anchor_kl_discard`
16. `phase_stats.cycles[*].learner_diag.anchor_kl_optional`
17. `phase_stats.learner.ppo_diag.anchor_kl_discard`
18. `phase_stats.learner.ppo_diag.anchor_kl_optional`

比較時には特に、

- final cycle の `avg_rank`
- final cycle の `avg_score`
- final cycle の `deal_in_rate`
- plateau 区間の安定性

を重視する。

## 9. 成功判定

この実験では、次を満たせば比較として成功とみなす。

1. A/B/C がすべて完走する
2. cycle ごとの `selfplay / learner / eval` が summary に残る
3. `call_count` が各条件で non-trivial に出る
4. eval の `avg_rank / avg_score / win_rate / deal_in_rate` が有限で比較可能
5. imitation diagnostics と anchor diagnostics が summary から確認できる
6. imitation value warmstart の有効化と value loss が summary から確認できる
7. A/B/C のどれを Stage02 の暫定 baseline にするか判断できる

## 10. 解釈ルール

### A が良い場合

- Stage02a はまだ最小構成の方が素直
- opponent 文脈や danger 情報は時期尚早、または入れ方の見直しが必要

### B が良い場合

- Stage01 由来の opponent 文脈は Stage02a にも有効
- call 解放段階でも trunk 文脈の headroom がある

### C が良い場合

- `danger_mask` を Stage02a にも持ち込む価値がある
- ただし discard/call 両方に効いている可能性が高いので、後続実験では寄与分解が必要

### どれも不安定な場合

- 条件比較に入る前に Stage02a の reward / value / cycle length を再検討する

## 11. 次アクション

### 勝ち条件が出た場合

- その条件を Stage02 provisional baseline として固定する
- 次の `exp_002` 以降では、その条件を基準にハイパラまたは特徴量を掘る

### A が勝った場合

- `danger_mask` をいきなり足すより、まず call 側の文脈特徴設計を見直す

### B または C が勝った場合

- 追加特徴が Stage02a にも有効とみなし、seed 拡張を検討する

## 12. 命名ルールについて

Stage02 では、実験番号を **`exp_001` から振り直す**。

理由:

- `Stage02_CallUnlock/exp_001` と見た時点で「Stage02 の最初の実験」と分かる
- stage ごとに README / runbook / report を閉じやすい
- 実行時系列は `experiments/Stage02_CallUnlock/exp_001/run_map.json` と Git 履歴で追える

したがって以後の Stage02 実験も stage ローカル番号で管理する。
