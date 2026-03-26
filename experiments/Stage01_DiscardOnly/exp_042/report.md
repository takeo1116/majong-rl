# Experiment Report: exp_042

作成日: 2026-03-14  
対象: `experiments/exp_042/runbook.md`  
目的: C++/encoder 高速化後に、`exp_041 B` と同条件で再実行し、速度改善と集計整合性を確認する

## 1. 実験概要

比較軸:
- A reference: `exp_041 B`（`discard_ukeire_hint=false`）
  - batch: （ローカル run）
- B replay: `exp_042 B`（Aと同条件、コードのみ高速化後）
  - batch: （ローカル run）

補助比較（手動実行）:
- C manual: `discard_ukeire_hint=true`（高速化後）
  - batch: （ローカル run）

共通（主要）:
- seeds: `42,43,44,45,46`
- `selfplay.imitation_matches=200`
- `training.imitation_epochs=8`
- `selfplay.num_matches=200`
- `training.batch_size=512`
- `training.gae_lambda=0.90`
- `model.hidden_dims=[512,256]`
- `model.policy_tower/value_tower enabled`

## 2. 実行結果

| 条件 | batch | success |
|---|---|---:|
| A (`exp_041 B`) | （ローカル run） | 5/5 |
| B (`exp_042 B`) | （ローカル run） | 5/5 |
| C (manual, hint on) | （ローカル run） | 5/5 |

注記:
- `exp_042` は初回実行を中断し、`.venv` の `_mahjong_core` を再インストール後に再実行。
- 比較対象として採用したのは **完走ログ** `experiments/exp_042/driver_logs/20260314_160356_B_replay_after_speedup.log`。

## 3. 主評価: 高速化効果（A vs B）

mean（seed=5）

| 指標 | A (`exp_041 B`) | B (`exp_042 B`) | 改善率 |
|---|---:|---:|---:|
| total_duration_sec | 1529.031 | 71.892 | **-95.3% (21.27x)** |
| imitation.duration_sec | 786.048 | 24.233 | **-96.9% (32.44x)** |
| selfplay.duration_sec | 53.727 | 23.904 | **-55.5% (2.25x)** |
| eval_before.duration_sec | 337.647 | 7.517 | **-97.8% (44.92x)** |
| learner.duration_sec | 9.192 | 8.874 | -3.5% |
| eval.duration_sec | 342.416 | 7.363 | **-97.8% (46.50x)** |

補足 throughput:
- imitation: `121.3 -> 3932.8 steps/s`（32.4x）
- self-play: `2212.0 -> 4960.1 steps/s`（2.24x）

所見:
- 高速化は明確。特に imitation/eval 系フェーズで大幅改善。
- learner フェーズは元々短く、改善幅は小さい。

## 4. 整合性チェック（A vs B）

チェック結果:
- seed ごとの `config.yaml`（`run_name` を除く）は一致。
- 速度以外の主要値は seed 単位で一致:
  - `imitation.total_steps`
  - `train_metrics.total_steps`（self-play）
  - `teacher_top1_match_rate`
  - `teacher_best_set_hit_rate`
  - `eval.{avg_rank, avg_score, win_rate, deal_in_rate}`
  - `ppo_diag.{clip_fraction, ratio_std, value_error_mean}`
- `batch_summary.json` の aggregate も A/B で一致（run_dir 参照先のみ差分）。

結論:
- **挙動を変えずに速度だけ改善**できている。

## 5. 補助比較: `discard_ukeire_hint=true`（B vs C）

mean（seed=5）

### 5.1 速度

| 指標 | B (`hint=false`) | C (`hint=true`) | 差分 |
|---|---:|---:|---:|
| total_duration_sec | 71.892 | 72.334 | +0.6% |
| imitation.duration_sec | 24.233 | 25.056 | +3.4% |
| selfplay.duration_sec | 23.904 | 23.880 | ほぼ同等 |

所見:
- 高速化後は `discard_ukeire_hint` ON/OFF で速度差はほぼ消失。

### 5.2 性能

| 指標 | B (`hint=false`) | C (`hint=true`) | 差分 |
|---|---:|---:|---:|
| teacher_top1_match_rate | 0.222741 | 0.232150 | +0.009409 |
| teacher_best_set_hit_rate | 0.701976 | 0.708177 | +0.006201 |
| eval avg_rank | 3.381667 | 3.453333 | -0.071667 (悪化) |
| eval avg_score | -13380.167 | -14045.333 | -665.167 (悪化) |
| eval win_rate | 0.047613 | 0.048606 | +0.000993 |
| eval deal_in_rate | 0.582312 | 0.582008 | -0.000304 |

所見:
- imitation 指標は改善するが、最終の `avg_rank/avg_score` は悪化。
- 現時点では `hint=true` を採用する根拠は弱い。

## 6. 総合結論

1. `exp_042` の主目的（高速化の再現確認）は達成。
2. A/B 比較で、速度以外の統計が一致しており、集計異常は見当たらない。
3. `discard_ukeire_hint=true` は高速化後でも時間コストは小さいが、今回の設定では対戦性能改善は確認できなかった。

## 7. 今回の判断

- 採用:
  - 高速化後の実装（A/B 同値 + 大幅時短）
- 保留:
  - `discard_ukeire_hint=true`（速度問題は解消、性能は継続評価）
- 見送り（暫定）:
  - 現条件での `hint=true` 常時採用

## 8. 次アクション

1. `hint=false` を現行比較の基準に維持。
2. `hint=true` は別条件（LR, PPO強度, imitation量）で再評価する。
3. 以後の速度比較は `exp_042 B` を新しい時間基準として扱う。

