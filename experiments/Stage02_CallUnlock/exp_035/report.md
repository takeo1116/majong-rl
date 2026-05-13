# Experiment Report: exp_035

作成日: 2026-05-10  
Stage: `Stage02_CallUnlock` / `Stage02b_OptionalActionUnlock`

## Summary

`exp_035` は、CQ-0290/CQ-0291 後に optional action unlock を全 ON にした short smoke である。

結論として、初回 run は `Stage2SelfPlayWorker` の buffer 初期化 bug で失敗したが、follow-up 修正後の再実行は 10 cycle を完走した。したがって、optional 全 ON の pipeline smoke は成功と判断する。

## Runs

| run | status | log | run_dir |
|---|---|---|---|
| initial smoke | failed | `driver_logs/20260510_192827_OPTIONAL_ALL_SMOKE_seed42.log` | n/a |
| rerun smoke | completed | `driver_logs/20260510_195627_OPTIONAL_ALL_SMOKE_seed42.log` | `runs/20260510_stage2b_optional_all_smoke_seed42_20fff85c` |

## Failure And Fix

初回 smoke は imitation までは進んだが、multi-cycle の `cycle_00` selfplay で全 worker が以下のエラーを出して停止した。

```text
'Stage2SelfPlayWorker' object has no attribute '_feat_buf'
```

原因は CQ-0291 実装中に `_read_optional_flag` を `staticmethod` として切り出した際、`Stage2SelfPlayWorker.__init__` 末尾にあるべき model setup と preallocated inference buffer 初期化が dead code 化していたこと。

follow-up で `_feat_buf` / `_mask_buf` / `_rc_buf` 初期化と `model.to(device).eval()` を `__init__` 末尾に戻し、regression tests を追加した。

## Successful Smoke Result

再実行 run:

```text
runs/20260510_stage2b_optional_all_smoke_seed42_20fff85c
```

10 cycle すべて完走し、最終 eval は以下。

| metric | value |
|---|---:|
| final avg_rank | 2.390 |
| final avg_score | 26352.0 |
| final win_rate | 0.2216 |
| final deal_in_rate | 0.1821 |
| total duration sec | 2398.963 |

Cycle eval ranks:

| cycle | avg_rank | win_rate | updates |
|---:|---:|---:|---:|
| 0 | 2.410 | 0.2469 | 481 |
| 1 | 2.490 | 0.2356 | 459 |
| 2 | 2.470 | 0.2280 | 475 |
| 3 | 2.360 | 0.2273 | 473 |
| 4 | 2.450 | 0.2435 | 480 |
| 5 | 2.295 | 0.2431 | 474 |
| 6 | 2.375 | 0.2290 | 476 |
| 7 | 2.480 | 0.2190 | 482 |
| 8 | 2.315 | 0.2332 | 480 |
| 9 | 2.390 | 0.2216 | 461 |

## Notes

- この smoke の目的は性能評価ではなく、optional 全 ON で selfplay / learner / eval が通るかの確認。
- smoke 実行時点では CQ-0292 batch 2 前だったため、`decision_family_counts` / `optional_decision_count` は summary に入っていない。
- CQ-0292 batch 2 後は optional family diagnostics が出るため、次の `exp_036` では family count も確認対象にする。

## Decision

`exp_035` は smoke 成功。次は optional 全 ON の 60 cycle 1seed probe (`exp_036`) に進む。
