# Experiment Report: exp_XXX

作成日: YYYY-MM-DD  
対象: `experiments/exp_XXX/runbook.md`

## 0. 記述ルール

- Git 管理下の参照は、必ず**リポジトリ相対パス**で書く
- **ローカル絶対パスは書かない**
- `runs/` 配下の run dir / `summary.json` / `run.log` などは、いつでも削除されうる前提で扱う
- `runs/` 配下を report の恒久参照先として書かない
- 実行時にしか取れない値は、「どの run dir を見たか」ではなく**数値そのものを report に転記して残す**

## 1. 実験概要

- 目的:
- 実行方式:
- seeds:
- 比較条件:
  - A:
  - B:
  - C:（必要なら）
- 主評価の優先順:

## 2. 実行結果

| 条件 | run label / batch label | success |
|---|---|---:|
| A | （run_map に対応を記録） | x/x |
| B | （run_map に対応を記録） | x/x |
| C | （run_map に対応を記録） | x/x |

注記:
- 再実行・手動実行・途中停止など、比較解釈に効く事実を短く残す

## 3. 主評価

mean ± std（seed=n）

| 条件 | 指標1 | 指標2 | 指標3 | 指標4 |
|---|---:|---:|---:|---:|
| A |  |  |  |  |
| B |  |  |  |  |
| C |  |  |  |  |

所見:
- 主評価優先順でどちらが良いか
- 差が小さいのか、明確差なのか

## 4. 副評価

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A |  |  |  |  |
| B |  |  |  |  |
| C |  |  |  |  |

所見:
- 主評価と整合するか
- 逆行する指標があるか

## 5. 補助観測

必要なものだけ入れる。

- learner 診断統計:
- self-play 統計:
- reward 分布:
- 教師再現率:
- 時間:
- 実行コスト:

## 6. 総合結論

1.
2.
3.

## 7. 今回の判断

- 採用:
- 保留:
- 見送り:

## 8. 次アクション

1.
2.
3.

## 9. 実行対応表

`run_map.json` は対応確認用であり、report には比較に必要な対応だけを転記する。
絶対パスや `runs/` 配下の run dir は書かず、run label / source label と参照文書で追えるようにする。

| seed | role | run label | source label | 参照 | 備考 |
|---|---|---|---|---|
| 42 | A | （run label） |  | `experiments/exp_XXX/run_map.json` |  |
| 42 | B | （run label） | （source label） | `experiments/exp_XXX/run_map.json` | reuse |
