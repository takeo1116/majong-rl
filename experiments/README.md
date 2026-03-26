# Experiments Directory Policy

このディレクトリは、**段階ごとの実験群**を整理し、
各実験の計画 (`runbook.md`) と結果 (`report.md`) を追跡するための場所である。

実験は単体の `exp_XXX` で完結させるのではなく、
まず大きな学習段階 (`StageXX_*`) に分け、その下に個別実験を並べる。

## 1. ディレクトリ構成

現在の基本構成は次のとおり。

```text
experiments/
  README.md
  Stage01_DiscardOnly/
    README.md
    exp_001/
      runbook.md
      report.md
    exp_002/
      runbook.md
      report.md
    ...
  Stage02_CallUnlock/
  _template/
```

### 役割

- `StageXX_* / README.md`
  - その段階全体の目的、主結果、current best、次ステージへの引き継ぎをまとめる
- `StageXX_* / exp_XXX / runbook.md`
  - 個別実験の計画、比較条件、評価軸を記録する
- `StageXX_* / exp_XXX / report.md`
  - 個別実験の結果と解釈を記録する
- `StageXX_* / exp_XXX / run_map.json`
  - 必要な場合のみ置く。`runs/` 削除後でも実行単位を追跡できるようにするための補助ファイル
- `StageXX_* / exp_XXX / bug_report.md`
  - bugfix 系実験で必要な場合のみ置く

## 2. 命名ルール

- 段階ディレクトリは `StageXX_Description` とする
  - 例: `Stage01_DiscardOnly`, `Stage02_CallUnlock`
- 実験ディレクトリは `exp_XXX` を基本とする
- bugfix を主目的とする段階的区切り実験は、必要に応じて
  - `exp_030_bugfix`
  - `exp_065_bugfix`
  のようにサフィックスを付けてよい

## 3. 運用ルール

- `report.md` を作る前に、必ず `runbook.md` を作成する
- `runbook.md` がない実験の `report.md` は作成しない
- `report.md` には、対応する `runbook.md` と必要な参照元を明記する
- 個別実験の詳細は `exp_XXX` に閉じ、段階全体の結論だけを `StageXX_* / README.md` に残す
- stage をまたぐ比較や current best の更新は、まず個別 report に残し、その後 stage README や `docs/PROJECT.md` に必要最小限だけ反映する

## 4. Git 管理方針

Git 管理するもの:
- `StageXX_* / README.md`
- `exp_XXX / runbook.md`
- `exp_XXX / report.md`
- `exp_XXX / run_map.json`（必要時のみ）
- `bug_report.md`（必要時のみ）

Git 管理しないもの:
- `runs/` 配下の生データ
- `driver_logs/` などの大量ログ
- 一時集計やローカル検証用の中間ファイル

## 5. 読む順番

その段階の状況を短時間で把握したいときは、次の順で読む。

1. `experiments/StageXX_*/README.md`
2. current best 候補の `exp_XXX/report.md`
3. 必要なら対応する `runbook.md`

細かい数値やログの由来まで追う必要があるときだけ、`run_map.json` や個別補足資料を見る。
