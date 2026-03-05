# Experiments Directory Policy

このディレクトリは、実験計画（runbook）と実験結果（report）を
1 対 1 で管理するための場所である。

## 1. ディレクトリ構成

- 実験は `exp_XXX` ディレクトリ単位で管理する。
- 各 `exp_XXX` には以下 2 ファイルを必須とする。
  - `runbook.md`
  - `report.md`

例:

```text
experiments/
  exp_001/
    runbook.md
    report.md
  exp_002/
    runbook.md
    report.md
```

## 2. 運用ルール

- `report.md` を作る前に、必ず `runbook.md` を作成する。
- `runbook.md` がない実験の `report.md` は作成しない。
- `report.md` には対応する runbook への参照を記載する。
  - 例: `Source Runbook: experiments/exp_001/runbook.md`

## 3. Git 管理方針

- `experiments/` 配下の `runbook.md` / `report.md` は Git 管理する。
- 実験の生データ（`runs/`）は Git 管理しない（`.gitignore` で除外）。
