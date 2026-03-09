# experiments/exp_XXX/runbook.md

最終更新: YYYY-MM-DD  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: この実験で切り分けたいことを 1 文で書く

---

## 0. この実験の位置づけ

- 直前までの結論:
- なぜ今この比較をするのか:
- この実験で更新したい判断:

## 1. この実験の問い

この runbook で答えたい問いを 2〜4 個に限定して書く。

1.
2.
3. （必要なら）

## 2. 実験方針

### 2.1 比較軸
- A:
- B:
- C:（必要なら）

### 2.2 共通固定
- config:
- seeds:
- model / encoder:
- 固定 override:

### 2.3 交絡回避
- 何を固定するか:
- 何を変えてよいか:
- reuse を使う場合の理由:

## 3. 実行方式

### 3.1 実行単位
- 単発 / batch / reuse / driver

### 3.2 既存実験からの流用
- 参照可能な既存 run:
- 流用するもの:
- 新規実行するもの:
- 実データ確認:
  - `runs/` 配下の参照 run が残っているか:
  - 残っていない場合、必要な値が `report.md` に転記されているか:
- 再実行が必要な理由:

### 3.2 reuse を使う場合
- 参照 run の作り方:
- `--reuse-phases`:
- 参照元と分岐先で一致必須のキー:

### 3.3 run_map
- `experiments/exp_XXX/run_map.json` をどう使うか
- report に最終的に何を転記するか

## 4. 実行コマンド

```bash
# 条件A
python -m mahjong_rl.cli ...
```

```bash
# 条件B
python -m mahjong_rl.cli ...
```

```bash
# 条件C（必要なら）
python -m mahjong_rl.cli ...
```

## 5. 成功判定

### 5.1 共通
- `summary.json.success == true`
- 必須成果物:
  - `summary.json`
  - `notes.md`
  - `config.yaml`

### 5.2 評価成果物
- `eval/eval_rotation.json` または `eval/eval_metrics.json`
- `eval/eval_diff.json`（必要な実験のみ）
- `batch_summary.json` / `batch_table.csv`（batch 実行時）

### 5.3 追跡キー
- `summary.phase_stats...`
- `summary.encoder_features...`
- `metrics/train_metrics.json`（必要な実験のみ）
- `summary.reuse_info...`（reuse 時）

## 6. 主評価と副評価

### 6.1 主評価
- 何を優先して採否判定するか

### 6.2 副評価
- after 指標:
- 補助指標:

### 6.3 比較優先順
- 例: `Δavg_rank -> Δavg_score -> Δdeal_in_rate -> Δwin_rate`

## 7. 集計方法

- どのファイルを正とするか:
- mean/std の単位:
- seed 対応の取り方:
- offline 集計が必要なもの:

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
- 長時間実行時の注意:
- 交絡要因:
- 再開方針:
- 計算時間見積もり:

## 9. レポートに必ず含める項目

- 条件一覧
- 実行対応表
- 主評価表
- 副評価表
- 補助観測（必要時）
- 結論
- 次アクション

## 10. 次アクション判定

- どの結果なら採用:
- どの結果なら却下:
- どの結果なら追加診断:
- 次に回すべき実験:

## 11. 作成前チェック

- [ ] 既存実験との条件重複を確認し、流用可否を判断した
- [ ] 参照する既存 run の実データが残っているか、または必要値が `report.md` に転記済みかを確認した
- [ ] 再実行する条件について、流用しない理由を明記した
