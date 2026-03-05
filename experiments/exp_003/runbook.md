# EXPERIMENT_STAGE3_RUNBOOK.md（Runbook 3）

最終更新: 2026-03-05  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel / PPO  
目的:  
1. PPO 直学習の平均性能と分散を、より信頼できる形で測る  
2. imitation → PPO の有効性を、10 seed で比較する  
3. warm start のコスト対効果を確認する  
4. 今後の標準実験設定を固める

---

## 0. この Runbook の位置づけ

Runbook 1 / 2 で確認できたこと:

- Stage 1 実験基盤は安定して動作する
- `evaluation.num_workers=10` は標準候補として妥当
- `selfplay.num_workers=10` も標準採用してよい
- `imitation.num_workers=10` も有効
- warm start は最終指標では有望な可能性がある
- ただし 5 seed ではまだ分散が大きく、断定には早い
- imitation は並列化で十分現実的な時間まで短縮できた

この Runbook 3 では、  
**「動く」「速い」から「比較結果として信頼できる」へ進む** ことを目的とする。

---

## 1. 基本方針

### 1.1 重視するもの
- 実験時間より **比較の信頼性**
- 単発の勝ち負けより **平均と分散**
- seed ごとの偶然より **設定としての傾向**

### 1.2 標準 worker 数
この Runbook では、以下を標準とする。

- `imitation.num_workers=10`
- `selfplay.num_workers=10`
- `evaluation.num_workers=10`

### 1.3 標準デバイス
- `training.device="cuda"`（使える環境なら）
- `selfplay.inference_device="cpu"`
- `evaluation.inference_device="cpu"`

### 1.4 比較で固定するもの
比較対象以外の条件は固定する。

- Observation: `full`
- Encoder: `flat`
- Model: `mlp`
- legal mask / reward / baseline
- worker 数
- eval 数
- self-play 数
- training.epochs
- seed 群

---

## 2. 事前確認

### 2.1 依存導入
```bash
pip install -e ".[dev]"
```

### 2.2 smoke test
```bash
python3 -m pytest tests/python/ -m smoke -v
```

### 2.3 config validation
```bash
python - <<'PY'
from pathlib import Path
from mahjong_rl.experiment import ExperimentConfig
from mahjong_rl.runner import Stage1Runner
cfg = ExperimentConfig.from_yaml(Path("configs/stage1_full_flat_mlp_ppo.yaml"))
print(Stage1Runner(cfg).validate_config())
PY
```

---

## 3. 共通設定

### 3.1 seed 群
この Runbook では、まず 10 seed を使う。

- `42,43,44,45,46,47,48,49,50,51`

### 3.2 比較の基本予算
今回は、質重視で以下を採用する。

- `selfplay.num_matches=200`
- `training.epochs=4`
- `evaluation.num_matches=50`

理由:
- Runbook 1 / 2 より明らかに大きい予算
- ただし極端に大きすぎず、10 seed 比較がまだ現実的
- 評価分散をかなり下げられる

### 3.3 成功判定
各 run / batch について以下を確認する。

- `summary.json.success == true`
- `phase_status` がすべて success
- `batch_summary.json` が生成される
- 主要指標4種が集約される

---

## 4. 実験A: PPO直学習 10 seed 比較

### 4.1 目的
- 現行 Stage 1 PPO の平均性能と分散を測る
- 以後の比較の基準（baseline）にする

### 4.2 config
- `configs/stage1_full_flat_mlp_ppo.yaml`

### 4.3 実行コマンド
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --override \
    selfplay.num_matches=200 \
    selfplay.num_workers=10 \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.epochs=4 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

### 4.4 見るもの
batch 集約レポートで以下を見る。

- `avg_rank` の平均 / 標準偏差
- `avg_score` の平均 / 標準偏差
- `win_rate` の平均 / 標準偏差
- `deal_in_rate` の平均 / 標準偏差
- `eval_before -> eval` 差分の平均
- 総実行時間
- 成功率

### 4.5 この実験の役割
以後の warm start 比較は、まずこの実験Aを基準にする。

---

## 5. 実験B: imitation → PPO 10 seed 比較

### 5.1 目的
- warm start が PPO 直学習に対して平均で有利か確認する
- imitation 並列化後の本命比較を行う

### 5.2 config
- `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`

### 5.3 実行コマンド
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --override \
    imitation.num_workers=10 \
    selfplay.num_matches=200 \
    selfplay.num_workers=10 \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.epochs=4 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

### 5.4 見るもの
- 主指標4種の平均 / 標準偏差
- `eval_before -> eval` 差分の平均
- imitation フェーズ時間
- 総実行時間
- PPO 直学習比の最終性能差
- コスト対効果

### 5.5 解釈ポイント
特に次を見る。

- **最終到達点で PPO より良いか**
- **eval_before -> eval で PPO 段が本当に改善しているか**
- **時間コストに見合うか**

---

## 6. 実験C: 軽量 imitation → PPO 10 seed 比較

### 6.1 目的
- warm start のコストを下げても効果が残るか確認する
- 「安くて効く warm start」があるかを見る

### 6.2 前提
この実験では、既存 imitation 設定を少し軽くする。  
もし軽量プリセットがあるならそれを使う。  
なければ override で imitation 側の量を減らす。

### 6.3 config
優先順:
1. 軽量 imitation 専用プリセット（あれば）
2. なければ `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`

### 6.4 推奨 override
軽量化の候補は次のいずれか、または複数。

- `imitation.num_matches` を減らす
- imitation 関連の epoch / sample 数を減らす
- imitation filter を有効化する（もし自然なら）

具体的な設定値は、現在の config に合わせて Codex 側で確認する。  
ただし、**PPO 部分の予算は実験A/Bと揃える**こと。

### 6.5 実行コマンド（例）
以下は例。実際の imitation 軽量化キー名はコードベースに合わせて調整する。

```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --override \
    imitation.num_workers=10 \
    imitation.num_matches=50 \
    selfplay.num_matches=200 \
    selfplay.num_workers=10 \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.epochs=4 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

### 6.6 見るもの
- 実験Bに対してどれだけ時間が減ったか
- それでも PPO より良いか
- 実験Bとの差（フル imitation と軽量 imitation の差）

---

## 7. レポートに必ず含める項目

### 7.1 実験A/B/C 共通
- batch_dir
- config
- override
- seed 群
- worker 数
- device 設定
- 成功率
- 総実行時間
- 主指標4種の平均 / 標準偏差
- 学習前後差分の平均

### 7.2 実験B/C（warm start 系）
追加で以下を入れる。

- imitation フェーズ平均時間
- self-play フェーズ平均時間
- eval_before / eval 平均時間
- PPO 直学習との最終性能差
- コスト対効果に関するコメント

---

## 8. 比較の仕方

### 8.1 第一比較
- 実験A vs 実験B
- 目的: warm start が平均で有利か

### 8.2 第二比較
- 実験B vs 実験C
- 目的: 軽量 imitation で十分か

### 8.3 判断基準
#### PPO を標準維持
以下なら PPO を標準維持でよい。
- imitation→PPO が平均で明確優位でない
- または優位でもコストが大きすぎる

#### warm start を標準候補
以下なら warm start を本格候補にする。
- 10 seed 平均で主指標が一貫して改善
- 標準偏差を見ても優位が崩れにくい
- コスト増に見合う改善がある

#### 軽量 warm start を有力候補
以下なら軽量版が有力。
- フル imitation に近い性能
- 実行時間が大きく短縮
- PPO より平均で優位

---

## 9. run 後に確認するもの

### 9.1 各 run
- `summary.json`
- `run.log`
- `notes.md`
- `eval/eval_metrics.json`
- `eval/eval_diff.json`
- `checkpoints/*.pt`

### 9.2 各 batch
- `batch_summary.json`
- `batch_table.csv` または `batch_table.jsonl`
- seed ごとの run_dir
- failure run があればその `summary.json` / `run.log`

---

## 10. この Runbook でまだやらないこと

この Runbook 3 では、以下はまだ対象外とする。

- モデル大型化比較
- CNN 導入
- PartialObservation 比較
- teacher/student 蒸留
- learner 分散化
- 複数サーバ分散実行

---

## 11. 推奨順序

1. **実験A: PPO 10 seed**
2. **実験B: imitation→PPO 10 seed**
3. **実験C: 軽量 imitation→PPO 10 seed**

---

## 12. メモ

- 今回は「速いか」より「信頼できる比較か」を優先する
- warm start は有望そうなので、PPO と対等に比較する価値が高い
- ただし warm start はまだ高コストなので、軽量版比較も重要
- 実験中に追加実装が必要だと判断した場合は、勝手に変更せずレポートする

---