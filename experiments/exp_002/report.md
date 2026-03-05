# Stage2 Experiment Report (Runbook 2)

作成日時: 2026-03-05 03:35 JST  
Source Runbook: `experiments/exp_002/runbook.md`  
対象: Runbook 2 実験A/B/C

## 1. 実験A: parallel eval worker ベンチ

条件固定:
- config: `configs/stage1_full_flat_mlp_ppo.yaml`
- seed: 42
- `selfplay.num_matches=50`
- `selfplay.num_workers=1`
- `evaluation.num_matches=20`
- `training.epochs=2`
- device: `training=cuda`, `selfplay=cpu`, `evaluation=cpu`

### 1.1 実行結果

| eval workers | run_dir | 総時間(s) | eval_before(s) | eval(s) | 総時間 speedup (vs 1) | eval speedup (vs 1) |
|---:|---|---:|---:|---:|---:|---:|
| 1  | `runs/20260305_stage1_full_flat_mlp_ppo_64051b2a` | 327 | 150 | 143 | 1.00 | 1.00 |
| 4  | `runs/20260305_stage1_full_flat_mlp_ppo_90900a47` | 132 | 48  | 47  | 2.48 | 3.04 |
| 8  | `runs/20260305_stage1_full_flat_mlp_ppo_e9480fc3` | 99  | 31  | 33  | 3.30 | 4.33 |
| 10 | `runs/20260305_stage1_full_flat_mlp_ppo_4300df92` | 82  | 26  | 23  | 3.99 | 6.22 |
| 12 | `runs/20260305_stage1_full_flat_mlp_ppo_2d075785` | 85  | 26  | 24  | 3.85 | 5.96 |
| 16 | `runs/20260305_stage1_full_flat_mlp_ppo_938db4bb` | 88  | 27  | 25  | 3.72 | 5.72 |
| 20 | `runs/20260305_stage1_full_flat_mlp_ppo_0aefab28` | 81  | 24  | 22  | 4.04 | 6.50 |

### 1.2 結論（実験A）

- 最速帯は `workers=10〜20`。
- 今回の最短は `workers=20`（81s）だが、`10`（82s）との差は小さく、実用上は同等。
- 10コア20スレッド構成を踏まえると、以後の標準は **`evaluation.num_workers=10`** が扱いやすい。

---

## 2. 実験B: PPO直学習の多seed比較

batch_dir: `runs/20260305_stage1_full_flat_mlp_ppo_batch_4a6c25d1`  
seed: 42,43,44,45,46  
条件: `selfplay.num_workers=8`, `evaluation.num_workers=10`

### 2.1 集約（after eval）

- `avg_rank`: **3.66 ± 0.0418**
- `avg_score`: **-17870 ± 2181**
- `win_rate`: **0.0040 ± 0.0059**
- `deal_in_rate`: **0.6123 ± 0.0597**
- 成功率: **5/5 (100%)**

### 2.2 学習前後差分（平均, eval_diff.delta）

- `avg_rank`: **-0.03**（微改善）
- `avg_score`: **-187**（悪化寄り）
- `win_rate`: **+0.00254**（微増）
- `deal_in_rate`: **+0.0112**（悪化）

### 2.3 フェーズ時間（seed平均）

- 総時間: **65.6s ± 4.4s**
- self-play: **6.4s**
- eval_before: **27.2s**
- eval: **28.8s**

---

## 3. 実験C: imitation → PPO の多seed比較

batch_dir: `runs/20260305_stage1_full_flat_mlp_imitation_then_ppo_batch_336246a3`  
seed: 42,43,44,45,46  
条件: 実験Bと同一（configのみ `imitation_then_ppo`）

### 3.1 集約（after eval）

- `avg_rank`: **3.55 ± 0.1414**
- `avg_score`: **-15657 ± 1071**
- `win_rate`: **0.0253 ± 0.0290**
- `deal_in_rate`: **0.5957 ± 0.0539**
- 成功率: **5/5 (100%)**

### 3.2 学習前後差分（平均, eval_diff.delta）

- `avg_rank`: **+0.09**（悪化）
- `avg_score`: **-1487**（悪化）
- `win_rate`: **-0.01185**（悪化）
- `deal_in_rate`: **-0.00171**（わずか改善）

### 3.3 フェーズ時間（seed平均）

- 総時間: **584.8s ± 4.2s**
- imitation: **468.2s**
- self-play: **59.6s**
- eval_before: **26.8s**
- eval: **27.8s**

---

## 4. 実験B vs 実験C（最終到達点比較）

`IMI - PPO`（after eval aggregate mean 差）:

- `avg_rank`: **-0.11**（IMI優位）
- `avg_score`: **+2213**（IMI優位）
- `win_rate`: **+0.0213**（IMI優位）
- `deal_in_rate`: **-0.0166**（IMI優位）

解釈:
- 最終指標だけ見れば、今回の5 seedでは **IMI→PPO が PPO直学習より良い**。
- ただし IMI run 内の `eval_before -> eval` は平均悪化で、PPO段の上積みが弱い。
- かつ実行時間は PPO 比で大きい（約 **8.9倍**）。

---

## 5. 結論

1. `evaluation.num_workers` は **10** を標準採用で問題なし（速度・運用のバランスが良い）。
2. 5 seedでは warm start 優位の傾向が見えるが、分散を考えると断定には追加試行が望ましい。
3. warm start は高コスト（imitation が支配的）なので、今後は
   - まず PPO直学習を標準路線に維持
   - warm start は比較目的で限定実施
   が妥当。

## 6. 次アクション候補

1. 同条件で seed を 10 本に拡張して有意性確認。
2. imitation データ生成の並列化（将来CQ）で時間コスト低減を検証。
3. warm start の imitation 設定（`imitation_matches`, filter）を減らした軽量版を比較。
