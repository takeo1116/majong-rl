# Stage2a RL 長期安定性 独立レビュー (exp_022)

作成日: 2026-05-01
作成者: Claude Code (Anthropic) / model: Claude Opus 4.7 (1M context)
対象: `experiments/Stage02_CallUnlock/exp_022` の collapse / late drift
参照ソース:
- `python/mahjong_rl/stage2_selfplay_worker.py`
- `python/mahjong_rl/stage2a_learner.py`
- `python/mahjong_rl/runner.py`
- `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `experiments/Stage02_CallUnlock/exp_022/report.md`
- `experiments/Stage02_CallUnlock/exp_022/runbook.md`
- `experiments/Stage02_CallUnlock/exp_021/report.md`

---

## 1. ソースコードから見た collapse / late drift の根本原因

実験の症状を一行でまとめると、

> `max_prob_mean ≈ 0.97`、`log_ratio_p01 ≈ -35.2`、`ratio_max ≈ 7.89e6` の状態で、PPO clip が片側にだけ効いていない更新が積み重なり、cycle 40 で entropy collapse → avg_rank 崩壊。

これを起こす因果連鎖は、コード上、次のように特定できる。

### 1.1 (ほぼ確信) **mixed PPO の baseline サンプル `old_log_prob` の意味が壊れている**

`stage2_selfplay_worker.py` の `_infer_discard` (L487-499) を見ると、`actor_type == "baseline"` のサンプルでも `old_log_prob` として **その時点の policy が baseline 選択 action に与える log_prob** を記録している:

```python
log_probs = torch.log_softmax(out.discard_logits, dim=-1)
lp = log_probs[0, action].item()
```

ところが action 自体は `RuleBasedBaseline.select_discard` の決定的 argmax から来ている (同 L177-181)。
つまり「行動分布」は baseline の δ-distribution、「old_log_prob」は学習対象 policy の評価値、という IS としては一貫しない量を `ratio = exp(new_log_prob - old_log_prob)` に詰め込んでいる。

帰結:

- policy が鋭くなる (max_prob=0.97) と、tail action に対する `log π_θ(a|s)` は容易に -30〜-40 まで沈む
- baseline はそういう tail action を堂々と argmax で選ぶ (rule baseline は policy の鋭さを知らない)
- その bin で `old_log_prob ≈ -35`、1 epoch 後の `new_log_prob` がたとえ -20 程度に上がるだけでも `ratio = exp(15) ≈ 3.3e6`
- **これは観測されている `ratio_max ≈ 7.89e6` のオーダーと一致する**

### 1.2 (ほぼ確信) **PPO clip は片側 (adv<0 × ratio≫1+ε) で実質効いていない**

`stage2a_learner.py` (L1508-1514) の標準的 PPO 損失:

```python
log_ratio = action_log_probs - old_log_probs
ratio = torch.exp(log_ratio)
surr1 = ratio * batch_advantages
surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * batch_advantages
surr_min = -torch.min(surr1, surr2)
```

ここで:

- `adv > 0`, `ratio ≫ 1+ε` → `min(surr1, surr2) = surr2` で clip 側が選ばれる → ratio に grad が通らず安全
- `adv < 0`, `ratio ≫ 1+ε` → `surr1` (大きく負) のほうが `surr2` (やや負) より小さいので `min(surr1, surr2) = surr1` → `surr_min = -ratio·adv` (大きい正値) が選ばれ、**`ratio` 経由で grad が流れる**。`d(loss)/d(log_prob) ≈ ratio·adv` のオーダーで、`ratio = 7e6` なら桁外れの一撃で log_prob が押し下げられる

これは PPO の有名な non-symmetry で、教科書通りであれば `target_kl` early-stop か KL penalty で吸収する想定。本実装にはどちらも入っていない (`grep -n "target_kl\|approx_kl\|early_stop" python/mahjong_rl/stage2a_learner.py` は空)。

### 1.3 (有力) **`gae_lambda = 0.0` + reward backfill (CQ-0274) + `value_loss_coef = 0.125`**

`stage2a_learner.py` (L1226-1258) で advantage = TD-error 一発。lambda=0 なので value 関数の偏りがそのまま advantage に乗る。

- baseline config は `gamma=0.50, gae_lambda=0.0, value_loss_coef=0.25` だが、**exp_022 runbook では `value_loss_coef=0.125`**。value がさらに underfit しやすい
- reward は CQ-0274 で「次の同 player decision まで累積」型 → 局終了直前 sample に終局点棒変動が一括計上される。adv の variance が高い
- value 校正が cycles を跨いで slowly drift → late cycle で adv が系統的に偏る → 1.1/1.2 と組み合わさると suppression がさらに加速

### 1.4 (有力) **anchor も entropy_coef も off で「摩擦ゼロ」**

`exp_022` は `policy_anchor.enabled=false`, `entropy_coef=0`。CQ-0240 の anchor KL は `_compute_anchor_kl_*` でしか発火しないので、exp_022 では `_anchor_model is None` → 0 を返す (`stage2a_learner.py` L1531-1533)。
entropy_coef=0.003 でも止まらなかったのは、1.1/1.2 のスパイク勾配の桁が違う (entropy 補正は O(1)、ratio·adv は O(10^6)) からで、**摩擦不足というより一発勾配の暴走**を見ていると考えるほうが筋が通る。

### 1.5 (中) **late samples in single cycle = 自然な off-policy ドリフト**

`epochs=1` だが、200 match × 数十 decisions/match → minibatch 数は数十〜100 オーダー。permutation 後、後半 minibatch の sample は前半 minibatch の更新を全部食らった後で評価される。これが「同 cycle 内 epoch=1 でも `log_ratio_p01=-35`」が発生しうる量的根拠。entropy collapse によって tail action がさらに尖ると、後半の minibatch ほど extreme ratio が出やすくなる。

---

## 2. 影響度順 — 実装/設計/実験設定の問題

| 順位 | 問題 | 種別 | 影響 |
|---|---|---|---|
| **1** | `actor_type=baseline` サンプルの `old_log_prob` を「現 policy の評価値」で埋めている | 実装/設計 | mixed PPO で `ratio_max ≫ 10^6` が出る原動力。collapse の主犯候補 |
| **2** | PPO に `target_kl` early-stop も KL penalty も無い | 実装欠落 | 1 の「片側 grad 暴走」を吸収できない |
| **3** | `gae_lambda=0.0` + reward backfill + `value_loss_coef=0.125` | 設定 | advantage が value のずれを直接拾い、後期 cycle で系統 bias |
| **4** | exp_022 で anchor / entropy_coef が両方 off | 実験設定 | 摩擦ゼロでドリフトが線形に蓄積 |
| **5** | `selfplay.temperature=1.0` で、policy が鋭くなった後も exploration を上げない | 設定 | tail action のサンプリング頻度が下がり、当たった時の old_log_prob が深く沈む = 1 を悪化 |
| 6 | value diagnostics が無い (`explained_variance`, value MSE per cycle) | 実装欠落 | 3 を事前検知できない |
| 7 | mixed PPO の baseline `value` も「現 model の予測」で埋めるので、value 学習対象としては OK だが、baseline action による報酬実現と一致しない (=value にとっても off-distribution) | 設計 | 影響は中程度。3 と複合 |

ノイズレベルの問題 (rebuild 漏れ、observation dim ずれ等) は影響軽微なので割愛。

---

## 3. 「実装なしで先に切れる実験」(優先度順)

config 差し替えだけで切れるもの。`exp_022` を baseline に、1 軸ずつ動かす。

1. **anchor (lagged) を弱く再投入**: `policy_anchor.enabled=true, type=kl, reference=lagged_policy, coef=0.5, update_interval_cycles=5`。anchor 自体は exp_021 で「短期改善は B(no-anchor) のほうが上」だったが、長期では anchor が必要、という仮説を 30→60 cycle で確認。
2. **`baseline_sample_weight=0` にして mixed → policy-only PPO**: rule_mix で baseline 行動データは集めるが PPO ratio には乗せない。collapse の主因が 1.1 ならここで治る。
3. **`entropy_coef=0.01` (失敗した 0.003 の ~3x)** + `target_kl` 無し: entropy bonus 単独で止められるかの線引き。止められなければ「一発勾配が主犯」が裏取り。
4. **`gae_lambda=0.5` (lambda > 0)** に変更: value bias を smooth できれば collapse タイミングが後ろにずれるはず。
5. **`selfplay.temperature=1.5`**: tail サンプル数を増やす。`old_log_prob` の極端値が薄まり ratio_max が下がる予想。

費用対効果が高いのは **1 と 2**。3-5 は「症状を遅らせる」効果のテストで、3 は entropy_coef のスイートスポット決定にも使える。

---

## 4. 実装が必要な対策 (優先度順)

### P0 — 最初に切るべき

**A. PPO `target_kl` early-stop**

- minibatch 単位で approximate KL `mean((ratio-1) - log_ratio)` を集計し、累積 KL > `target_kl` なら残りの minibatch をスキップ
- config: `training.target_kl: 0.02` (default `None` で disable)
- 理由: 1.2 の non-symmetry を直接押さえる、最も標準的な PPO 安全装置。コード変更小さい。

**B. mixed PPO baseline サンプルの `old_log_prob` 設計修正**

2 通りある:

- B1 (推奨, structural): `actor_type=baseline` の sample を **PPO ratio から完全に除外**。imitation 経路 (`rule_mix_learner.ppo_mode="separated"` の baseline_imitation stage) に寄せ、value 学習にだけ使う。
- B2 (互換): selfplay 時に baseline サンプルの `log_prob = 0.0` で固定 (= behavior が決定的 1-hot とみなす)。一見シンプルだが PPO ratio がほぼ「現在 log_prob を底上げする方向の grad だけが流れる」になり別の歪みが出る。**B1 を強く推す**。

理由: 1.1 が ratio 暴走の構造的原因であり、A の target_kl 入れても tail サンプルが 1 個入るたびに early-stop が発動して学習効率を奪うだけ。

### P1

**C. Lagged anchor を default に戻す + warm-up を持たせる**

- `policy_anchor.enabled=true, reference=lagged_policy, coef=0.5, update_interval_cycles=5, warmup_cycles=5`
- 既存実装あり (`runner.py` L2010-2029)。config 変更だけ → 実は P1 ではなく実験 (3 章) で十分

**D. value 健全性 diagnostics**

- `explained_variance = 1 - Var(returns - values) / Var(returns)`
- `value_pred_mean`, `value_pred_std`, `value_target_std`
- cycle ごとに `summary.json.phase_stats.cycles[*].learner_metrics.value_diag` に記録
- 実装小、CQ-0281 の枠を踏襲できる

### P2

**E. selfplay temperature の cycle スケジュール**

- `selfplay.temperature_schedule: [1.5, 1.0, 0.8]` のような cycle 比例 anneal
- ただし C と D を入れた後で効果検証

**F. lr decay across cycles**

- linear decay or step。target_kl が効くなら lr decay は二の次で、まず A/B を先

---

## 5. 「target_kl / lr decay / entropy_coef / value/advantage 改善 / selfplay 分布」のうち、いま最適な選択

**現コードと exp_022 結果から判断する限り、最有力は次の 2 段階**:

1. **target_kl early-stop (A)** + **mixed PPO baseline 除外 (B1)** をセットで入れる
   - 観測症状 (`log_ratio_p01=-35`, `ratio_max=7.89e6`) は「分布が広いから collapse」ではなく「一発勾配が暴走しているから collapse」の signature。これに最も直接効くのが target_kl と、暴走の燃料 (= 不正確な old_log_prob を持つ baseline サンプル) を断つこと
2. その上で **selfplay temperature を 1.0 → 1.2-1.5** に上げる (E)。tail action のサンプル数を増やし、推定の variance を下げる

**選ばれにくいもの**:

- `entropy_coef` 単独は効果薄い予想。`0.003 → 0` の差で動かない実測がある以上、単独で 1.2 の non-symmetry 暴走を押さえるには `entropy_coef ≥ 0.05` 必要だが、その値だと imitation との乖離が大きすぎて bias を残す
- `lr decay` は対症療法。1 cycle 内の最後の minibatch で勾配が暴走する以上、lr を下げても暴走の閾値が下がるだけで、構造は変わらない
- `value/advantage 改善 (gae_lambda 引き上げ)` は確かに 1.3 を緩めるが、1.1/1.2 を放置したまま入れても collapse タイミングを 5-10 cycles 遅らせる程度。優先度は target_kl/baseline 除外より下

つまり「**ratio 制御 (target_kl) + ratio の不正源を絶つ (mixed PPO baseline 除外)**」が、いま現コード+結果に最もフィットする一手。

---

## 6. 次に切るべき CQ 案

`docs/CHANGE_QUEUE.md` の規約に合わせ、`CQ-0282`〜`CQ-0285` あたりで提案 (番号は実際に切る時調整)。

### CQ-0282 — Stage2a PPO target_kl early stop

- Type: RL | Training | Test
- Priority: **High**
- Title: PPO minibatch 単位 approx KL 累積で `target_kl` 超過時に早期停止
- 要点:
  - `_ppo_discard_epoch` / `_ppo_call_epoch` で minibatch 後に `approx_kl = mean((ratio-1) - log_ratio)` を集計
  - epoch 累積 KL > `training.target_kl` で残り minibatch を break (epoch 終了)
  - `ppo_diag` に `approx_kl_mean`, `early_stop_fraction`, `n_early_stops` を追記
  - default `target_kl: null` (disable) で既存挙動互換、実験 config では `0.02` を初期値として推奨
- 受け入れ条件: target_kl=0.001 で early stop が発火、target_kl=null では発火しない、既存 PPO テスト pass

### CQ-0283 — mixed PPO で baseline サンプルを ratio から除外

- Type: RL | Training | Test
- Priority: **High**
- Title: `actor_type=baseline` を PPO 政策更新から除外し、value/imitation のみに使う
- 要点:
  - `_compute_ppo_branch_targets` で baseline サンプルの `weight=0` (policy update 寄与ゼロ)
  - もしくは `_train_ppo` 時に `actor_type=="policy"` のみで sample を絞り、baseline は別 stage で imitation/value のみ実行
  - `rule_mix_learner.ppo_mode="separated"` の挙動を default 化検討
  - `summary.json.cycles[*].learner_metrics` に `n_policy_samples_used`, `n_baseline_samples_excluded` を記録
- 受け入れ条件: baseline sample を含む shard で PPO ratio 統計に baseline 由来の値が混じらない、既存 separated mode と挙動一致

### CQ-0284 — value 関数 健全性 diagnostics

- Type: RL | Eval | Test
- Priority: Medium
- Title: PPO learner に explained_variance / value pred-target stats を追加
- 要点:
  - `ppo_diag.value_diag` に `explained_variance`, `value_pred_mean`, `value_pred_std`, `value_target_mean`, `value_target_std`, `value_mse`
  - branch 別 (discard/call) も
  - 既存 value_loss は維持
- 受け入れ条件: explained_variance が `[-1, 1]` でクランプ前生値、片符号 only/空 branch でも crash しない

### CQ-0285 — Stage2a selfplay temperature schedule

- Type: Training | Test
- Priority: Medium (CQ-0282/0283 後に実験で必要なら確定)
- Title: cycle 比例で selfplay temperature を anneal
- 要点:
  - `selfplay.temperature_schedule: {start: 1.5, end: 1.0, mode: "linear"}` の dict 形式追加
  - 既存の scalar `selfplay.temperature` も後方互換維持
  - Stage2a multi_cycle で cycle index → temperature を解決、`run_stage2a_selfplay_parallel` に渡す
- 受け入れ条件: schedule なし時は既存 scalar 動作と一致、warmup→終端 cycle で線形変化、parallel worker にも temperature が伝わる

---

## まとめ (1 行)

> exp_022 の collapse は entropy 不足ではなく、**mixed PPO の不整合な `old_log_prob` が引き起こす ratio 暴走を、PPO clip の片側非対称が止められないこと**が主因。次の打ち手は **CQ-0282 (target_kl early-stop) + CQ-0283 (baseline sample を ratio から除外)** をセットで入れ、その上で anchor/temperature を再評価する流れが、コードと観測に最もフィットする。

---

## 署名

このレビューは Anthropic の CLI コーディング・アシスタント Claude Code が、
ユーザー (takeo1116) の依頼を受けて、リポジトリ内のソースコードと
実験結果のみを直接参照して独立に作成したものです。

- 作成: Claude Code (Anthropic)
- model: Claude Opus 4.7 (1M context)
- 作成日: 2026-05-01
- 対話セッション: ローカルの Claude Code CLI

(その後、`exp_023` で本レビューが指摘した「mixed PPO の baseline actor sample
を PPO ratio から除外する」変更が 3 seed × 60 cycle で 2.985 → 2.199 (tail10)
の改善を出し、CQ-0282 として default 化された。)
