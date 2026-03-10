# PROJECT.md

最終更新: 2026-03-11  
この文書の役割: プロジェクトの大目標・現在地・次アクションを短時間で復元する意思決定ハブ。

---

## 1. 北極星（不変）

最終目標は、麻雀AIの学習基盤を段階的に拡張し、最終的に実戦ルールに近い環境で強い方策を学習できる状態に到達すること。  
短期的には、**imitation 初期方策を PPO で壊さず改善する条件の確立**が最優先。

---

## 2. 現在地（2026-03-10）

フェーズ: **Stage1後半（value 診断改善と policy 更新安定性のトレードオフ切り分けフェーズ）**

- できている
  - runbook/report/driver 運用、phase 再利用、resume
  - imitation 学習の成立
  - PPO ハイパラ基礎探索
  - `shanten_hint` と `tie_aware_best_set` の導入・比較
  - learner 診断統計（`ppo_diag`）の成果物化
- 未解決
  - 固定した shaping 条件と joint imitation 条件の下でも、`improve/worsen` 群で advantage の符号が整合しない理由
  - value 診断改善と通常評価改善がなぜ乖離するのか
  - hidden size 拡大で `clip_fraction` / `ratio_std` が悪化する理由
  - 単純な `lr` / `epochs` 弱化で上記トレードオフを解けない理由
  - value/target 改善と policy 更新安定性の両立条件

---

## 3. Current Focus（今の実験目的）

現在の主目的は次の1点。

> reward shaping 条件と joint imitation 条件を固定した上で、`shanten_diag` / `turn_diag` と PPO 更新安定性指標を併せて見て、value 診断改善と通常評価悪化の乖離、そしてそのトレードオフが単純な PPO 弱化で解けるかを説明する。

現行固定軸（診断基準点）:

- `feature_encoder.shanten_hint=true`
- `training.imitation_loss_mode=tie_aware_best_set`
- `training.lr=0.0001`
- `training.epochs=4`
- `training.value_loss_coef=0.25`
- `training.clip_epsilon=0.2`
- `training.batch_size=256`
- `training.gamma=0.99`
- `training.gae_lambda=0.95`

評価の優先順:

1. `eval_before -> eval` の delta（悪化幅）
2. after 指標（`avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`）
3. learner 診断統計（`advantage/return/value_error/ratio/clip_fraction`）
4. self-play / reward 分布

---

## 4. これまでの意思決定ログ（Decision Log）

短く履歴を残し、同じ探索を繰り返さないための要約。

- `exp_005〜013`: PPO ノブ探索  
  - 採用寄り: `lr=1e-4`, `epochs=4`, `value_loss_coef=0.25`, `clip=0.2`, `batch_size=256`, `gae=0.95`
  - ただし PPO 後悪化は残存
- `exp_014`（imitation-only, shanten_hint）  
  - on は小幅改善も決定打なし
- `exp_015`（warm start + PPO, shanten_hint）  
  - on は総合で悪化、単純採用見送り
- `exp_016`（教師再現率診断）  
  - shanten_hint は teacher 再現には効く
- `exp_017`（大型比較: strict vs tie-aware）  
  - imitation-only では tie-aware 優位
  - PPO込みでは優位が縮小、明確勝者なし
- `exp_018`（更新強度比較）  
  - baseline（`epochs=4, lr=1e-4`）を維持
  - `epochs=2` は明確に悪化、`lr=5e-5` は learner 診断統計を穏やかにするが主評価では更新できず
  - PPO 悪化の主因は更新強度だけではなく、reward / target 品質側も疑う段階へ進んだ
- `exp_019`（PPO 診断: baseline vs weak-lr）  
  - `lr=5e-5` でも主評価は baseline を更新できず、`clip_fraction / ratio` の軽減だけでは不十分と確認
  - reward の sparse 性と target 品質側の診断を優先する方針が強化された
- `exp_020`（minimal shaping reward）  
  - `shanten_delta_reward` を加えると、PPO 後悪化は baseline より一貫して小さくなった
  - `linear_decay` が `constant` より自然で、主評価・after 指標とも最良
  - sparse reward 主因説はかなり強化されたが、悪化はまだ完全には消えていない
- `exp_021`（linear_decay scale sweep）
  - 実用域は `0.01〜0.02`
  - 総合推奨は `scale=0.01`
  - `scale=0.1` は過剰 shaping として明確に悪化
- `exp_022`（mode 比較）
  - `mode=both` を維持採用
  - `mode=improve_only` は baseline も更新できず不採用
  - 暫定標準 reward 条件は `linear_decay + scale=0.01 + mode=both`
- `exp_023`（shanten-conditioned learner 診断）
  - `improve` 群の advantage mean は依然として負、`worsen` 群の advantage mean は依然として正
  - baseline reward と標準 shaping reward で `shanten_diag` はほぼ変わらず、reward sparse 性だけでは残差を説明できない
  - 次段は reward 探索より value/target 仮説を優先する
- `exp_024`（joint imitation + value current_shanten）
  - `imitation_value_warmstart.coef=0.1` は通常評価（`eval_before -> eval` / after 指標）を改善
  - ただし `shanten_diag` の符号整合は回復せず、`improve` はなお負、`worsen` はなお正
  - value current_shanten の追加価値は見えず、当面は不採用
  - 暫定標準候補は「reward shaping 標準 + joint imitation coef=0.1 + current_shanten off」
- `exp_025`（B 条件単条件診断）
  - `shanten_diag` に `old_value/new_value/value_update_delta` を追加した診断で、`improve` 群の value misfit が最も大きいことを確認
  - `turn_diag` では `late` bucket の `value_error` が `early/mid` より大きく、終盤で value misfit が強い
  - 次段は reward 探索ではなく、value/target 改善仮説を比較する
- `exp_026`（大きいモデル + value current shanten）
  - `shanten_diag` / `turn_diag` / global `value_error` は改善し、表現力不足仮説は一定程度支持された
  - ただし通常評価は `exp_025` を更新できず不採用
  - 診断改善と通常評価改善が一致しないため、次は改善要因の分離が必要
- `exp_027`（強化版 value 表現のスケーリング）
  - `[768,384]` / `[1024,512]` でも `shanten_diag` / `turn_diag` / global `value_error` はさらに改善した
  - 一方で通常評価はさらに悪化し、`clip_fraction` と `ratio_std` も悪化した
  - value 表現強化そのものは有効だが、現状では policy 更新安定性とのトレードオフが強く、単純なサイズ拡大は採用しない
- `exp_028`（大きめモデル条件で PPO 更新強度を弱化）
  - `weak-lr` は `clip_fraction` / `ratio_std` をほぼ `exp_025` 近傍まで戻し、通常評価も `exp_027 A` より回復した
  - ただし `shanten_diag` / `turn_diag` の value 診断改善はほぼ失われ、`exp_025` 近傍へ戻った
  - `weak-epochs` は value 診断改善を維持したが通常評価はさらに悪化した
  - 単純な `lr` / `epochs` 弱化だけでは、value 診断改善と通常評価改善の両立はできない

---

## 5. Do Not Re-test Yet（再検証保留）

以下は「今の実装条件のまま」では優先度を下げる。

- `shanten_hint` の on/off 単純比較（PPO込みで on 採用根拠が弱い）
- `value_loss_coef=1.0`（悪化傾向）
- 極端な更新強度（例: epochs 過大、clip 過大）
- `epochs=2` の weak-epochs 比較（baseline より明確に悪化）
- `lr=5e-5` の weak-lr 比較（診断対照としては有用だが、改善条件としては未採用）
- reward shaping なしの baseline reward 単独運用を「十分」とみなすこと
- `scale=0.1` の極端 shaping
- `mode=improve_only`
- reward shaping の追加比較を続けること（当面は learner/value 仮説を優先）
- `model.value_features.current_shanten.enabled=true` の追加比較（現時点では価値が見えない）
- reward shaping の追加探索（value/target の残差説明が先）
- hidden size 拡大 + current shanten 同時導入を、そのまま採用すること
- 大きめモデル条件で `lr` / `epochs` を単純に弱めれば解決するとみなすこと

再検討条件:

- reward 設計や target 定義を変更したとき
- ルール拡張で学習分布が変わったとき

---

## 6. 可観測性（いま見えるもの）

run 成果物で確認可能:

- `eval_before/eval/eval_diff`（rotation対応）
- self-play 統計（wins/deal-ins/draws/tsumo/ron/ryukyoku/num_rounds）
- reward 内訳統計（`point_delta/shanten_delta/total` の `mean/std/p50/p90/p99`）
- reward shaping 設定（`enabled/scale/mode/schedule_type`）
- imitation metrics
  - `value_loss`
  - `imitation_value_warmstart`
  - `model_features.value_features.current_shanten.enabled`
- learner 診断統計 `ppo_diag`
  - `advantage_*`, `return_*`, `old_value_*`, `value_error_*`, `ratio_*`, `clip_fraction`
  - `shanten_diag`
    - `improve/same/worsen` ごとの `advantage/return/old_value/new_value/value_update_delta/value_error`
    - `available_samples/unavailable_samples/status`
- `turn_diag`
    - `early/mid/late` ごとの `advantage/return/old_value/new_value/value_update_delta/value_error`
- 大きいモデル + current shanten 条件での同診断（`exp_026`）との比較
- batch 側 run 別 learner 診断統計

不足が出たら CQ 化して可観測性を先に上げる（推測で進めない）。

---

## 7. Trigger to Move Stage（次段へ進む条件）

「PPO 改善が回る」と判定する目安（暫定）。

必須条件:

1. 主比較条件で `eval_before -> eval` の悪化が消える（少なくとも 5 seeds で一貫して改善または非悪化）
2. `shanten_diag` を含む learner 診断統計が、少なくとも改善方向と矛盾しない
3. 追加1回の再現実験でも同傾向が出る

達成後の次段:

- ゲームルールを段階的に実麻雀へ拡張
- 各拡張ごとに runbook を分離し、劣化点を局所化
- 「実麻雀への近似」と「学習安定」の両立を判定基準にする

---

## 8. 運用原則

1. 実験は runbook 定義→レビュー→実行→report の順で進める  
2. 実行は原則 `scripts/local/exp_XXX_driver.py` で自動化する  
3. `run_map.json` はローカル管理、最終対応表は `report.md` に残す  
4. レビューNG時は旧CQを削除し、新規CQで再定義する  
5. 仕様変更（CQ）と実験解釈（report）を混ぜない

---

## 9. 更新ルール（この文書）

各実験完了後に最低限更新する:

1. `Current Focus`（必要なら）
2. `Decision Log`（1〜2行追加）
3. `Do Not Re-test Yet`（採否変化があれば）
4. `Trigger to Move Stage`（判定条件が変われば）

詳細な数値表は `experiments/exp_XXX/report.md` に置き、この文書は判断軸だけを保つ。

---

## 10. 一文要約

**reward shaping と joint imitation で通常評価は前進したが、`improve/worsen` 群の advantage はまだ整合していない。value 表現を強めると診断値は一貫して改善する一方、通常評価はむしろ悪化し、さらに単純な `lr` / `epochs` 弱化でもそのトレードオフは解けなかった。次は policy-value 干渉または target 定義を本丸として切る必要がある。**
