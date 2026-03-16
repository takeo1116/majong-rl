# PROJECT.md

最終更新: 2026-03-16  
この文書の役割: プロジェクトの大目標・現在地・次アクションを短時間で復元する意思決定ハブ。

---

## 1. 北極星（不変）

最終目標は、麻雀AIの学習基盤を段階的に拡張し、最終的に実戦ルールに近い環境で強い方策を学習できる状態に到達すること。  
短期的には、**imitation 初期方策を安定して超える更新設計の確立**が最優先。  
現時点では「PPO を壊さず回す」段階はかなり進み、焦点は **学習信号の質** に移っている。

---

## 2. 現在地（2026-03-16）

フェーズ: **Stage1後半（anchor / multi-cycle / rule-mix まで実装し、PPO の「更新設計そのもの」を見直すフェーズ）**

- できている
  - runbook/report/driver 運用、phase 再利用、resume
  - imitation 学習の成立
  - PPO ハイパラ基礎探索
  - `shanten_hint` と `tie_aware_best_set` の導入・比較
  - learner 診断統計（`ppo_diag`）の成果物化
  - reward scale バグの特定と修正、post-fix baseline の再取得
  - `exclude_post_riichi_discards` の導入と learner/batch 成果物化
  - 高表現力モデル（`hidden_dims=[512,256] + dual_towers`）の成立確認
  - `policy_anchor` / `multi_cycle` / `rule_mix` / `mixed_ppo` の実装と完走確認
  - 実行速度の大幅改善により `20 seeds x 20 cycles` 規模の検証が可能になった
- 未解決
  - imitation 基準を長期的に安定して超える更新則
  - rule データを actor 改善に結びつける学習信号設計
  - `rule` 行動を PPO に入れるときの off-policy / target mismatch の扱い
  - 現行 PPO で改善しない主因が「ハイパラ不足」なのか「更新目標の不整合」なのかの切り分け

---

## 3. Current Focus（今の実験目的）

現在の主目的は次の2点。

> 1. `exp_050` / `exp_052` までで確認された「少し改善しても長期では沈む」挙動の原因を、更新設計レベルで切り分ける。  
> 2. `rule` を教師候補として活かすなら、BC / separated learner / mixed PPO のどれが機能するのか、あるいはどれも足りないのかを明確にする。

現行固定軸（診断基準点）:

- `feature_encoder.shanten_hint=true`
- `training.imitation_loss_mode=tie_aware_best_set`
- `training.imitation_value_warmstart.enabled=true`
- `training.imitation_value_warmstart.coef=0.3`
- `training.lr=0.0001`
- `training.epochs=2`
- `training.value_loss_coef=0.25`
- `training.clip_epsilon=0.2`
- `training.batch_size=512`
- `training.gamma=0.99`
- `training.gae_lambda=0.85`
- `training.exclude_post_riichi_discards.enabled=true`
- `model.hidden_dims=[512,256]`
- `model.policy_tower.enabled=true`
- `model.value_tower.enabled=true`
- `reward.point_delta_scale=0.0001`
- `reward.shaping.shanten_delta.enabled=true`
- `reward.shaping.shanten_delta.scale=0.01`
- `reward.shaping.shanten_delta.mode=both`
- `reward.shaping.shanten_delta.schedule.type=linear_decay`
- `training.policy_anchor.enabled=true`
- `training.policy_anchor.type=kl`
- `training.policy_anchor.coef=0.5`
- `training.policy_anchor.reference=imitation_fixed`
- `training.entropy_coef=0.0`

評価の優先順:

1. imitation 基準（`cycle0.eval_before`）に対して長期的に上回れるか
2. after 指標（`avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`）
3. 各 cycle 内 `eval_before -> eval` 差分
4. learner 診断統計（`ratio/clip_fraction/policy_anchor/mixed_ppo/shanten_diag`）

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
- `exp_029`（small model + `current_shanten=true` で tower 構造比較）
  - `value_tower only` は更新安定性は改善したが、通常評価・`shanten_diag`・`turn_diag` を悪化させ不採用
  - `policy_tower only` は after 指標・更新安定性・value 診断のバランスが最良で、暫定採用候補
  - `dual towers` は更新安定性は最良だが、通常評価で `policy_tower only` を更新できず保留
  - shared trunk 干渉仮説は部分的に支持され、少なくとも policy 側 tower は有効候補
- `exp_030`（baseline 単条件、reward / `delta_t` 分解診断）
  - `improve` / `worsen` の逆転は `advantage` だけでなく `reward` と `delta_t` の段階ですでに存在
  - 逆転の主因はまず `point_delta_reward` 群差で、`shanten_delta_reward` の符号自体は正しいが絶対値が小さい
  - value はその逆転をさらに増幅しているが、value 単独原因ではない
- `exp_031`（post-fix baseline 再取得）
  - reward scale バグ修正後の baseline を再取得
  - PPO 後悪化は残るが、診断値は正常スケールに戻った
- `exp_032`（post-fix `policy_tower_only`）
  - pre-fix で良く見えた `policy_tower_only` は post-fix では baseline を更新できず不採用
- `exp_033`（`exclude_post_riichi_discards=true`）
  - PPO 後悪化はかなり縮小
  - ただし `same > 0 / improve < 0` は残存し、主因ではないと確認
- `exp_034`（高表現力モデル + dual towers）
  - PPO 後悪化と `shanten_diag` は改善
  - ただし `clip_fraction` / `ratio_std` / `turn_diag` が悪化し、そのままでは不採用
- `exp_035`（高表現力モデルの更新強度調整）
  - `batch_size=512, epochs=2` が有効
  - `exp_034` より after 指標、更新安定性、turn 歪みが改善
  - 高表現力側の baseline 候補になった
- `exp_036`（`batch_size=1024` と lr 比較）
  - `batch_size=1024` は update は綺麗でも after 指標が大きく悪化
  - 高表現力条件では過度に保守的で不採用
- `exp_037`（`gae_lambda` / imitation value warmstart 比較）
  - `gae_lambda=0.90` 単独は診断改善があるが after では弱い
  - `coef=0.3` 単独は不採用
  - `gae_lambda=0.90 + coef=0.3`（D）が総合では最良候補
  - 現時点の主系列 baseline は `exp_037 D`
- `exp_044`（turn_context / huber / advantage clip）
  - `turn_context` は小幅改善、`huber` / `advantage clip` は決定打なし
  - 「value の安定化だけで PPO 崩れを止める」のは不十分と判断
- `exp_045`（long cycle 100, 1 seed）
  - 長期学習では、初期 imitation より悪化する方向が強く、自然回復は確認できなかった
  - 「更新を弱めればそのうち戻る」仮説はかなり後退
- `exp_046-050`（policy anchor + entropy）
  - `policy_anchor(kl, coef=0.5) + entropy=0.0` が最も有望
  - ただし `20 seeds x 20 cycles` の `exp_050` では、各 cycle 内差分は一部改善しても imitation 基準は超え続けられなかった
  - anchor は「壊れる速度を落とす」方向には効くが、長期改善の本質解ではない
- `exp_051-052`（rule_mix + two-stage learner）
  - `actor3 + rule1` 相当の rule_mix は anchor-only より少し良い
  - ただし `exp_052`（20 seeds x 20 cycles）でも imitation 基準超えは安定せず、平均最良は早期 cycle にとどまった
  - rule データを separated learner（baseline BC -> policy PPO）で入れるだけでは長期改善は作れない
- `20260316` ad-hoc: `mixed_ppo`, `policy_ratio=0.75`, `1 seed x 10 cycles`
  - 実装は意図どおり動作し、baseline BC を外して `policy + rule` を PPO に一本化できた
  - ただし seed 42 では cycle 0 だけ改善し、その後は imitation 基準より悪化
- `20260316` ad-hoc: `mixed_ppo`, `policy_ratio=0.0`, `1 seed x 10 cycles`
  - `num_policy_samples=0`, `num_baseline_samples>0` で、rule-only PPO 学習が成立することを確認
  - それでも cycle 0 の小改善後は長期悪化し、`rule` を今の PPO target にそのまま流すだけでは学習が進まないと確認

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
- `batch_size=1024` 系の再探索
- `policy_tower only` を post-fix baseline 候補とみなすこと
- `coef=0.3` 単独 warmstart
- `gae_lambda=0.90` 単独を即採用すること
- `model.value_features.current_shanten.enabled=true` の追加比較（現時点では価値が見えない）
- hidden size 拡大 + current shanten 同時導入を、そのまま採用すること
- 大きめモデル条件で `lr` / `epochs` を単純に弱めれば解決するとみなすこと
- advantage 逆転を「value だけの問題」とみなすこと
- `policy_anchor` や `entropy` の微調整だけで長期改善が出るとみなすこと
- `rule_mix + separated learner` をそのまま大規模に掘り続けること
- `mixed_ppo(policy_ratio=0.75, baseline_sample_weight=1.0)` を探索なしで本命化すること
- `policy_ratio=0.0` でも上がらない現状で、より複雑な混合条件にすぐ戻ること

再検討条件:

- reward 設計や target 定義を変更したとき
- ルール拡張で学習分布が変わったとき
- `rule` データの loss への入れ方を変更したとき（importance correction, auxiliary BC/KL, advantage-weighted imitation など）

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
  - `policy_anchor`
  - `mixed_ppo`
  - `shanten_diag`
    - `improve/same/worsen` ごとの `advantage/return/old_value/new_value/value_update_delta/value_error`
    - `available_samples/unavailable_samples/status`
- `turn_diag`
    - `early/mid/late` ごとの `advantage/return/old_value/new_value/value_update_delta/value_error`
- `shanten_diag` の追加内訳
    - `reward`
    - `point_delta_reward`
    - `shanten_delta_reward`
    - `delta_t`
- 大きいモデル + current shanten 条件での同診断（`exp_026`）との比較
- batch 側 run 別 learner 診断統計
- `post_riichi_exclusion`
- `imitation value_loss`
- `model_features.policy_tower/value_tower`

不足が出たら CQ 化して可観測性を先に上げる（推測で進めない）。

---

## 7. Trigger to Move Stage（次段へ進む条件）

「主系列を次段へ進める」と判定する目安（暫定）。

必須条件:

1. imitation 基準（`cycle0.eval_before`）を cycle 後半でも平均的に上回る
2. after 指標が `exp_050` / `exp_052` を一貫して更新する
3. `eval_before -> eval` が改善していても、長期着地が悪化しない
4. 追加再現実験でも同傾向が出る

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

**anchor / rule_mix / mixed_ppo まで進めた結果、「更新を安定化するだけ」では imitation 基準を長期的に超えられないことが見えてきた。現在の本丸は、rule を優秀な教師候補とみなしつつ、それを actor 改善に結びつける更新設計（target / off-policy 扱い / 補助損失）をどう作るかである。**
