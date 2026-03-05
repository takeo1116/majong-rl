"""Stage 1 Self-Play Worker: 対局生成と shard 書き出し"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from mahjong_rl.env import Stage1Env
from mahjong_rl.encoders import FlatFeatureEncoder, ChannelTensorEncoder
from mahjong_rl.models import MLPPolicyValueModel
from mahjong_rl.action_selector import ActionSelector, SelectionMode
from mahjong_rl.baseline import RuleBasedBaseline
from mahjong_rl.profiler import Profiler
from mahjong_rl.shard import LearningSample, ShardWriter


class SelfPlayWorker:
    """Stage 1 用 self-play worker

    学習ポリシーとベースラインの混合対戦で対局を生成し、
    学習ポリシー席のステップを shard file に書き出す。
    """

    def __init__(
        self,
        config: dict,
        model: torch.nn.Module,
        encoder,
        output_dir: Path,
        worker_id: str = "worker_0",
        inference_device: torch.device | None = None,
        profiler: Profiler | None = None,
    ):
        self._config = config
        self._profiler = profiler
        self._device = inference_device or torch.device("cpu")
        self._model = model.to(self._device)
        self._encoder = encoder
        self._output_dir = Path(output_dir)
        self._worker_id = worker_id

        self._baseline = RuleBasedBaseline()

        # self-play 設定
        sp = config.get("selfplay", {})
        self._policy_ratio = sp.get("policy_ratio", 0.5)
        self._temperature = sp.get("temperature", 1.0)
        self._save_baseline_actions = sp.get("save_baseline_actions", False)
        max_per_shard = sp.get("max_samples_per_shard", 10000)

        self._selector = ActionSelector(
            mode=SelectionMode.SAMPLE,
            temperature=self._temperature,
        )
        self._writer = ShardWriter(self._output_dir, max_samples=max_per_shard)

        obs_mode = config.get("experiment", {}).get("observation_mode", "full")
        self._observation_mode = obs_mode

        # メタデータ
        self._experiment_id = config.get("experiment", {}).get("name", "")
        self._model_version = 0
        self._generation = 0

    def run(self, num_matches: int, seed_start: int = 0,
            match_seeds: list[int] | None = None) -> dict:
        """指定数の半荘を生成し、shard に書き出す

        Args:
            num_matches: 半荘数
            seed_start: シード開始値 (match_seeds 未指定時に使用)
            match_seeds: 各 match の seed リスト (指定時は seed_start を無視)

        Returns:
            統計情報 dict
        """
        if match_seeds is not None and len(match_seeds) != num_matches:
            raise ValueError(
                f"match_seeds の長さ ({len(match_seeds)}) と "
                f"num_matches ({num_matches}) が一致しません")

        total_steps = 0
        total_rounds = 0
        run_id = uuid.uuid4().hex[:8]
        profiler = self._profiler or Profiler(enabled=False)
        self._round_results: list[dict] = []

        profiler.start("selfplay_match_loop")
        for match_idx in range(num_matches):
            seed = match_seeds[match_idx] if match_seeds is not None else seed_start + match_idx
            episode_id = f"ep_{seed}"

            stats = self._play_one_match(
                seed=seed,
                episode_id=episode_id,
                run_id=run_id,
            )
            total_steps += stats["steps"]
            total_rounds += stats["rounds"]
        profiler.stop("selfplay_match_loop")

        profiler.start("selfplay_shard_write")
        self._writer.close()
        profiler.stop("selfplay_shard_write")

        # round_results.jsonl 出力 (CQ-0105)
        self._write_round_results()

        # 局結果集計 (CQ-0106)
        round_stats = self._compute_round_stats()

        result = {
            "num_matches": num_matches,
            "total_steps": total_steps,
            "total_rounds": total_rounds,
            "output_dir": str(self._output_dir),
            "inference_device": str(self._device),
        }
        result.update(round_stats)
        return result

    def _play_one_match(self, seed: int, episode_id: str, run_id: str) -> dict:
        """1 半荘を実行しサンプルを収集する"""
        env = Stage1Env(observation_mode=self._observation_mode)
        torch.manual_seed(seed)
        obs, info = env.reset(seed=seed)

        # 4 席を policy/baseline に割り当て
        seat_is_policy = self._assign_seats(seed)

        steps = 0
        sample_step = 0  # ポリシーサンプルの意思決定順序カウンタ
        round_count = 0
        prev_round_number = info["round_number"]

        max_steps = 10000
        while steps < max_steps:
            current = env.current_player
            mask = env.get_legal_mask()

            if seat_is_policy[current]:
                # ポリシー席: action 選択前の観測を保存用にエンコード
                pre_features = self._encoder.encode(obs)
                pre_features_flat = pre_features.flatten() if pre_features.ndim > 1 else pre_features
                tile_type, log_prob, value = self._policy_step(obs, mask)
            else:
                # ベースライン席: ルールベースで選択
                tile_type = self._baseline_step(env, mask)
                log_prob = 0.0
                value = 0.0
                # baseline 保存が有効なら観測をエンコード
                if self._save_baseline_actions:
                    pre_features = self._encoder.encode(obs)
                    pre_features_flat = pre_features.flatten() if pre_features.ndim > 1 else pre_features
                else:
                    pre_features_flat = None

            obs, rewards, terminated, truncated, info = env.step(tile_type)

            # 局境界判定
            round_number = info["round_number"]
            round_over = (round_number != prev_round_number) or terminated
            if round_over:
                round_count += 1
            prev_round_number = round_number

            # 局終了イベント記録 (CQ-0105, CQ-0108)
            for evt in info.get("round_end_events", []):
                wps = evt["winner_players"]
                lp = evt["loser_player"]
                policy_wps = [w for w in wps if seat_is_policy[w]]
                self._round_results.append({
                    "event_type": evt["event_type"],
                    "winner_players": wps,
                    "loser_player": lp,
                    "is_policy_win": len(policy_wps) > 0,
                    "is_policy_deal_in": (lp >= 0 and seat_is_policy[lp]),
                    "is_draw": evt["event_type"] == "ryukyoku",
                    "policy_winner_players": policy_wps,
                    "round_id": evt["round_id"],
                    "episode_id": episode_id,
                    "worker_id": self._worker_id,
                    "seed": seed,
                })

            # サンプル収集
            should_save = seat_is_policy[current] or (
                self._save_baseline_actions and not seat_is_policy[current])
            if should_save:
                actor_type = "policy" if seat_is_policy[current] else "baseline"
                sample = LearningSample(
                    observation=pre_features_flat,
                    legal_mask=mask.astype(np.float32),
                    action=tile_type,
                    reward=float(rewards[current]),
                    log_prob=float(log_prob),
                    value=float(value),
                    terminated=terminated,
                    round_over=round_over,
                    experiment_id=self._experiment_id,
                    run_id=run_id,
                    worker_id=self._worker_id,
                    shard_id="",
                    model_version=self._model_version,
                    generation=self._generation,
                    timestamp=time.time(),
                    episode_id=episode_id,
                    round_id=round_number,
                    step_id=sample_step,
                    player_id=current,
                    actor_type=actor_type,
                )
                self._writer.add(sample)
                sample_step += 1

            steps += 1
            if terminated:
                break

        return {"steps": steps, "rounds": round_count}

    def _assign_seats(self, seed: int) -> list[bool]:
        """4 席の policy/baseline 割り当てを決める"""
        rng = np.random.RandomState(seed)
        return [rng.random() < self._policy_ratio for _ in range(4)]

    def _policy_step(self, obs, mask: np.ndarray) -> tuple[int, float, float]:
        """ポリシーモデルで打牌を選択する"""
        features = self._encoder.encode(obs)
        features_flat = features.flatten() if features.ndim > 1 else features
        features_t = torch.from_numpy(features_flat).unsqueeze(0).to(self._device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model(features_t, mask_t)

        tile_type, log_prob = self._selector.select(output.logits[0], mask_t[0])

        # value head の最初の値を使用
        value = 0.0
        for v_tensor in output.values.values():
            value = v_tensor.item()
            break

        return tile_type, float(log_prob.item()), value

    def _baseline_step(self, env: Stage1Env, mask: np.ndarray) -> int:
        """ベースラインで打牌を選択する"""
        hand = env.env_state.round_state.players[env.current_player].hand
        return self._baseline.select_discard(list(hand), mask)

    def _write_round_results(self) -> None:
        """round_results.jsonl を出力する (CQ-0105)"""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / "round_results.jsonl"
        with open(path, "w") as f:
            for r in self._round_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _compute_round_stats(self) -> dict:
        """局結果から集計統計を計算する (CQ-0106, CQ-0108)

        集計定義:
        - num_rounds/tsumo_count/ron_count/ryukyoku_count: 全体基準（全局カウント）
        - policy_wins: winner_players 内の policy 席の人数合計（multi-ron で複数加算）
        - policy_win_by_tsumo/policy_win_by_ron: 同上（和了種別ごと）
        - policy_deal_ins/policy_draws: policy 席基準（局単位）
        """
        tsumo_count = 0
        ron_count = 0
        ryukyoku_count = 0
        policy_wins = 0
        policy_deal_ins = 0
        policy_draws = 0
        policy_win_by_tsumo = 0
        policy_win_by_ron = 0

        for r in self._round_results:
            et = r["event_type"]
            if et == "tsumo":
                tsumo_count += 1
            elif et == "ron":
                ron_count += 1
            else:
                ryukyoku_count += 1

            # winner_players から policy 席の勝者数を直接カウント（multi-ron 対応）
            wps = r["winner_players"]
            policy_win_count = sum(
                1 for w in wps if self._is_policy_seat(r, w))
            policy_wins += policy_win_count
            if policy_win_count > 0:
                if et == "tsumo":
                    policy_win_by_tsumo += policy_win_count
                elif et == "ron":
                    policy_win_by_ron += policy_win_count

            if r["is_policy_deal_in"]:
                policy_deal_ins += 1
            if r["is_draw"]:
                policy_draws += 1

        return {
            "num_rounds": len(self._round_results),
            "tsumo_count": tsumo_count,
            "ron_count": ron_count,
            "ryukyoku_count": ryukyoku_count,
            "policy_wins": policy_wins,
            "policy_deal_ins": policy_deal_ins,
            "policy_draws": policy_draws,
            "policy_win_by_tsumo": policy_win_by_tsumo,
            "policy_win_by_ron": policy_win_by_ron,
        }

    @staticmethod
    def _is_policy_seat(round_result: dict, player_id: int) -> bool:
        """round_result の is_policy_win は配列内に1人でもいれば True だが、
        個別 player が policy 席かは winner_players と is_policy_win から判断できない。
        そのため _play_one_match で seat_is_policy 情報を round_result に保持する。
        """
        policy_winners = round_result.get("policy_winner_players", [])
        return player_id in policy_winners
