"""Stage 1 評価対戦ランナー: 主要指標の集計"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mahjong_rl.env import Stage1Env
from mahjong_rl.action_selector import ActionSelector, SelectionMode
from mahjong_rl.baseline import RuleBasedBaseline
from mahjong_rl import RoundEndReason


@dataclass
class EvalMetrics:
    """評価指標"""
    avg_rank: float
    avg_score: float
    win_rate: float
    deal_in_rate: float
    num_matches: int
    num_rounds: int

    def save(self, path: Path) -> None:
        """JSON に保存"""
        data = {
            "avg_rank": self.avg_rank,
            "avg_score": self.avg_score,
            "win_rate": self.win_rate,
            "deal_in_rate": self.deal_in_rate,
            "num_matches": self.num_matches,
            "num_rounds": self.num_rounds,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class EvaluationRunner:
    """評価対戦ランナー

    プレイヤー0=学習ポリシー（argmax）、プレイヤー1-3=ベースライン。
    """

    def __init__(self, model: torch.nn.Module, encoder, observation_mode: str = "full"):
        self._model = model
        self._encoder = encoder
        self._observation_mode = observation_mode
        self._baseline = RuleBasedBaseline()
        self._selector = ActionSelector(mode=SelectionMode.ARGMAX)

    def evaluate(
        self,
        num_matches: int = 100,
        seed_start: int = 0,
        eval_dir: Path | None = None,
    ) -> EvalMetrics:
        """評価対戦を実行して指標を返す"""
        all_ranks = []
        all_scores = []
        total_wins = 0
        total_deal_ins = 0
        total_rounds = 0

        for i in range(num_matches):
            seed = seed_start + i
            result = self._play_one_match(seed)
            all_ranks.append(result["rank"])
            all_scores.append(result["score"])
            total_wins += result["wins"]
            total_deal_ins += result["deal_ins"]
            total_rounds += result["rounds"]

        metrics = EvalMetrics(
            avg_rank=float(np.mean(all_ranks)),
            avg_score=float(np.mean(all_scores)),
            win_rate=total_wins / max(total_rounds, 1),
            deal_in_rate=total_deal_ins / max(total_rounds, 1),
            num_matches=num_matches,
            num_rounds=total_rounds,
        )

        if eval_dir:
            metrics.save(eval_dir / "eval_metrics.json")

        return metrics

    def _play_one_match(self, seed: int) -> dict:
        """1 半荘を実行して結果を返す"""
        env = Stage1Env(observation_mode=self._observation_mode)
        obs, info = env.reset(seed=seed)

        # プレイヤー 0 がポリシー
        policy_player = 0
        wins = 0
        deal_ins = 0
        round_count = 0
        prev_round_number = info["round_number"]
        prev_scores = list(info["scores"])

        max_steps = 10000
        for _ in range(max_steps):
            current = env.current_player
            mask = env.get_legal_mask()

            if current == policy_player:
                tile_type = self._policy_step(obs, mask)
            else:
                tile_type = self._baseline_step(env, mask)

            obs, rewards, terminated, truncated, info = env.step(tile_type)

            # 局終了判定
            round_number = info["round_number"]
            round_over = (round_number != prev_round_number) or terminated
            if round_over:
                round_count += 1
                scores = info["scores"]

                # 和了判定: ポリシープレイヤーのスコアが増加
                score_diff = scores[policy_player] - prev_scores[policy_player]
                if score_diff > 0:
                    wins += 1

                # 放銃判定: ポリシープレイヤーのスコアが減少（他家和了による）
                if score_diff < 0:
                    # 他家のスコアが大きく増加していれば放銃の可能性
                    for p in range(4):
                        if p != policy_player and (scores[p] - prev_scores[p]) > 0:
                            deal_ins += 1
                            break

                prev_scores = list(scores)
                prev_round_number = round_number

            if terminated:
                break

        # 最終順位計算
        final_scores = info["scores"]
        rank = self._compute_rank(final_scores, policy_player)

        return {
            "rank": rank,
            "score": final_scores[policy_player] - 25000,
            "wins": wins,
            "deal_ins": deal_ins,
            "rounds": round_count,
        }

    def _policy_step(self, obs, mask: np.ndarray) -> int:
        """ポリシーモデルで打牌を選択する（argmax）"""
        features = self._encoder.encode(obs)
        features_flat = features.flatten() if features.ndim > 1 else features
        features_t = torch.from_numpy(features_flat).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        with torch.no_grad():
            output = self._model(features_t, mask_t)

        tile_type, _ = self._selector.select(output.logits[0], mask_t[0])
        return tile_type

    def _baseline_step(self, env: Stage1Env, mask: np.ndarray) -> int:
        """ベースラインで打牌を選択する"""
        hand = env.env_state.round_state.players[env.current_player].hand
        return self._baseline.select_discard(list(hand), mask)

    @staticmethod
    def _compute_rank(scores: list, player: int) -> int:
        """スコアから順位を計算する (1-indexed)"""
        player_score = scores[player]
        rank = 1
        for i, s in enumerate(scores):
            if i != player and s > player_score:
                rank += 1
        return rank
