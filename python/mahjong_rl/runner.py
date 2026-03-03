"""Stage 1 統合ランナー: config.yaml → self-play → learner → eval"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from mahjong_rl.experiment import ExperimentConfig, RunDirectory
from mahjong_rl.encoders import FlatFeatureEncoder, ChannelTensorEncoder
from mahjong_rl.models import MLPPolicyValueModel
from mahjong_rl.selfplay_worker import SelfPlayWorker
from mahjong_rl.learner import Learner
from mahjong_rl.evaluator import EvaluationRunner

logger = logging.getLogger(__name__)


class Stage1Runner:
    """Stage 1 実験の統合ランナー

    config.yaml を入力として以下のフェーズを順に実行する:
      1. run directory 初期化
      2. (optional) imitation warm start
      3. self-play データ生成
      4. learner による学習
      5. evaluator による評価

    各フェーズは experiment.phases 設定で有効/無効を制御できる。
    デフォルト: ["selfplay", "learner", "eval"]
    imitation 付き: ["imitation", "selfplay", "learner", "eval"]
    """

    def __init__(self, config: ExperimentConfig, base_dir: Path = Path("runs")):
        self._config = config
        self._base_dir = base_dir

    def _get_phases(self) -> list[str]:
        """実行フェーズのリストを取得する"""
        phases = self._config.experiment.get("phases", None)
        if phases is not None:
            return list(phases)
        return ["selfplay", "learner", "eval"]

    def run(self) -> dict:
        """実験を実行して結果を返す

        Returns:
            結果 dict: run_dir, phases, selfplay_stats, train_metrics, eval_metrics
        """
        result = {}
        phases = self._get_phases()
        result["phases"] = phases
        total_phases = len(phases) + 1  # +1 for init

        # 1. Run directory 初期化
        phase_num = 1
        logger.info(f"[Phase {phase_num}/{total_phases}] run directory 初期化")
        run_dir = RunDirectory(base_dir=self._base_dir).create(self._config)
        result["run_dir"] = str(run_dir)
        logger.info(f"  run_dir: {run_dir}")

        # モデル・エンコーダ生成
        encoder = self._create_encoder()
        model = self._create_model(encoder)
        obs_mode = self._config.experiment.get("observation_mode", "full")

        for phase in phases:
            phase_num += 1
            label = f"[Phase {phase_num}/{total_phases}]"

            if phase == "imitation":
                logger.info(f"{label} imitation warm start")
                try:
                    result["imitation_metrics"] = self._run_imitation(
                        run_dir, model, encoder)
                except Exception as e:
                    logger.error(f"  imitation フェーズで失敗: {e}")
                    result["error"] = f"imitation: {e}"
                    return result

            elif phase == "selfplay":
                logger.info(f"{label} self-play データ生成")
                try:
                    result["selfplay_stats"] = self._run_selfplay(
                        run_dir, model, encoder)
                except Exception as e:
                    logger.error(f"  self-play フェーズで失敗: {e}")
                    result["error"] = f"selfplay: {e}"
                    return result

            elif phase == "learner":
                logger.info(f"{label} learner 学習")
                try:
                    selfplay_dir = run_dir / "selfplay"
                    result["train_metrics"] = self._run_learner(
                        run_dir, selfplay_dir, model)
                except Exception as e:
                    logger.error(f"  learner フェーズで失敗: {e}")
                    result["error"] = f"learner: {e}"
                    return result

            elif phase == "eval":
                logger.info(f"{label} evaluator 評価")
                try:
                    result["eval_metrics"] = self._run_eval(
                        run_dir, model, encoder, obs_mode)
                except Exception as e:
                    logger.error(f"  evaluator フェーズで失敗: {e}")
                    result["error"] = f"evaluator: {e}"
                    return result

        logger.info("実験完了")
        return result

    def _run_imitation(self, run_dir: Path, model, encoder) -> dict:
        """imitation warm start フェーズ"""
        sp_cfg = self._config.selfplay
        imitation_dir = run_dir / "imitation"

        # baseline 教師データ生成
        imi_config = dict(self._as_dict())
        imi_sp = dict(sp_cfg)
        imi_sp["save_baseline_actions"] = True
        imi_sp["policy_ratio"] = 0.0  # 全席 baseline
        imi_config["selfplay"] = imi_sp

        worker = SelfPlayWorker(
            config=imi_config,
            model=model,
            encoder=encoder,
            output_dir=imitation_dir,
        )
        imi_matches = sp_cfg.get("imitation_matches",
                                 sp_cfg.get("num_matches", 10))
        sp_stats = worker.run(
            num_matches=imi_matches,
            seed_start=sp_cfg.get("imitation_seed_start",
                                  sp_cfg.get("seed_start", 0)),
        )
        logger.info(f"  imitation data: {sp_stats['total_steps']} steps")

        # imitation 学習
        imi_train_config = dict(self._as_dict())
        imi_train_config["training"] = dict(imi_train_config["training"])
        imi_train_config["training"]["algorithm"] = "imitation"

        learner = Learner(
            config=imi_train_config,
            model=model,
            run_dir=run_dir,
        )
        imi_epochs = self._config.training.get("imitation_epochs",
                                                self._config.training.get("epochs", 4))
        metrics = learner.train(
            imitation_dir,
            num_epochs=imi_epochs,
            filter_actor_type="baseline",
        )
        learner.save_checkpoint(tag="imitation")
        logger.info(f"  imitation loss: {metrics['policy_loss']:.4f}")
        return metrics

    def _run_selfplay(self, run_dir: Path, model, encoder) -> dict:
        """self-play フェーズ"""
        sp_cfg = self._config.selfplay
        selfplay_dir = run_dir / "selfplay"
        worker = SelfPlayWorker(
            config=self._as_dict(),
            model=model,
            encoder=encoder,
            output_dir=selfplay_dir,
        )
        sp_stats = worker.run(
            num_matches=sp_cfg.get("num_matches", 10),
            seed_start=sp_cfg.get("seed_start", 0),
        )
        logger.info(f"  total_steps: {sp_stats['total_steps']}")
        return sp_stats

    def _run_learner(self, run_dir: Path, shard_dir: Path, model) -> dict:
        """learner フェーズ"""
        learner = Learner(
            config=self._as_dict(),
            model=model,
            run_dir=run_dir,
        )
        train_metrics = learner.train(shard_dir)
        learner.save_checkpoint(tag="final")
        logger.info(f"  policy_loss: {train_metrics['policy_loss']:.4f}")
        return train_metrics

    def _run_eval(self, run_dir: Path, model, encoder, obs_mode: str) -> dict:
        """eval フェーズ"""
        eval_cfg = self._config.evaluation
        eval_dir = run_dir / "eval"
        eval_runner = EvaluationRunner(
            model=model,
            encoder=encoder,
            observation_mode=obs_mode,
        )
        eval_metrics = eval_runner.evaluate(
            num_matches=eval_cfg.get("num_matches", 10),
            seed_start=eval_cfg.get("seed_start", 0),
            eval_dir=eval_dir,
        )
        logger.info(f"  avg_rank: {eval_metrics.avg_rank:.2f}")
        return {
            "avg_rank": eval_metrics.avg_rank,
            "avg_score": eval_metrics.avg_score,
            "win_rate": eval_metrics.win_rate,
            "deal_in_rate": eval_metrics.deal_in_rate,
        }

    def _create_encoder(self):
        """設定からエンコーダを生成する"""
        enc_cfg = self._config.feature_encoder
        name = enc_cfg.get("name", "FlatFeatureEncoder")
        obs_mode = enc_cfg.get(
            "observation_mode",
            self._config.experiment.get("observation_mode", "full"),
        )
        if name == "ChannelTensorEncoder":
            return ChannelTensorEncoder(observation_mode=obs_mode)
        return FlatFeatureEncoder(observation_mode=obs_mode)

    def _create_model(self, encoder):
        """設定からモデルを生成する"""
        model_cfg = self._config.model
        hidden_dims = model_cfg.get("hidden_dims", [256, 128])
        value_heads = model_cfg.get("value_heads", ["round_delta"])

        # エンコーダの出力次元を取得
        meta = encoder.metadata()
        import math
        input_dim = math.prod(meta.output_shape)

        return MLPPolicyValueModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            value_heads=value_heads,
        )

    def _as_dict(self) -> dict:
        """ExperimentConfig を dict として返す"""
        return {
            "experiment": self._config.experiment,
            "feature_encoder": self._config.feature_encoder,
            "model": self._config.model,
            "reward": self._config.reward,
            "selfplay": self._config.selfplay,
            "training": self._config.training,
            "evaluation": self._config.evaluation,
            "export": self._config.export,
        }
