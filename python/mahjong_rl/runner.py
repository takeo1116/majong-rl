"""Stage 1 統合ランナー: config.yaml → self-play → learner → eval"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from mahjong_rl.experiment import ExperimentConfig, RunDirectory

VALID_DEVICES = {"cpu", "cuda", "auto"}


def resolve_device(requested: str) -> torch.device:
    """デバイス文字列を torch.device に解決する

    Args:
        requested: 'cpu', 'cuda', 'auto' のいずれか

    Returns:
        解決された torch.device

    Raises:
        RuntimeError: 'cuda' が要求されたが利用不可の場合
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda が要求されましたが利用できません")
    return torch.device(requested)
from mahjong_rl.encoders import FlatFeatureEncoder, ChannelTensorEncoder
from mahjong_rl.models import MLPPolicyValueModel
from mahjong_rl.selfplay_worker import SelfPlayWorker
from mahjong_rl.learner import Learner
from mahjong_rl.evaluator import EvaluationRunner, compute_eval_diff

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
        self._global_seed: int | None = None

    def _get_phases(self) -> list[str]:
        """実行フェーズのリストを取得する"""
        phases = self._config.experiment.get("phases", None)
        if phases is not None:
            return list(phases)
        return ["selfplay", "learner", "eval"]

    def validate_config(self) -> list[str]:
        """config のバリデーションを行い、エラーメッセージのリストを返す

        空リスト = バリデーション成功
        """
        errors: list[str] = []
        cfg = self._config

        # phases の値チェック
        valid_phases = {"imitation", "selfplay", "learner", "eval"}
        phases = self._get_phases()
        for p in phases:
            if p not in valid_phases:
                errors.append(
                    f"不正なフェーズ '{p}' (有効値: {sorted(valid_phases)})")

        # imitation が phases にあるのに selfplay がない場合の警告
        if "imitation" in phases and "selfplay" not in phases:
            errors.append(
                "imitation フェーズがあるのに selfplay フェーズがありません")

        # eval mode チェック
        eval_mode = cfg.evaluation.get("mode", "single")
        if eval_mode not in ("single", "rotation"):
            errors.append(
                f"不正な evaluation.mode '{eval_mode}' (有効値: single, rotation)")

        # observation mode チェック
        obs_mode = cfg.experiment.get("observation_mode", "full")
        if obs_mode not in ("full", "partial"):
            errors.append(
                f"不正な observation_mode '{obs_mode}' (有効値: full, partial)")

        # encoder 名チェック
        enc_name = cfg.feature_encoder.get("name", "FlatFeatureEncoder")
        valid_encoders = {"FlatFeatureEncoder", "ChannelTensorEncoder"}
        if enc_name not in valid_encoders:
            errors.append(
                f"不正な encoder '{enc_name}' (有効値: {sorted(valid_encoders)})")

        # model 名チェック
        model_name = cfg.model.get("name", "MLPPolicyValueModel")
        valid_models = {"MLPPolicyValueModel"}
        if model_name not in valid_models:
            errors.append(
                f"不正な model '{model_name}' (有効値: {sorted(valid_models)})")

        # global_seed 型・値域チェック
        seed = cfg.experiment.get("global_seed", None)
        if seed is not None:
            if not isinstance(seed, (int, float)):
                errors.append(
                    f"global_seed は整数で指定してください (型: {type(seed).__name__})")
            elif isinstance(seed, float) and seed != int(seed):
                errors.append(
                    f"global_seed は整数で指定してください (値: {seed})")
            elif not (0 <= int(seed) <= 2**32 - 1):
                errors.append(
                    f"global_seed は 0 〜 {2**32 - 1} の範囲で指定してください (値: {int(seed)})")

        # seed_start 値域チェック
        for key_path, label in [
            (("selfplay", "seed_start"), "selfplay.seed_start"),
            (("evaluation", "seed_start"), "evaluation.seed_start"),
        ]:
            section = getattr(cfg, key_path[0], {})
            sv = section.get(key_path[1], None)
            if sv is not None:
                if isinstance(sv, (int, float)):
                    if sv < 0:
                        errors.append(
                            f"{label} は 0 以上で指定してください (値: {sv})")

        # デバイス設定チェック
        for device_key, label in [
            (("training", "device"), "training.device"),
            (("selfplay", "inference_device"), "selfplay.inference_device"),
            (("evaluation", "inference_device"), "evaluation.inference_device"),
        ]:
            section = getattr(cfg, device_key[0], {})
            dv = section.get(device_key[1], "auto")
            if dv not in VALID_DEVICES:
                errors.append(
                    f"不正な {label} '{dv}' (有効値: {sorted(VALID_DEVICES)})")

        return errors

    def run(self) -> dict:
        """実験を実行して結果を返す

        Returns:
            結果 dict: run_dir, phases, selfplay_stats, train_metrics, eval_metrics
        """
        # バリデーション
        errors = self.validate_config()
        if errors:
            raise ValueError(
                "config バリデーションエラー:\n" + "\n".join(f"  - {e}" for e in errors))

        result = {}
        phases = self._get_phases()
        result["phases"] = phases
        total_phases = len(phases) + 1  # +1 for init
        phase_status: dict[str, str] = {}

        # Global seed 固定
        self._global_seed = self._setup_global_seed()
        result["global_seed"] = self._global_seed

        # 1. Run directory 初期化
        phase_num = 1
        logger.info(f"[Phase {phase_num}/{total_phases}] run directory 初期化")
        run_dir = RunDirectory(base_dir=self._base_dir).create(self._config)
        result["run_dir"] = str(run_dir)
        logger.info(f"  run_dir: {run_dir}")

        # run.log 用 FileHandler 追加
        file_handler = self._setup_file_logging(run_dir)

        # モデル・エンコーダ生成
        encoder = self._create_encoder()
        model = self._create_model(encoder)
        obs_mode = self._config.experiment.get("observation_mode", "full")

        # デバイス解決と記録
        result["resolved_devices"] = self._resolve_all_devices()
        logger.info(f"  devices: {result['resolved_devices']}")

        for phase in phases:
            phase_num += 1
            label = f"[Phase {phase_num}/{total_phases}]"

            if phase == "imitation":
                logger.info(f"{label} imitation warm start")
                try:
                    result["imitation_metrics"] = self._run_imitation(
                        run_dir, model, encoder)
                    phase_status["imitation"] = "success"
                except Exception as e:
                    logger.error(f"  imitation フェーズで失敗: {e}")
                    result["error"] = f"imitation: {e}"
                    phase_status["imitation"] = "failed"
                    self._finalize(run_dir, result, phase_status, file_handler)
                    return result

            elif phase == "selfplay":
                logger.info(f"{label} self-play データ生成")
                try:
                    result["selfplay_stats"] = self._run_selfplay(
                        run_dir, model, encoder)
                    phase_status["selfplay"] = "success"
                except Exception as e:
                    logger.error(f"  self-play フェーズで失敗: {e}")
                    result["error"] = f"selfplay: {e}"
                    phase_status["selfplay"] = "failed"
                    self._finalize(run_dir, result, phase_status, file_handler)
                    return result

            elif phase == "learner":
                # 学習前評価 (eval も phases に含まれる場合のみ)
                if "eval" in phases:
                    try:
                        logger.info(f"{label} 学習前評価 (eval_before)")
                        eval_before_dir = run_dir / "eval_before"
                        result["eval_before"] = self._run_eval(
                            run_dir, model, encoder, obs_mode,
                            eval_dir_override=eval_before_dir)
                        logger.info(f"  eval_before avg_rank: {result['eval_before'].get('avg_rank', '?')}")
                    except Exception as e:
                        logger.warning(f"  学習前評価をスキップ: {e}")

                logger.info(f"{label} learner 学習")
                try:
                    selfplay_dir = run_dir / "selfplay"
                    result["train_metrics"] = self._run_learner(
                        run_dir, selfplay_dir, model)
                    phase_status["learner"] = "success"
                except Exception as e:
                    logger.error(f"  learner フェーズで失敗: {e}")
                    result["error"] = f"learner: {e}"
                    phase_status["learner"] = "failed"
                    self._finalize(run_dir, result, phase_status, file_handler)
                    return result

            elif phase == "eval":
                logger.info(f"{label} evaluator 評価")
                try:
                    result["eval_metrics"] = self._run_eval(
                        run_dir, model, encoder, obs_mode)
                    phase_status["eval"] = "success"

                    # 学習前後差分レポート生成
                    if "eval_before" in result:
                        diff = compute_eval_diff(
                            result["eval_before"], result["eval_metrics"])
                        result["eval_diff"] = diff
                        diff_path = run_dir / "eval" / "eval_diff.json"
                        diff_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(diff_path, "w") as f:
                            json.dump(diff, f, indent=2, ensure_ascii=False)
                        logger.info(f"  eval_diff: avg_rank {diff['avg_rank']['delta']:+.3f}")
                except Exception as e:
                    logger.error(f"  evaluator フェーズで失敗: {e}")
                    result["error"] = f"evaluator: {e}"
                    phase_status["eval"] = "failed"
                    self._finalize(run_dir, result, phase_status, file_handler)
                    return result

        logger.info("実験完了")
        self._finalize(run_dir, result, phase_status, file_handler)
        return result

    def _finalize(self, run_dir: Path, result: dict,
                  phase_status: dict[str, str],
                  file_handler: logging.FileHandler) -> None:
        """run 終了時の共通処理: summary 保存・notes 追記・ログ後始末"""
        self._save_summary(run_dir, result, phase_status)
        self._append_notes(run_dir, result, phase_status)
        self._teardown_file_logging(file_handler)

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

        sp_device = resolve_device(
            self._config.selfplay.get("inference_device", "auto"))
        worker = SelfPlayWorker(
            config=imi_config,
            model=model,
            encoder=encoder,
            output_dir=imitation_dir,
            inference_device=sp_device,
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

        training_device = resolve_device(
            self._config.training.get("device", "auto"))
        learner = Learner(
            config=imi_train_config,
            model=model,
            run_dir=run_dir,
            device=training_device,
        )
        imi_epochs = self._config.training.get("imitation_epochs",
                                                self._config.training.get("epochs", 4))
        imi_filter = self._config.training.get("imitation_filter", None)
        metrics = learner.train(
            imitation_dir,
            num_epochs=imi_epochs,
            filter_actor_type="baseline",
            imitation_filter=imi_filter,
        )
        learner.save_checkpoint(tag="imitation")
        logger.info(f"  imitation loss: {metrics['policy_loss']:.4f}")
        return metrics

    def _run_selfplay(self, run_dir: Path, model, encoder) -> dict:
        """self-play フェーズ"""
        sp_cfg = self._config.selfplay
        selfplay_dir = run_dir / "selfplay"
        sp_device = resolve_device(
            sp_cfg.get("inference_device", "auto"))
        worker = SelfPlayWorker(
            config=self._as_dict(),
            model=model,
            encoder=encoder,
            output_dir=selfplay_dir,
            inference_device=sp_device,
        )
        sp_stats = worker.run(
            num_matches=sp_cfg.get("num_matches", 10),
            seed_start=sp_cfg.get("seed_start", 0),
        )
        logger.info(f"  total_steps: {sp_stats['total_steps']}")
        return sp_stats

    def _run_learner(self, run_dir: Path, shard_dir: Path, model) -> dict:
        """learner フェーズ"""
        training_device = resolve_device(
            self._config.training.get("device", "auto"))
        learner = Learner(
            config=self._as_dict(),
            model=model,
            run_dir=run_dir,
            device=training_device,
        )
        train_metrics = learner.train(shard_dir)
        learner.save_checkpoint(tag="final")
        logger.info(f"  policy_loss: {train_metrics['policy_loss']:.4f}")
        return train_metrics

    def _run_eval(self, run_dir: Path, model, encoder, obs_mode: str,
                  eval_dir_override: Path | None = None) -> dict:
        """eval フェーズ

        evaluation.mode で単席 / rotation を切り替え可能。
        - "single" (デフォルト): 単席評価
        - "rotation": 全席ローテーション評価
        """
        eval_cfg = self._config.evaluation
        eval_dir = eval_dir_override or (run_dir / "eval")
        eval_device = resolve_device(
            eval_cfg.get("inference_device", "auto"))
        eval_runner = EvaluationRunner(
            model=model,
            encoder=encoder,
            observation_mode=obs_mode,
            inference_device=eval_device,
        )
        eval_mode = eval_cfg.get("mode", "single")
        num_matches = eval_cfg.get("num_matches", 10)
        seed_start = eval_cfg.get("seed_start", 0)

        if eval_mode == "rotation":
            seats = eval_cfg.get("rotation_seats", [0, 1, 2, 3])
            rotation_result = eval_runner.evaluate_rotation(
                num_matches=num_matches,
                seed_start=seed_start,
                eval_dir=eval_dir,
                seats=seats,
            )
            agg = rotation_result.aggregate
            logger.info(f"  avg_rank (rotation): {agg.avg_rank:.2f}")
            return {
                "eval_mode": "rotation",
                "rotation_seats": seats,
                "avg_rank": agg.avg_rank,
                "avg_score": agg.avg_score,
                "win_rate": agg.win_rate,
                "deal_in_rate": agg.deal_in_rate,
            }
        else:
            policy_seats = eval_cfg.get("policy_seats", None)
            eval_metrics = eval_runner.evaluate(
                num_matches=num_matches,
                seed_start=seed_start,
                eval_dir=eval_dir,
                policy_seats=policy_seats,
            )
            logger.info(f"  avg_rank: {eval_metrics.avg_rank:.2f}")
            return {
                "eval_mode": "single",
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

    def _setup_file_logging(self, run_dir: Path) -> logging.FileHandler:
        """run.log 用の FileHandler を設定する"""
        handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        # mahjong_rl ロガーにアタッチ（ルートロガーのレベルに依存しない）
        ml_logger = logging.getLogger("mahjong_rl")
        ml_logger.addHandler(handler)
        if ml_logger.getEffectiveLevel() > logging.DEBUG:
            ml_logger.setLevel(logging.DEBUG)
        return handler

    def _teardown_file_logging(self, handler: logging.FileHandler) -> None:
        """FileHandler を除去する"""
        handler.flush()
        handler.close()
        logging.getLogger("mahjong_rl").removeHandler(handler)

    def _save_summary(self, run_dir: Path, result: dict,
                      phase_status: dict[str, str]) -> None:
        """summary.json を保存する"""
        # shard 数カウント
        selfplay_dir = run_dir / "selfplay"
        shard_count = len(list(selfplay_dir.glob("shard_*.parquet"))) if selfplay_dir.exists() else 0

        # checkpoint 有無
        ckpt_dir = run_dir / "checkpoints"
        has_checkpoint = any(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else False

        # eval 有無
        has_eval = "eval_metrics" in result

        # フェーズ別統計 (CQ-0053)
        phase_stats = {}
        if "selfplay_stats" in result:
            sp = result["selfplay_stats"]
            phase_stats["selfplay"] = {
                "total_steps": sp.get("total_steps", 0),
                "total_matches": sp.get("total_matches", 0),
                "shard_count": shard_count,
            }
        if "imitation_metrics" in result:
            imi = result["imitation_metrics"]
            # imitation shard 数
            imi_dir = run_dir / "imitation"
            imi_shard_count = len(list(imi_dir.glob("shard_*.parquet"))) if imi_dir.exists() else 0
            phase_stats["imitation"] = {
                "total_steps": imi.get("total_steps", 0),
                "num_updates": imi.get("num_updates", 0),
                "shard_count": imi_shard_count,
                "policy_loss": imi.get("policy_loss"),
            }
        if "train_metrics" in result:
            tm = result["train_metrics"]
            phase_stats["learner"] = {
                "total_steps": tm.get("total_steps", 0),
                "num_updates": tm.get("num_updates", 0),
                "policy_loss": tm.get("policy_loss"),
                "value_loss": tm.get("value_loss"),
                "mode": tm.get("mode"),
            }
        if "eval_metrics" in result:
            em = result["eval_metrics"]
            phase_stats["eval"] = {
                "eval_mode": em.get("eval_mode"),
                "avg_rank": em.get("avg_rank"),
                "avg_score": em.get("avg_score"),
                "win_rate": em.get("win_rate"),
                "deal_in_rate": em.get("deal_in_rate"),
            }

        # actor_type 内訳 (shard から集計)
        actor_type_counts = self._count_actor_types(run_dir)

        # device 情報
        resolved = result.get("resolved_devices", {})
        device_info = {
            "training": {
                "requested": self._config.training.get("device", "auto"),
                "resolved": resolved.get("training", "cpu"),
            },
            "selfplay": {
                "requested": self._config.selfplay.get("inference_device", "auto"),
                "resolved": resolved.get("selfplay", "cpu"),
            },
            "evaluation": {
                "requested": self._config.evaluation.get("inference_device", "auto"),
                "resolved": resolved.get("evaluation", "cpu"),
            },
        }

        summary = {
            "global_seed": result.get("global_seed"),
            "phases": result.get("phases", []),
            "phase_status": phase_status,
            "success": "error" not in result,
            "error": result.get("error"),
            "shard_count": shard_count,
            "has_checkpoint": has_checkpoint,
            "has_eval": has_eval,
            "phase_stats": phase_stats,
            "actor_type_counts": actor_type_counts,
            "device_info": device_info,
            "env_info": self._collect_env_info(),
        }

        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _count_actor_types(self, run_dir: Path) -> dict[str, int]:
        """shard ファイルから actor_type ごとの件数を集計する"""
        from mahjong_rl.shard import ShardReader
        counts: dict[str, int] = {}
        for subdir_name in ["selfplay", "imitation"]:
            subdir = run_dir / subdir_name
            if not subdir.exists() or not list(subdir.glob("shard_*.parquet")):
                continue
            try:
                reader = ShardReader(subdir)
                tensors = reader.read_as_tensors()
                for at in tensors.get("actor_types", []):
                    counts[at] = counts.get(at, 0) + 1
            except Exception:
                pass
        return counts

    def _append_notes(self, run_dir: Path, result: dict,
                      phase_status: dict[str, str]) -> None:
        """notes.md に実行結果の概要を追記する"""
        notes_path = run_dir / "notes.md"
        lines = [
            "",
            "## 実行結果",
            f"- 状態: {'成功' if 'error' not in result else '失敗'}",
            f"- global_seed: {result.get('global_seed')}",
            f"- phases: {result.get('phases', [])}",
        ]
        # フェーズ別ステータス
        for phase, status in phase_status.items():
            lines.append(f"  - {phase}: {status}")

        # デバイス情報
        resolved = result.get("resolved_devices", {})
        if resolved:
            lines.append(f"- devices: training={resolved.get('training', '?')}, "
                         f"selfplay={resolved.get('selfplay', '?')}, "
                         f"eval={resolved.get('evaluation', '?')}")

        # エラー情報
        if "error" in result:
            lines.append(f"- エラー: {result['error']}")

        # 主要指標
        if "eval_metrics" in result:
            em = result["eval_metrics"]
            lines.append("")
            lines.append("## 主要指標")
            lines.append(f"- avg_rank: {em.get('avg_rank', '?'):.2f}"
                         if isinstance(em.get('avg_rank'), (int, float))
                         else f"- avg_rank: {em.get('avg_rank', '?')}")
            lines.append(f"- avg_score: {em.get('avg_score', '?'):.1f}"
                         if isinstance(em.get('avg_score'), (int, float))
                         else f"- avg_score: {em.get('avg_score', '?')}")
            lines.append(f"- win_rate: {em.get('win_rate', '?'):.3f}"
                         if isinstance(em.get('win_rate'), (int, float))
                         else f"- win_rate: {em.get('win_rate', '?')}")
            lines.append(f"- deal_in_rate: {em.get('deal_in_rate', '?'):.3f}"
                         if isinstance(em.get('deal_in_rate'), (int, float))
                         else f"- deal_in_rate: {em.get('deal_in_rate', '?')}")

        # checkpoint
        ckpt_dir = run_dir / "checkpoints"
        has_ckpt = any(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else False
        lines.append(f"- checkpoint: {'あり' if has_ckpt else 'なし'}")
        lines.append("")

        with open(notes_path, "a") as f:
            f.write("\n".join(lines))

    def _setup_global_seed(self) -> int:
        """global seed を設定する

        experiment.global_seed が指定されていればそれを使い、
        未指定なら乱数で生成する。
        """
        seed = self._config.experiment.get("global_seed", None)
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info(f"  global_seed: {seed}")
        return seed

    def _resolve_all_devices(self) -> dict:
        """全フェーズのデバイスを解決し、解決結果を返す"""
        cfg = self._config
        training_dev = resolve_device(cfg.training.get("device", "auto"))
        sp_dev = resolve_device(cfg.selfplay.get("inference_device", "auto"))
        eval_dev = resolve_device(cfg.evaluation.get("inference_device", "auto"))
        return {
            "training": str(training_dev),
            "selfplay": str(sp_dev),
            "evaluation": str(eval_dev),
        }

    @staticmethod
    def _collect_env_info() -> dict:
        """PyTorch/CUDA 環境情報を収集する"""
        info: dict = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
        return info

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
