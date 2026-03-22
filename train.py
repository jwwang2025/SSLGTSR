from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from sslgtsr.data.dataset import SocialRecDataset
from sslgtsr.data.sampling import BPRSampler
from sslgtsr.models.sslgtsr import SSLGTSR
from sslgtsr.training.trainer import Trainer, build_graphs_from_split
from sslgtsr.utils.config import load_yaml, make_run_paths
from sslgtsr.utils.logging import write_json
from sslgtsr.utils.seed import seed_everything
from sslgtsr.models.topo_pe import TopoPEConfig
from sslgtsr.models.cross_view_ssl import CrossViewSSLConfig


def pick_device(device_str: str) -> torch.device:
    if device_str.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--run_name", type=str, default="latest")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg["seed"]))

    device = pick_device(cfg.get("device", "cpu"))

    ds = SocialRecDataset(
        data_dir=args.data_dir,
        interactions_file=cfg["data"]["interactions_file"],
        social_file=cfg["data"]["social_file"],
        is_social_directed=cfg["data"]["is_social_directed"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        min_user_interactions=cfg["data"]["min_user_interactions"],
        seed=int(cfg["seed"]),
    )
    split = ds.load()

    ui_norm, uu_norm = build_graphs_from_split(
        num_users=split.num_users,
        num_items=split.num_items,
        train_ui=split.train_ui,
        uu=split.uu,
        directed=cfg["data"]["is_social_directed"],
    )

    model = SSLGTSR(
        num_users=split.num_users,
        num_items=split.num_items,
        emb_dim=cfg["model"]["emb_dim"],
        n_layers=cfg["model"]["n_layers"],
        tf_heads=cfg["model"]["transformer"]["n_heads"],
        tf_dropout=cfg["model"]["transformer"]["dropout"],
        ssl_temperature=cfg["model"]["ssl"]["temperature"],
        ssl_edge_drop_rate=cfg["model"]["ssl"]["edge_drop_rate"],
        ssl_feature_drop_rate=cfg["model"]["ssl"]["feature_drop_rate"],
        topo_pe=TopoPEConfig(**cfg["model"].get("topo_pe", {"enabled": False})),
        cross_view_ssl=CrossViewSSLConfig(**cfg["model"].get("cross_view_ssl", {"enabled": False})),
    )

    paths = make_run_paths(cfg["train"]["runs_dir"], run_name=args.run_name)
    write_json(paths.run_dir / "config.json", cfg)

    sampler = BPRSampler(
        num_users=split.num_users,
        num_items=split.num_items,
        user_pos=split.user_pos_train,
        seed=int(cfg["seed"]),
    )

    trainer = Trainer(
        model=model,
        ui_norm_adj_coo=ui_norm,
        uu_norm_adj_coo=uu_norm,
        device=device,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        bpr_reg=cfg["train"]["bpr_reg"],
        ssl_weight=cfg["model"]["ssl"]["weight"],
        seed=int(cfg["seed"]),
        early_stopping_patience=int(cfg["train"].get("early_stopping_patience", 10)),
        cfg=cfg,
    )

    # steps per epoch: heuristic (cover interactions roughly once)
    steps = max(1, int(np.ceil(split.train_ui.shape[0] / cfg["train"]["batch_size"])))
    best = {"epoch": -1, "metric": -1.0}
    should_stop = False

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        if should_stop:
            break

        train_metrics = trainer.train_one_epoch(
            sampler=sampler,
            steps=steps,
            batch_size=int(cfg["train"]["batch_size"]),
        )

        log: Dict[str, Any] = {"epoch": epoch, **train_metrics}

        if epoch % int(cfg["train"]["eval_every"]) == 0:
            val_metrics = trainer.evaluate(
                user_pos_train=split.user_pos_train,
                user_pos_eval=split.user_pos_val,
                topk=cfg["train"]["topk"],
            )
            log.update({f"val_{k}": v for k, v in val_metrics.items()})
            key = f"val_NDCG@{cfg['train']['topk'][0]}"
            current_metric = log.get(key, -1.0)

            if current_metric > best["metric"]:
                best = {"epoch": epoch, "metric": current_metric}
                trainer.save(paths.ckpt_path, extra={"best": best, "num_users": split.num_users, "num_items": split.num_items})
                trainer.patience_counter = 0
            else:
                trainer.patience_counter += 1
                if trainer.patience_counter >= trainer.early_stopping_patience:
                    should_stop = True
                    print(f"Early stopping triggered at epoch {epoch}. No improvement for {trainer.early_stopping_patience} evaluation rounds.")

        write_json(paths.run_dir / "last_log.json", log)
        print(log)

    print("Best:", best, "ckpt:", str(paths.ckpt_path))

    # final test with best checkpoint (if exists)
    if paths.ckpt_path.exists():
        trainer.load(paths.ckpt_path)
        test_metrics = trainer.evaluate(
            user_pos_train=split.user_pos_train,
            user_pos_eval=split.user_pos_test,
            topk=cfg["train"]["topk"],
        )
        print("Test:", test_metrics)


if __name__ == "__main__":
    main()


