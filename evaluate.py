from __future__ import annotations

import argparse

import torch

from sslgtsr.data.dataset import SocialRecDataset
from sslgtsr.models.sslgtsr import SSLGTSR
from sslgtsr.training.trainer import Trainer, build_graphs_from_split
from sslgtsr.utils.config import load_yaml
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
    ap.add_argument("--ckpt", type=str, required=True)
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
    )
    extra = trainer.load(args.ckpt)
    metrics = trainer.evaluate(
        user_pos_train=split.user_pos_train,
        user_pos_eval=split.user_pos_test,
        topk=cfg["train"]["topk"],
    )
    print("Loaded extra:", extra)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()


