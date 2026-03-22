"""
Hyperparameter search script for SSLGTSR ablation study.
Section 4.3: Effect of attention sample size, message propagation layers, and SSL loss weight.

Usage:
    python search_hyperparams.py --config configs/ablation.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from sslgtsr.utils.config import load_yaml
from sslgtsr.utils.logging import write_json


def merge_config(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def run_experiment(
    config: Dict[str, Any],
    data_dir: str,
    run_name: str,
) -> Dict[str, float]:
    """
    Run a single experiment with given config.
    Returns validation metrics.
    """
    from sslgtsr.data.dataset import SocialRecDataset
    from sslgtsr.data.sampling import BPRSampler
    from sslgtsr.models.sslgtsr import SSLGTSR
    from sslgtsr.training.trainer import Trainer, build_graphs_from_split
    from sslgtsr.models.topo_pe import TopoPEConfig
    from sslgtsr.models.cross_view_ssl import CrossViewSSLConfig
    from sslgtsr.utils.seed import seed_everything

    import numpy as np
    import torch

    seed_everything(int(config["seed"]))

    device = torch.device("cuda" if config.get("device", "cpu").lower() == "cuda" and torch.cuda.is_available() else "cpu")

    ds = SocialRecDataset(
        data_dir=data_dir,
        interactions_file=config["data"]["interactions_file"],
        social_file=config["data"]["social_file"],
        is_social_directed=config["data"]["is_social_directed"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        min_user_interactions=config["data"]["min_user_interactions"],
        seed=int(config["seed"]),
    )
    split = ds.load()

    ui_norm, uu_norm = build_graphs_from_split(
        num_users=split.num_users,
        num_items=split.num_items,
        train_ui=split.train_ui,
        uu=split.uu,
        directed=config["data"]["is_social_directed"],
    )

    model = SSLGTSR(
        num_users=split.num_users,
        num_items=split.num_items,
        emb_dim=config["model"]["emb_dim"],
        n_layers=config["model"]["n_layers"],
        tf_heads=config["model"]["transformer"]["n_heads"],
        tf_dropout=config["model"]["transformer"]["dropout"],
        ssl_temperature=config["model"]["ssl"]["temperature"],
        ssl_edge_drop_rate=config["model"]["ssl"]["edge_drop_rate"],
        ssl_feature_drop_rate=config["model"]["ssl"]["feature_drop_rate"],
        topo_pe=TopoPEConfig(**config["model"].get("topo_pe", {"enabled": False})),
        cross_view_ssl=CrossViewSSLConfig(**config["model"].get("cross_view_ssl", {"enabled": False})),
    )

    sampler = BPRSampler(
        num_users=split.num_users,
        num_items=split.num_items,
        user_pos=split.user_pos_train,
        seed=int(config["seed"]),
    )

    trainer = Trainer(
        model=model,
        ui_norm_adj_coo=ui_norm,
        uu_norm_adj_coo=uu_norm,
        device=device,
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
        bpr_reg=config["train"]["bpr_reg"],
        ssl_weight=config["model"]["ssl"]["weight"],
        seed=int(config["seed"]),
        early_stopping_patience=int(config["train"].get("early_stopping_patience", 10)),
        cfg=config,
    )

    steps = max(1, int(np.ceil(split.train_ui.shape[0] / config["train"]["batch_size"])))
    best = {"epoch": -1, "metric": -1.0}

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = trainer.train_one_epoch(
            sampler=sampler,
            steps=steps,
            batch_size=int(config["train"]["batch_size"]),
        )

        if epoch % int(config["train"]["eval_every"]) == 0:
            val_metrics = trainer.evaluate(
                user_pos_train=split.user_pos_train,
                user_pos_eval=split.user_pos_val,
                topk=config["train"]["topk"],
            )
            key = f"val_{config.get('eval_metric', 'NDCG@10')}"
            current_metric = val_metrics.get(config.get("eval_metric", "NDCG@10"), -1.0)

            if current_metric > best["metric"]:
                best = {"epoch": epoch, "metric": current_metric}

    return best


def run_search(
    config: Dict[str, Any],
    data_dir: str,
    search_type: str,
    values: List[Any],
    fixed_params: Dict[str, Any],
    eval_metric: str,
) -> List[Dict[str, Any]]:
    """
    Run hyperparameter search for a specific parameter.
    """
    results = []
    base_cfg = config["base"]

    print(f"\n{'='*60}")
    print(f"Search: {search_type}")
    print(f"Values: {values}")
    print(f"{'='*60}\n")

    for value in values:
        # Create config for this run
        cfg = copy.deepcopy(base_cfg)
        cfg = merge_config(cfg, fixed_params)

        # Override the search parameter
        if search_type == "attn_sample_size":
            cfg["model"]["attn_sample_size"] = value
        elif search_type == "num_layers":
            cfg["model"]["n_layers"] = value
        elif search_type == "ssl_weight":
            cfg["model"]["ssl"]["weight"] = value
            cfg["train"]["bpr_reg"] = fixed_params.get("bpr_reg", 1.0e-5)

        run_name = f"{search_type}_{value}"
        print(f"\n>>> Running: {search_type}={value}")

        try:
            result = run_experiment(cfg, data_dir, run_name)
            results.append({
                "parameter": search_type,
                "value": value,
                "best_epoch": result["epoch"],
                "best_metric": result["metric"],
                "eval_metric": eval_metric,
            })
            print(f">>> Result: {eval_metric}={result['metric']:.4f} at epoch {result['epoch']}")
        except Exception as e:
            print(f">>> Error: {e}")
            results.append({
                "parameter": search_type,
                "value": value,
                "best_epoch": -1,
                "best_metric": -1.0,
                "error": str(e),
            })

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Hyperparameter search for SSLGTSR ablation study")
    ap.add_argument("--config", type=str, required=True, help="Path to ablation config YAML")
    ap.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    ap.add_argument("--search", type=str, default="all", choices=["all", "attn_sample_size", "num_layers", "ssl_weight"], help="Which search to run")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_dir = args.data_dir
    eval_metric = cfg.get("eval_metric", "NDCG@10")
    base_cfg = cfg["base"]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_cfg["train"]["runs_dir"]) / f"ablation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Run searches
    search_config = cfg.get("search", {})

    if args.search in ["all", "attn_sample_size"]:
        if "attn_sample_size" in search_config:
            results = run_search(
                cfg, data_dir,
                "attn_sample_size",
                search_config["attn_sample_size"]["values"],
                search_config["attn_sample_size"]["fixed"],
                eval_metric,
            )
            all_results.extend(results)

    if args.search in ["all", "num_layers"]:
        if "num_layers" in search_config:
            results = run_search(
                cfg, data_dir,
                "num_layers",
                search_config["num_layers"]["values"],
                search_config["num_layers"]["fixed"],
                eval_metric,
            )
            all_results.extend(results)

    if args.search in ["all", "ssl_weight"]:
        if "ssl_weight" in search_config:
            results = run_search(
                cfg, data_dir,
                "ssl_weight",
                search_config["ssl_weight"]["values"],
                search_config["ssl_weight"]["fixed"],
                eval_metric,
            )
            all_results.extend(results)

    # Save results
    results_file = output_dir / "search_results.json"
    write_json(results_file, {
        "timestamp": timestamp,
        "eval_metric": eval_metric,
        "results": all_results,
    })

    print(f"\n{'='*60}")
    print("Search completed!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}\n")

    # Print summary
    print("\n=== Summary ===")
    for search_type in ["attn_sample_size", "num_layers", "ssl_weight"]:
        type_results = [r for r in all_results if r.get("parameter") == search_type]
        if type_results:
            print(f"\n{search_type}:")
            for r in type_results:
                if "error" not in r:
                    print(f"  {r['value']}: {eval_metric}={r['best_metric']:.4f}")
                else:
                    print(f"  {r['value']}: ERROR")


if __name__ == "__main__":
    main()
