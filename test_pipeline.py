"""
Quick smoke test for the SSLGTSR project.
Run: python test_pipeline.py
"""

from __future__ import annotations

import numpy as np
import torch

from sslgtsr.data.dataset import SocialRecDataset
from sslgtsr.data.sampling import BPRSampler
from sslgtsr.models.sslgtsr import SSLGTSR
from sslgtsr.training.trainer import Trainer, build_graphs_from_split
from sslgtsr.models.topo_pe import TopoPEConfig
from sslgtsr.models.cross_view_ssl import CrossViewSSLConfig


def test_imports():
    """Test all major imports."""
    print("Testing imports...")
    from sslgtsr import SSLGTSR, SocialRecDataset, Trainer
    from sslgtsr.models import (
        LightGCNLayer, TransformerLayer, PropagationBlock,
        TwoViewFusion, GraphAttentionSampler
    )
    print("  All imports OK!")


def test_lightgcn_propagate():
    """Test LightGCN propagation function."""
    print("Testing LightGCN propagation...")
    from sslgtsr.models.lightgcn_layer import lightgcn_propagate
    
    n, d, k = 10, 8, 3
    adj = torch.eye(n, dtype=torch.float32)
    x = torch.randn(n, d)
    out = lightgcn_propagate(adj, x, k)
    assert out.shape == (n, d), f"Expected {(n,d)}, got {out.shape}"
    print("  LightGCN propagate OK!")


def test_transformer_layer():
    """Test Transformer layer."""
    print("Testing Transformer layer...")
    from sslgtsr.models.transformer_layer import TransformerLayer
    
    b, d, h, t = 4, 16, 2, 8
    layer = TransformerLayer(d, h)
    center = torch.randn(b, d)
    neighbors = torch.randn(b, t, d)
    out = layer(center, neighbors)
    assert out.shape == (b, d), f"Expected {(b,d)}, got {out.shape}"
    print("  Transformer layer OK!")


def test_propagation_block():
    """Test PropagationBlock."""
    print("Testing PropagationBlock...")
    from sslgtsr.models.propagation_block import PropagationBlock
    
    u, i, d = 10, 20, 16
    block = PropagationBlock(d, n_heads=2)
    
    ui_adj = torch.eye(u + i, dtype=torch.float32)
    q_u = torch.randn(u, d)
    e_i = torch.randn(i, d)
    attn_idx = torch.randint(0, u, (u, 5))
    
    q_out, e_out = block.forward_ui(q_u, e_i, ui_adj, attn_idx)
    assert q_out.shape == (u, d)
    assert e_out.shape == (i, d)
    
    p_u = torch.randn(u, d)
    p_out = block.forward_uu(p_u, ui_adj[:u, :u], attn_idx)
    assert p_out.shape == (u, d)
    print("  PropagationBlock OK!")


def test_fusion():
    """Test TwoViewFusion."""
    print("Testing TwoViewFusion...")
    from sslgtsr.models.fusion_layer import TwoViewFusion
    
    u, d, k = 10, 16, 4
    fusion = TwoViewFusion(d)
    p_list = [torch.randn(u, d) for _ in range(k)]
    q_list = [torch.randn(u, d) for _ in range(k)]
    out = fusion(p_list, q_list)
    assert out.shape == (u, d), f"Expected {(u,d)}, got {out.shape}"
    print("  TwoViewFusion OK!")


def test_attention_sampling():
    """Test GraphAttentionSampler."""
    print("Testing GraphAttentionSampler...")
    from sslgtsr.models.attn_sampling import GraphAttentionSampler, AttnSamplingConfig
    
    cfg = AttnSamplingConfig(enabled=True, topk=5)
    sampler = GraphAttentionSampler(cfg)
    
    u, d = 10, 16
    user_ui = torch.randn(u, d)
    user_uu = torch.randn(u, d)
    gs = torch.eye(u, dtype=torch.float32)
    
    indices = sampler.sample(user_ui, user_uu, gs)
    assert indices is not None
    assert indices.shape == (u, 5), f"Expected {(u,5)}, got {indices.shape}"
    print("  GraphAttentionSampler OK!")


def test_model_forward():
    """Test full model forward pass."""
    print("Testing SSLGTSR model forward...")
    
    u, i, d, k = 10, 20, 16, 2
    model = SSLGTSR(
        num_users=u, num_items=i, emb_dim=d, n_layers=k,
        tf_heads=2, tf_dropout=0.0,
        ssl_temperature=0.2, ssl_edge_drop_rate=0.1,
        ssl_feature_drop_rate=0.1,
    )
    
    ui_adj = torch.eye(u + i, dtype=torch.float32)
    uu_adj = torch.eye(u, dtype=torch.float32)
    
    model.init_attn_indices(u, topk=5, device=torch.device('cpu'))
    
    attn_idx = torch.randint(0, u, (u, 5))
    model.set_attn_indices(attn_idx)
    
    out = model(ui_adj, uu_adj)
    assert out.user_fused.shape == (u, d)
    assert out.item_ui.shape == (i, d)
    assert out.user_ui.shape == (u, d)
    assert out.user_uu.shape == (u, d)
    print("  SSLGTSR forward OK!")


def test_full_pipeline():
    """Test complete training pipeline with toy data."""
    print("Testing full pipeline with toy data...")
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        n_users, n_items = 50, 100
        rng = np.random.default_rng(42)
        
        ui_path = os.path.join(tmpdir, "interactions.txt")
        uu_path = os.path.join(tmpdir, "social.txt")
        
        with open(ui_path, "w") as f:
            for u in range(n_users):
                n_interactions = rng.integers(5, 20)
                items = rng.choice(n_items, size=n_interactions, replace=False)
                for it in items:
                    f.write(f"{u}\t{it}\n")
        
        with open(uu_path, "w") as f:
            for u in range(n_users):
                n_friends = rng.integers(2, 8)
                friends = rng.choice(n_users, size=n_friends, replace=False)
                for v in friends:
                    if v != u:
                        f.write(f"{u}\t{v}\n")
        
        ds = SocialRecDataset(
            data_dir=tmpdir,
            interactions_file="interactions.txt",
            social_file="social.txt",
            is_social_directed=False,
            val_ratio=0.1,
            test_ratio=0.1,
            min_user_interactions=3,
            seed=42,
        )
        split = ds.load()
        print(f"  Dataset: {split.num_users} users, {split.num_items} items")
        
        ui_norm, uu_norm = build_graphs_from_split(
            num_users=split.num_users,
            num_items=split.num_items,
            train_ui=split.train_ui,
            uu=split.uu,
            directed=False,
        )
        
        model = SSLGTSR(
            num_users=split.num_users,
            num_items=split.num_items,
            emb_dim=32,
            n_layers=2,
            tf_heads=2,
            tf_dropout=0.1,
            ssl_temperature=0.2,
            ssl_edge_drop_rate=0.1,
            ssl_feature_drop_rate=0.1,
            topo_pe=TopoPEConfig(enabled=False),
            cross_view_ssl=CrossViewSSLConfig(enabled=False),
        )
        
        trainer = Trainer(
            model=model,
            ui_norm_adj_coo=ui_norm,
            uu_norm_adj_coo=uu_norm,
            device=torch.device('cpu'),
            lr=0.001,
            weight_decay=0.0,
            bpr_reg=1e-5,
            ssl_weight=0.0,
            seed=42,
            early_stopping_patience=5,
        )
        
        sampler = BPRSampler(
            num_users=split.num_users,
            num_items=split.num_items,
            user_pos=split.user_pos_train,
            seed=42,
        )
        
        print("  Training 2 steps...")
        metrics = trainer.train_one_epoch(sampler, steps=2, batch_size=32)
        print(f"  Train metrics: {metrics}")
        
        print("  Evaluating...")
        val_metrics = trainer.evaluate(
            user_pos_train=split.user_pos_train,
            user_pos_eval=split.user_pos_val,
            topk=[10],
        )
        print(f"  Val metrics: {val_metrics}")
        
        print("  Full pipeline OK!")


if __name__ == "__main__":
    print("=" * 60)
    print("SSLGTSR Pipeline Smoke Test")
    print("=" * 60)
    
    test_imports()
    test_lightgcn_propagate()
    test_transformer_layer()
    test_propagation_block()
    test_fusion()
    test_attention_sampling()
    test_model_forward()
    test_full_pipeline()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
