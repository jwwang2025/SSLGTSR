from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _dcg(rel: np.ndarray) -> float:
    # rel is binary relevance in ranked order
    if rel.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, rel.size + 2))
    return float((rel / denom).sum())


def recall_at_k(
    ranked_items: np.ndarray,
    ground_truth: Sequence[int],
    k: int,
) -> Tuple[float, float]:
    """
    Compute Recall@K and HR@K (Hit Rate).
    Recall@K = (# of hits in top K) / (# of ground truth items)
    HR@K = 1 if any hit in top K, else 0
    """
    topk = ranked_items[:k]
    gt_set = set(map(int, ground_truth))
    num_gt = len(gt_set)

    if num_gt == 0:
        return 0.0, 0.0

    hits = sum(1 for i in topk if int(i) in gt_set)
    recall = hits / num_gt
    hr = 1.0 if hits > 0 else 0.0
    return hr, float(recall)


def hr_ndcg_at_k(
    ranked_items: np.ndarray,
    ground_truth: Sequence[int],
    k: int,
) -> Tuple[float, float]:
    topk = ranked_items[:k]
    gt_set = set(map(int, ground_truth))
    rel = np.array([1.0 if int(i) in gt_set else 0.0 for i in topk], dtype=np.float32)
    hr = 1.0 if rel.sum() > 0 else 0.0
    dcg = _dcg(rel)
    # ideal dcg: all relevant items at the top
    ideal_rel = np.sort(rel)[::-1]
    idcg = _dcg(ideal_rel)
    ndcg = 0.0 if idcg == 0 else (dcg / idcg)
    return hr, float(ndcg)


def evaluate_topk(
    scores: np.ndarray,  # [U, I]
    user_pos_train: List[np.ndarray],
    user_pos_eval: List[np.ndarray],
    topk_list: Sequence[int],
) -> Dict[str, float]:
    """
    scores: predicted preference matrix
    user_pos_train: training positives (to be masked out)
    user_pos_eval: eval positives (val or test) as ground-truth
    """
    max_k = int(max(topk_list))
    recall_sum = {k: 0.0 for k in topk_list}
    ndcg_sum = {k: 0.0 for k in topk_list}
    n_users = scores.shape[0]

    for u in range(n_users):
        # mask training positives
        train_pos = user_pos_train[u]
        if train_pos.size > 0:
            scores[u, train_pos] = -1e9
        gt = user_pos_eval[u]
        if gt.size == 0:
            continue

        ranked = np.argpartition(-scores[u], max_k - 1)[:max_k]
        # sort those candidates
        ranked = ranked[np.argsort(-scores[u, ranked])]

        for k in topk_list:
            hr, recall = recall_at_k(ranked, gt, int(k))
            _, ndcg = hr_ndcg_at_k(ranked, gt, int(k))
            recall_sum[k] += recall
            ndcg_sum[k] += ndcg

    denom = float(n_users)
    out: Dict[str, float] = {}
    for k in topk_list:
        out[f"Recall@{k}"] = recall_sum[k] / denom
        out[f"NDCG@{k}"] = ndcg_sum[k] / denom
    return out


