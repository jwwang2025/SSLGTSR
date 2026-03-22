from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SocialRecSplit:
    train_ui: np.ndarray  # [E_train, 2]
    val_ui: np.ndarray  # [E_val, 2]
    test_ui: np.ndarray  # [E_test, 2]
    uu: np.ndarray  # [E_uu, 2]
    num_users: int
    num_items: int
    user_pos_train: List[np.ndarray]  # user -> positive items in train
    user_pos_val: List[np.ndarray]  # user -> positive items in val
    user_pos_test: List[np.ndarray]  # user -> positive items in test
    user_pos_all: List[np.ndarray]  # user -> positive items in train+val+test


def _read_two_col(path: Path) -> np.ndarray:
    rows: List[Tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            a, b = int(parts[0]), int(parts[1])
            rows.append((a, b))
    return np.asarray(rows, dtype=np.int64)


def _remap_ids(edges: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    uniq = np.unique(edges.reshape(-1))
    mapping = {int(x): int(i) for i, x in enumerate(uniq)}
    remapped = np.vectorize(mapping.get)(edges)
    return remapped.astype(np.int64), mapping


def _build_user_pos(num_users: int, ui_edges: np.ndarray) -> List[np.ndarray]:
    bucket: List[List[int]] = [[] for _ in range(num_users)]
    for u, i in ui_edges:
        bucket[int(u)].append(int(i))
    return [np.asarray(v, dtype=np.int64) for v in bucket]


class SocialRecDataset:
    """
    A minimal dataset loader for social recommendation:
    - interactions.txt: (user_id, item_id)
    - social.txt: (user_id, user_id)

    IDs can be arbitrary integers; they will be remapped to contiguous [0..N).
    """

    def __init__(
        self,
        data_dir: str | Path,
        interactions_file: str = "interactions.txt",
        social_file: str = "social.txt",
        is_social_directed: bool = False,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        min_user_interactions: int = 5,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.interactions_path = self.data_dir / interactions_file
        self.social_path = self.data_dir / social_file
        self.is_social_directed = is_social_directed
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.min_user_interactions = int(min_user_interactions)
        self.seed = int(seed)

    def load(self) -> SocialRecSplit:
        rng = np.random.default_rng(self.seed)

        ui_raw = _read_two_col(self.interactions_path)
        if ui_raw.size == 0:
            raise ValueError(f"No interactions found in {self.interactions_path}")

        # remap users and items separately
        user_ids = np.unique(ui_raw[:, 0])
        item_ids = np.unique(ui_raw[:, 1])
        user_map = {int(x): int(i) for i, x in enumerate(user_ids)}
        item_map = {int(x): int(i) for i, x in enumerate(item_ids)}
        ui = np.column_stack(
            [
                np.vectorize(user_map.get)(ui_raw[:, 0]),
                np.vectorize(item_map.get)(ui_raw[:, 1]),
            ]
        ).astype(np.int64)

        num_users = int(len(user_ids))
        num_items = int(len(item_ids))

        # filter users with too few interactions
        counts = np.bincount(ui[:, 0], minlength=num_users)
        keep_users = np.where(counts >= self.min_user_interactions)[0]
        if len(keep_users) < num_users:
            keep_mask = np.isin(ui[:, 0], keep_users)
            ui = ui[keep_mask]
            # re-index users to be contiguous
            new_user_map = {int(u): int(i) for i, u in enumerate(keep_users)}
            ui[:, 0] = np.vectorize(new_user_map.get)(ui[:, 0])
            num_users = int(len(keep_users))

        # split per-user (leave-one-out style generalized by ratios)
        user_pos_all = _build_user_pos(num_users, ui)
        train_rows: List[Tuple[int, int]] = []
        val_rows: List[Tuple[int, int]] = []
        test_rows: List[Tuple[int, int]] = []

        for u in range(num_users):
            items = user_pos_all[u]
            if items.size < 3:
                # keep all in train if too few
                for it in items.tolist():
                    train_rows.append((u, it))
                continue
            items = rng.permutation(items)
            n = items.size
            n_test = max(1, int(round(n * self.test_ratio)))
            n_val = max(1, int(round(n * self.val_ratio)))
            n_train = max(1, n - n_val - n_test)
            train_items = items[:n_train]
            val_items = items[n_train : n_train + n_val]
            test_items = items[n_train + n_val :]
            for it in train_items.tolist():
                train_rows.append((u, it))
            for it in val_items.tolist():
                val_rows.append((u, it))
            for it in test_items.tolist():
                test_rows.append((u, it))

        train_ui = np.asarray(train_rows, dtype=np.int64)
        val_ui = np.asarray(val_rows, dtype=np.int64)
        test_ui = np.asarray(test_rows, dtype=np.int64)

        user_pos_train = _build_user_pos(num_users, train_ui)
        user_pos_val = _build_user_pos(num_users, val_ui)
        user_pos_test = _build_user_pos(num_users, test_ui)
        user_pos_all2 = _build_user_pos(num_users, np.vstack([train_ui, val_ui, test_ui]))

        # load social edges and remap using user_map (unknown users dropped)
        if self.social_path.exists():
            uu_raw = _read_two_col(self.social_path)
            if uu_raw.size == 0:
                uu = np.zeros((0, 2), dtype=np.int64)
            else:
                # map raw user ids -> internal ids; drop missing
                def map_user(x: int) -> int:
                    return user_map.get(int(x), -1)

                a = np.vectorize(map_user)(uu_raw[:, 0])
                b = np.vectorize(map_user)(uu_raw[:, 1])
                keep = (a >= 0) & (b >= 0)
                uu = np.column_stack([a[keep], b[keep]]).astype(np.int64)
                # if we re-indexed users due to filtering, map again
                if "new_user_map" in locals():
                    def map_to_new_id(x: int) -> int:
                        return new_user_map.get(int(x), -1)
                    a = np.vectorize(map_to_new_id)(uu[:, 0])
                    b = np.vectorize(map_to_new_id)(uu[:, 1])
                    keep = (a >= 0) & (b >= 0)
                    uu = np.column_stack([a[keep], b[keep]]).astype(np.int64)
        else:
            uu = np.zeros((0, 2), dtype=np.int64)

        return SocialRecSplit(
            train_ui=train_ui,
            val_ui=val_ui,
            test_ui=test_ui,
            uu=uu,
            num_users=num_users,
            num_items=num_items,
            user_pos_train=user_pos_train,
            user_pos_val=user_pos_val,
            user_pos_test=user_pos_test,
            user_pos_all=user_pos_all2,
        )


