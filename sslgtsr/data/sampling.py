from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np


@dataclass
class BPRBatch:
    users: np.ndarray  # [B]
    pos_items: np.ndarray  # [B]
    neg_items: np.ndarray  # [B]


class BPRSampler:
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_pos: List[np.ndarray],
        seed: int = 42,
    ) -> None:
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.user_pos = user_pos
        self.rng = np.random.default_rng(seed)
        self.user_pos_sets = [set(map(int, arr.tolist())) for arr in user_pos]

        self.users_with_pos = np.array([u for u in range(self.num_users) if len(self.user_pos_sets[u]) > 0], dtype=np.int64)
        if self.users_with_pos.size == 0:
            raise ValueError("No users with positive interactions for sampling.")

    def sample(self, batch_size: int) -> BPRBatch:
        users = self.rng.choice(self.users_with_pos, size=batch_size, replace=True)
        pos_items = np.empty(batch_size, dtype=np.int64)
        neg_items = np.empty(batch_size, dtype=np.int64)

        for idx, u in enumerate(users.tolist()):
            pos = self.user_pos[u]
            pos_items[idx] = int(pos[self.rng.integers(0, pos.size)])

            # rejection sampling for negative
            while True:
                ni = int(self.rng.integers(0, self.num_items))
                if ni not in self.user_pos_sets[u]:
                    neg_items[idx] = ni
                    break

        return BPRBatch(users=users, pos_items=pos_items, neg_items=neg_items)


