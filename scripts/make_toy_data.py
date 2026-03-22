from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_users", type=int, default=200)
    ap.add_argument("--n_items", type=int, default=300)
    ap.add_argument("--avg_ui_deg", type=float, default=20.0)
    ap.add_argument("--avg_uu_deg", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # interactions: each user interacts with Poisson(avg_ui_deg) items
    ui_rows = []
    for u in range(args.n_users):
        deg = max(1, int(rng.poisson(args.avg_ui_deg)))
        items = rng.choice(args.n_items, size=min(deg, args.n_items), replace=False)
        for i in items.tolist():
            ui_rows.append((u, int(i)))

    # social edges: random graph
    uu_rows = []
    for u in range(args.n_users):
        deg = int(rng.poisson(args.avg_uu_deg))
        if deg <= 0:
            continue
        friends = rng.choice(args.n_users, size=min(deg, args.n_users - 1), replace=False)
        for v in friends.tolist():
            if int(v) == u:
                continue
            uu_rows.append((u, int(v)))

    inter_path = out_dir / "interactions.txt"
    soc_path = out_dir / "social.txt"
    with inter_path.open("w", encoding="utf-8") as f:
        for u, i in ui_rows:
            f.write(f"{u}\t{i}\n")
    with soc_path.open("w", encoding="utf-8") as f:
        for u, v in uu_rows:
            f.write(f"{u}\t{v}\n")

    print(f"Wrote {len(ui_rows)} interactions to {inter_path}")
    print(f"Wrote {len(uu_rows)} social edges to {soc_path}")


if __name__ == "__main__":
    main()


