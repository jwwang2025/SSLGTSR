from __future__ import annotations

"""
Convert HetRec 2011 LastFM 2K dataset into this repo's expected format:

- Input (original dataset):
    dataset/hetrec2011-lastfm-2k/user_artists.dat      (userID, artistID, weight)
    dataset/hetrec2011-lastfm-2k/user_friends.dat      (userID, friendID)

- Output (for our framework):
    <out_dir>/interactions.txt   (user_id item_id)
    <out_dir>/social.txt         (user_id user_id)

Here we treat every (user, artist) pair as one implicit interaction, ignoring the play-count `weight`.
"""

import argparse
from pathlib import Path


def _read_tsv(path: Path, has_header: bool = True) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first and has_header:
                first = False
                continue
            parts = line.split("\t")
            rows.append(parts)
    return rows


def convert_lastfm(
    src_dir: str | Path,
    out_dir: str | Path,
    min_weight: int = 0,
) -> None:
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ua_path = src_dir / "user_artists.dat"
    uf_path = src_dir / "user_friends.dat"

    if not ua_path.exists():
        raise FileNotFoundError(f"user_artists.dat not found at {ua_path}")
    if not uf_path.exists():
        raise FileNotFoundError(f"user_friends.dat not found at {uf_path}")

    # interactions: userID artistID (implicit feedback)
    ua_rows = _read_tsv(ua_path, has_header=True)
    inter_out = out_dir / "interactions.txt"
    with inter_out.open("w", encoding="utf-8") as f:
        for cols in ua_rows:
            if len(cols) < 2:
                continue
            user_id = cols[0]
            artist_id = cols[1]
            weight = int(cols[2]) if len(cols) >= 3 else 1
            if weight < min_weight:
                continue
            f.write(f"{user_id}\t{artist_id}\n")

    # social edges: userID friendID
    uf_rows = _read_tsv(uf_path, has_header=True)
    social_out = out_dir / "social.txt"
    with social_out.open("w", encoding="utf-8") as f:
        for cols in uf_rows:
            if len(cols) < 2:
                continue
            u = cols[0]
            v = cols[1]
            f.write(f"{u}\t{v}\n")

    print(f"Wrote interactions to {inter_out}")
    print(f"Wrote social edges to {social_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Path to original HetRec LastFM 2K directory (e.g. dataset/hetrec2011-lastfm-2k)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for interactions.txt / social.txt (e.g. data/lastfm)",
    )
    ap.add_argument(
        "--min_weight",
        type=int,
        default=0,
        help="Optional: minimum play-count weight to keep a (user, artist) interaction.",
    )
    args = ap.parse_args()
    convert_lastfm(args.src_dir, args.out_dir, min_weight=args.min_weight)


if __name__ == "__main__":
    main()


