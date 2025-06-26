#!/usr/bin/env python3
"""
build_dataset.py  (flexible I/O version)
----------------------------------------
Creates:
    • <dataset>.inter   (RecBole-ready, mapped IDs, typed header)
    • <dataset>.item    (mapped metadata or minimal)
    • total_data.hdf5   (binary interactions)
    • index.npz         (train/val/test labels)

Example
-------
python build_dataset.py steam \
        --in_dir  /path/to/raw_data   \
        --out_dir /path/to/recbole_ready
"""

import numpy as np, pandas as pd, h5py, pathlib, argparse, random, sys, os

random.seed(42)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sequential_map(values):
    """old-id → 0…N-1  (ascending order)"""
    return {old: new for new, old in enumerate(sorted(set(values)))}

def _write_hdf5(users, items, ratings, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("user",   data=users,   dtype="i4", compression="gzip")
        f.create_dataset("item",   data=items,   dtype="i4", compression="gzip")
        f.create_dataset("rating", data=ratings, dtype="f",  compression="gzip")
    print(f"[LOG]  HDF5   → {out_path}  ({len(users):,} rows)")

def _write_inter(users, items, ratings, out_path):
    pd.DataFrame({
        "user_id:token": users,
        "item_id:token": items,
        "rating:float":  ratings
    }).to_csv(out_path, sep="\t", index=False)
    print(f"[LOG]  .inter → {out_path}  ({len(users):,} rows)")

def _split_indices(user_hist):
    """labels per interaction: 0 train | 1 val | 2 test | -1 skip"""
    lab = []
    for hist in user_hist:
        m = len(hist)
        if m < 3:
            lab.extend([-1]*m); continue
        order = list(range(m)); random.shuffle(order)
        s1, s2 = int(0.8*m), int(0.9*m)
        for t in range(m):
            if t == order[0]:          lab.append(2)
            elif t in order[:s1]:      lab.append(0)
            elif t in order[:s2]:      lab.append(1)
            else:                      lab.append(2)
    return np.asarray(lab, dtype="i1")

def _save_index(idx, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, data=idx)
    print(f"[LOG]  index  → {out_path}  ({len(idx):,} labels)")

# --------------------------------------------------------------------------- #
# dataset-specific loaders
# --------------------------------------------------------------------------- #
def load_movielens(in_dir):
    src = pathlib.Path(in_dir) / "ml-1m" / "ratings.dat"
    users, items, ratings = [], [], []
    with open(src) as f:
        for line in f:
            u,i,r,_ = line.strip().split("::")
            users.append(int(u)); items.append(int(i)); ratings.append(float(r))
    users = np.asarray([_sequential_map(users)[u] for u in users], dtype="i4")
    items = np.asarray(items, dtype="i4")
    ratings = np.asarray(ratings, dtype="f")
    return users, items, ratings, None

def load_steam(in_dir):
    base = pathlib.Path(in_dir) / "steam"
    inter = pd.read_csv(base / "steam.inter", sep="\t").dropna()
    meta  = pd.read_csv(base / "steam.item", sep="\t")

    u_map = _sequential_map(inter["user_id:token"])
    i_map = _sequential_map(
        np.concatenate([inter["product_id:token"],
                        meta["id:token"].dropna()])
    )

    users = np.asarray([u_map[u] for u in inter["user_id:token"]], dtype="i4")
    items = np.asarray([i_map[i] for i in inter["product_id:token"]], dtype="i4")
    ratings = inter["play_hours:float"].astype("f").to_numpy()

    # remap item metadata  (keep header = id:token)
    meta = meta[meta["id:token"].isin(i_map)].copy()
    meta["id:token"] = meta["id:token"].map(i_map)          # overwrite with new IDs
    cols = ["id:token"] + [c for c in meta.columns if c != "id:token"]
    return users, items, ratings, meta[cols]

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["ml-1m", "steam"],
                    help="Which dataset to process")
    ap.add_argument("--in_dir",  default="data",
                    help="Directory containing raw files (default: ./data)")
    ap.add_argument("--out_dir", default=None,
                    help="Directory to write processed files "
                         "(default: <in_dir>/<dataset>)")
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_dir or pathlib.Path(args.in_dir)/args.dataset)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\n🏗  Building dataset '{args.dataset}'")
    print(f"🔸 input  dir: {args.in_dir}")
    print(f"🔸 output dir: {out_root}\n")

    if args.dataset == "ml-1m":
        users, items, ratings, item_df = load_movielens(args.in_dir)
    else:
        users, items, ratings, item_df = load_steam(args.in_dir)

    _write_hdf5(users, items, ratings, out_root / "total_data.hdf5")
    _write_inter(users, items, ratings, out_root / f"{args.dataset}.inter")

    if item_df is None:
        # minimal file with just the mapped IDs
        pd.DataFrame({"id:token": np.unique(items)}
                    ).to_csv(out_root / f"{args.dataset}.item",
                            sep="\t", index=False)
        print("[LOG]  .item  → minimal file written (id:token).")
    else:
        # keep metadata but rename the ID column back to id:token
        item_df.rename(columns={"item_id:token": "id:token"}, inplace=True)
        cols = ["id:token"] + [c for c in item_df.columns if c != "id:token"]
        item_df[cols].to_csv(out_root / f"{args.dataset}.item",
                            sep="\t", index=False)
        print(f"[LOG]  .item  → metadata file written ({len(item_df):,} items, id:token)")

    # build index.npz
    hist = [[] for _ in range(users.max()+1)]
    for u,i,r in zip(users, items, ratings):
        hist[u].append((i,r))
    _save_index(_split_indices(hist), out_root / "index.npz")

    print("\nDone – all files perfectly aligned.\n")
