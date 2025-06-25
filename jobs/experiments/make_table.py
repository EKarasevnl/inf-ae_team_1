#!/usr/bin/env python3
"""
Create LaTeX-ready rows from RecBole *.out logs.

$ ./make_latex_table.py ml-1m
$ ./make_latex_table.py slurm_out/steam -o slurm_out/tables/steam_table.txt
"""
from __future__ import annotations
import argparse, glob, re, ast, textwrap
from collections import defaultdict, Counter
from pathlib import Path

# ────────────────────────── CONSTANTS ───────────────────────────────────────
COLUMN_ORDER = ["POP", "NeuMF", "LightGCN", "EASE", "MVAE"]
BASIC_METRICS = ["recall", "ndcg", "psp", "giniindex",
                 "mmf-category", "mmf_c10-category"]
# ────────────────────────────────────────────────────────────────────────────


def pull_numbers(text: str) -> dict[str, float]:
    """Return {'metric_name': value, …} from a .out file (keys are lower-case)."""
    m = re.search(r'test result:\s*OrderedDict\(\[(.*?)\]\)', text, re.S)
    if not m:
        return {}
    items = ast.literal_eval(f"[{m.group(1)}]")      # safe
    return {k.lower(): v for k, v in items}


def guess_second_k(all_results: dict[str, dict[str, float]]) -> int:
    """
    Detect which second top-k (20, 100, …) is present besides 10.
    If multiple, pick the most common; default to 100.
    """
    ks = []
    for metrics in all_results.values():
        ks += [int(m.split("@")[1])
               for m in metrics
               if "@" in m and not m.endswith("@10")]
    if not ks:
        return 100
    # most common non-10 k
    return Counter(ks).most_common(1)[0][0]


def build_metric_map(k2: int) -> tuple[dict[str, str],
                                       list[str],
                                       dict[str, int]]:
    """
    Generate all lookup tables once we know whether the second cut-off
    in your logs is 20 or 100 (``k2``).

    Returns
    -------
    metric_map : dict[str, str]
        Key  : metric name as it appears in the *.out* file (lower-case)
        Value: label to show in the LaTeX table.

    row_order : list[str]
        Sequence of row labels **exactly** as they must appear in the table
        (keeps the pair-wise @10 / @k₂ ordering you were using before).

    scale : dict[str, int]
        Multipliers (AUC stays 1, everything else is ×100).
    """
    # ------------------------------------------------------------------ labels
    metric_map = {
        "auc"                    : "AUC",

        "recall@10"              : "HR@10",
        f"recall@{k2}"           : f"HR@{k2}",

        "ndcg@10"                : "NDCG@10",
        f"ndcg@{k2}"             : f"NDCG@{k2}",

        "psp@10"                 : "PSP@10",
        f"psp@{k2}"              : f"PSP@{k2}",

        "giniindex@10"           : "Gini-Index@10",
        f"giniindex@{k2}"        : f"Gini-Index@{k2}",

        "mmf-category@10"        : "MMF@10",
        f"mmf-category@{k2}"     : f"MMF@{k2}",

        "mmf_c10-category@10"    : "MMF_c10@10",
        f"mmf_c10-category@{k2}" : f"MMF_c10@{k2}",
    }

    # ---------------------------------------------------------------- row order
    row_order = [
        "AUC",
        "HR@10",               f"HR@{k2}",
        "NDCG@10",             f"NDCG@{k2}",
        "PSP@10",              f"PSP@{k2}",
        "Gini-Index@10",       f"Gini-Index@{k2}",
        "MMF@10",              f"MMF@{k2}",
        "MMF_c10@10",          f"MMF_c10@{k2}",
    ]

    # ------------------------------------------------------------- multipliers
    scale = {k: (1 if k == "auc" else 100) for k in metric_map}

    return metric_map, row_order, scale


def build_table_rows(results: dict[str, dict[str, float]],
                     metric_map: dict[str, str],
                     row_order: list[str],
                     scale: dict[str, int],
                     folder_label: str) -> list[str]:
    """Return LaTeX table rows (no bold)."""
    model_names = [m for m in COLUMN_ORDER if m in results]
    header = f"\\multirow{{{len(row_order)}}}{{*}}{{Our {folder_label}}}"

    rows = []
    # row_order holds the pretty labels; find the raw key each time
    for i, row_label in enumerate(row_order):
        metric_key = next(k for k, v in metric_map.items() if v == row_label)
        row_vals = []
        for m in model_names:
            v = results[m].get(metric_key)
            if v is None:
                row_vals.append("--")
            else:
                s = v * scale[metric_key]
                fmt = "{:.4g}" if metric_key == "auc" else "{:.2f}"
                row_vals.append(fmt.format(s))
        rows.append(
            (header if i == 0 else " " * len(header))
            + f" & {row_label:<15} & " + " & ".join(row_vals) + r" \\"
        )
    return rows


def main(folder: str, out_file: str | None):
    folder = folder.rstrip("/")

    # ── collect numbers ────────────────────────────────────────────────────
    results: dict[str, dict[str, float]] = defaultdict(dict)
    for file in glob.glob(f"{folder}/*.out"):
        stem       = Path(file).stem
        base_model = re.sub(r"_auc$", "", stem, flags=re.I)
        with open(file, "r", encoding="utf-8", errors="ignore") as fh:
            parsed = pull_numbers(fh.read())
        if parsed:
            results[base_model].update(parsed)

    if not results:
        raise SystemExit("No usable *.out files found in that folder.")

    # ── configure metric map based on detected k2 ──────────────────────────
    k2 = guess_second_k(results)      # e.g. 20 or 100
    metric_map, row_order, scale = build_metric_map(k2)

    # ── build rows & write file ────────────────────────────────────────────
    rows   = build_table_rows(results, metric_map, row_order,
                              scale, Path(folder).name.upper())
    target = Path(out_file or f"{Path(folder).name.upper()}_table.txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows), encoding="utf-8")
    print(f"✔  wrote {target}   (second-k = {k2})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        Parse each *.out file in FOLDER, merge <model>.out + <model>_auc.out,
        detect whether your logs use @20 or @100, and emit LaTeX rows.
        """).strip()
    )
    ap.add_argument("folder", help="directory containing the .out logs")
    ap.add_argument("-o", "--out_file",
                    help="output .txt path (default: FOLDERNAME_table.txt)")
    main(**vars(ap.parse_args()))
