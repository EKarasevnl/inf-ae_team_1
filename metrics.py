import numpy as np
from collections import Counter
import json

SPLIT_STR = True # In case the category string is a non-list-formatted-string, we will split it by space, otherwise we will parse it as it is.

def _parse_genre_string(value):
    """
    Parses a genre string into a list of genres, handling multiple formats.
    - Handles: "Action Crime Thriller"
    - Handles: "[Casual, Free to Play, Indie]"
    - Handles: "Action" (single genre)
    """
    if not isinstance(value, str):
        return value
    
    # if SPLIT_STR is True we will preprocess the string to handle both cases:
    # 1. If it starts with '[' and ends with ']', treat it as a list.
    # 2. If it does not, treat it as a space-separated string.
    if SPLIT_STR:
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            content = value[1:-1]
            return [genre.strip() for genre in content.split(',') if genre.strip()]

        return [genre.strip() for genre in value.split(' ') if genre.strip()]
    else:
        return value

def compute_mmf(user_recommendations, topk, group_key, group_map, group_weights=None):
    """
    Compute weighted Max-Min Fairness (MMF) with robust parsing.
    """
    metric_name = f"MMF-{group_key.upper()}"
    print(f"\n=======================================================================")
    print(f"[MMF DEBUG] INITIALIZING MMF COMPUTATION FOR: {metric_name}")
    print(f"=======================================================================")

    # --- Step 1: Calculate Catalog Share using robust parsing ---
    print("[MMF] STEP 1: Calculating Catalog Share (Weight) for each individual group...")
    if not group_map:
        print("[MMF] CRITICAL: 'group_map' is required. Aborting.")
        return {}

    all_catalog_groups = Counter()
    for raw_groups in group_map.values():
        groups = _parse_genre_string(raw_groups)
        if isinstance(groups, list):
            all_catalog_groups.update(groups)
        else:
            all_catalog_groups.update([groups])

    group_counts = all_catalog_groups
    total_group_tags_in_catalog = sum(group_counts.values())
    print(f"[MMF] Found {total_group_tags_in_catalog} total group tags across {len(group_counts)} unique groups.")
    
    if not group_weights:
        group_weights = {g: c / total_group_tags_in_catalog for g, c in group_counts.items()} if total_group_tags_in_catalog > 0 else {}
        print("[MMF] Calculated group weights based on catalog group distribution.")
    else:
        print("[MMF] Using pre-defined group weights.")

    mmf_metrics = {}
    k_to_process = list(topk)

    # --- Step 2: Loop through each K ---
    for k in k_to_process:
        print(f"\n[MMF] ----------------------------------------------------")
        print(f"[MMF] STEP 2: Calculating {metric_name} for recommendations at k={k}")
        print(f"[MMF] ----------------------------------------------------")
        
        recommendations_for_k = user_recommendations.get(k, [])
        metric_label = f"{metric_name}@{k}"

        # --- Calculate Exposure Share using robust parsing ---
        all_exposed_groups = Counter()
        for item in recommendations_for_k:
            item_id = item.get("id")
            if item_id and item_id in group_map:
                raw_groups = group_map[item_id]
                groups = _parse_genre_string(raw_groups)
                
                if isinstance(groups, list):
                    all_exposed_groups.update(groups)
                else:
                    all_exposed_groups.update([groups])
        
        group_exposures = all_exposed_groups
        total_exposures = sum(group_exposures.values())
        print(f"[MMF] Found {len(recommendations_for_k)} recommended items, corresponding to {total_exposures} total group exposures.")
        
        if total_exposures == 0:
            mmf_metrics[metric_label] = 0.0
            print(f"[MMF] -> RESULT: {metric_label}: 0.0000 (No recommendations)")
            continue

        exposure_share = {group: count / total_exposures for group, count in group_exposures.items()}
        fairness_scores = {g: exposure_share.get(g, 0.0) / w if w > 0 else 0.0 for g, w in group_weights.items()}

        # --- Create and print detailed summary (original format) ---
        print("\n[MMF] Sub-step c: Generating MMF summary...")

        summary_data = []
        for group in sorted(group_weights.keys()):
            summary_data.append({ "Group": group, "MMF Score": fairness_scores.get(group, 0.0), "Exposure Share": exposure_share.get(group, 0.0), "Catalog Share": group_weights.get(group, 0.0), "Exposure Count": group_exposures.get(group, 0), "Catalog Count": group_counts.get(group, 0) })

        summary_data.sort(key=lambda x: x['MMF Score'], reverse=True)
        
        header = f"{'Group':<35} | {'MMF Score':<15} | {'Exposure (% / Count)':<28} | {'Catalog (% / Count)':<28}"
        print(f"\n--- MMF Summary for {metric_label} ---")
        print(header)
        print("-" * len(header))
        for item in summary_data:
            exp_share_str = f"{item['Exposure Share']*100:6.2f}% ({item['Exposure Count']})"
            cat_share_str = f"{item['Catalog Share']*100:6.2f}% ({item['Catalog Count']})"
            print(f"{str(item['Group']):<35} | {item['MMF Score']:<15.4f} | {exp_share_str:<28} | {cat_share_str:<28}")
        print("-" * len(header))
        print(f"--- Total Group Exposures Analyzed: {total_exposures} ---\n")

        # --- Report the final MMF score ---
        min_fairness = summary_data[-1]['MMF Score'] if summary_data else 0.0
        mmf_metrics[metric_label] = min_fairness
        print(f"[MMF] -> FINAL RESULT FOR {metric_label}: {min_fairness:.4f}")

    print(f"\n=======================================================================")
    print(f"[MMF DEBUG] MMF computation process finished.")
    print(f"=======================================================================")
    return mmf_metrics