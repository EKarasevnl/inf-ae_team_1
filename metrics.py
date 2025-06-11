import numpy as np
from collections import Counter
import json

def compute_mmf(user_recommendations, topk, group_key, group_map, group_weights=None):
    """
    Compute weighted Max-Min Fairness (MMF) with enhanced debugging.
    Calculates MMF for each k in topk and an overall MMF for all recommendations.
    """
    metric_name = f"MMF-{group_key.upper()}"
    print(f"\n=======================================================================")
    print(f"[MMF DEBUG] INITIALIZING MMF COMPUTATION FOR: {metric_name}")
    print(f"=======================================================================")

    # --- Step 0: DEBUG AND VERIFY INPUTS ---
    print("\n[MMF DEBUG] --- Verifying Input Data Structures ---")
    print(f"[MMF DEBUG] topk values to be analyzed: {topk}")
    if group_map:
        print(f"[MMF DEBUG] group_map: Contains {len(group_map)} item-to-group mappings.")
        sample_group_map = {k: group_map[k] for k in list(group_map)[:5]}
        print(f"[MMF DEBUG]   -> Sample group_map: {json.dumps(sample_group_map, indent=2)}")
    if user_recommendations:
        print(f"[MMF DEBUG] user_recommendations: Contains data for k values: {list(user_recommendations.keys())}")
    print("[MMF DEBUG] --- End of Input Verification ---\n")

    # --- Step 1: Calculate Catalog Share (Weights) ---
    print("[MMF] STEP 1: Calculating Catalog Share (Weight) for each group...")
    if not group_weights:
        if not group_map:
            print("[MMF] CRITICAL: 'group_map' is required. Aborting.")
            return {}
        group_counts = Counter(group_map.values())
        total_items_in_catalog = sum(group_counts.values())
        print(f"[MMF] Found {total_items_in_catalog} total items across {len(group_counts)} groups.")
        if total_items_in_catalog == 0:
            print("[MMF] CRITICAL: 'group_map' is empty. Cannot calculate weights.")
            return {}
        group_weights = {group: count / total_items_in_catalog for group, count in group_counts.items()}
    else:
        print("[MMF] Using pre-defined group weights.")

    mmf_metrics = {}
    # Combine topk and the 'ALL' case for processing
    k_to_process = list(topk) + ['ALL']

    # --- Step 2: Loop through each K and the 'ALL' case ---
    for k in k_to_process:
        is_overall_calc = (k == 'ALL')
        
        if is_overall_calc:
            print(f"\n[MMF] ====================================================")
            print(f"[MMF] STEP 3: Calculating Overall {metric_name} (using all recommendations)")
            print(f"[MMF] ====================================================")
            # Use the recommendation list from the largest k as the 'overall' list
            if not user_recommendations:
                print("[MMF] No recommendations available to calculate overall MMF.")
                continue
            largest_k = max(user_recommendations.keys())
            recommendations_for_k = user_recommendations.get(largest_k, [])
            metric_label = f"{metric_name}@ALL"
        else:
            print(f"\n[MMF] ----------------------------------------------------")
            print(f"[MMF] STEP 2: Calculating {metric_name} for recommendations at k={k}")
            print(f"[MMF] ----------------------------------------------------")
            recommendations_for_k = user_recommendations.get(k, [])
            metric_label = f"{metric_name}@{k}"

        total_exposures = len(recommendations_for_k)
        print(f"[MMF] Found {total_exposures} total recommendations to analyze.")

        if total_exposures == 0:
            mmf_metrics[metric_label] = 0.0
            print(f"[MMF] -> RESULT: {metric_label}: 0.0000 (No recommendations)")
            continue

        # --- Step 2a/3a: Calculate Exposure Share ---
        print("\n[MMF] Sub-step a: Calculating Exposure Share...")
        group_exposures = Counter(group_map.get(item["id"]) for item in recommendations_for_k if "id" in item and group_map.get(item["id"]))
        exposure_share = {group: count / total_exposures for group, count in group_exposures.items()}
        print(f"[MMF] Found {len(exposure_share)} groups represented in this set of recommendations.")

        # --- Step 2b/3b: Calculate Fairness Score ---
        print("\n[MMF] Sub-step b: Calculating Fairness Score (Exposure / Catalog) for each group...")
        fairness_scores = {}
        for group, cat_share in group_weights.items():
            exp_share = exposure_share.get(group, 0.0)
            if cat_share > 0:
                fairness_scores[group] = exp_share / cat_share

        # --- Step 2c/3c: Report the final MMF score ---
        print("\n[MMF] Sub-step c: Identifying the minimum fairness score (MMF)...")
        if not fairness_scores:
            min_fairness = 0.0
            min_group = "N/A"
        else:
            min_group = min(fairness_scores, key=fairness_scores.get)
            min_fairness = fairness_scores[min_group]

        mmf_metrics[metric_label] = min_fairness
        
        if min_group != "N/A":
            print(f"[MMF] -> The most disadvantaged group is '{min_group}' with a score of {min_fairness:.4f}")
        
        print(f"[MMF] -> FINAL RESULT FOR {metric_label}: {min_fairness:.4f}")

    print(f"\n=======================================================================")
    print(f"[MMF DEBUG] MMF computation process finished.")
    print(f"=======================================================================")
    return mmf_metrics