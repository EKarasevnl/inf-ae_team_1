import numpy as np
from collections import Counter
import json

def compute_mmf(user_recommendations, topk, group_key, group_map, group_weights=None):
    """
    Compute weighted Max-Min Fairness (MMF) with debugging and a detailed summary table.
    Calculates MMF for each k in topk.
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
    if user_recommendations:
        print(f"[MMF DEBUG] user_recommendations: Contains data for k values: {list(user_recommendations.keys())}")
    print("[MMF DEBUG] --- End of Input Verification ---\n")

    # --- Step 1: Calculate Catalog Share (Weights) and Counts ---
    print("[MMF] STEP 1: Calculating Catalog Share (Weight) for each group...")
    if not group_map:
        print("[MMF] CRITICAL: 'group_map' is required for MMF calculation and summary. Aborting.")
        return {}

    group_counts = Counter(group_map.values())
    total_items_in_catalog = sum(group_counts.values())
    print(f"[MMF] Found {total_items_in_catalog} total items across {len(group_counts)} groups from group_map.")
    
    if not group_weights:
        if total_items_in_catalog == 0:
            print("[MMF] CRITICAL: 'group_map' is empty. Cannot calculate weights.")
            return {}
        group_weights = {group: count / total_items_in_catalog for group, count in group_counts.items()}
        print("[MMF] Calculated group weights based on group_map.")
    else:
        print("[MMF] Using pre-defined group weights. Catalog counts for summary are still derived from group_map.")

    mmf_metrics = {}
    k_to_process = list(topk)

    # --- Step 2: Loop through each K ---
    for k in k_to_process:
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

        # --- Sub-steps for calculation ---
        group_exposures = Counter(group_map.get(item["id"]) for item in recommendations_for_k if "id" in item and group_map.get(item["id"]))
        exposure_share = {group: count / total_exposures for group, count in group_exposures.items()}
        
        fairness_scores = {}
        for group, cat_share in group_weights.items():
            exp_share = exposure_share.get(group, 0.0)
            if cat_share > 0:
                fairness_scores[group] = exp_share / cat_share
            else:
                fairness_scores[group] = 0.0

        # --- Create and print detailed summary ---
        print("\n[MMF] Sub-step c: Generating MMF summary...")

        summary_data = []
        for group in group_weights.keys():
            summary_data.append({
                "Group": group,
                "MMF Score": fairness_scores.get(group, 0.0),
                "Exposure Share": exposure_share.get(group, 0.0),
                "Catalog Share": group_weights.get(group, 0.0),
                "Exposure Count": group_exposures.get(group, 0),
                "Catalog Count": group_counts.get(group, 0)
            })

        summary_data.sort(key=lambda x: x['MMF Score'], reverse=True)

        # Define table layout with updated header
        header = f"{'Group':<35} | {'MMF Score':<15} | {'Exposure (% / Count)':<28} | {'Catalog (% / Count)':<28}"
        
        # Print the summary table
        print(f"\n--- MMF Summary for {metric_label} ---")
        print(header)
        print("-" * len(header))
        for item in summary_data:
            # Swapped format to: Percentage% (Count)
            exp_share_str = f"{item['Exposure Share']*100:6.2f}% ({item['Exposure Count']})"
            cat_share_str = f"{item['Catalog Share']*100:6.2f}% ({item['Catalog Count']})"
            print(f"{str(item['Group']):<35} | {item['MMF Score']:<15.4f} | {exp_share_str:<28} | {cat_share_str:<28}")
        print("-" * len(header))
        print(f"--- Total Exposures Analyzed: {total_exposures} ---\n")

        # --- Report the final MMF score ---
        min_fairness = 0.0
        if summary_data:
            min_fairness = summary_data[-1]['MMF Score']
        
        mmf_metrics[metric_label] = min_fairness
        print(f"[MMF] -> FINAL RESULT FOR {metric_label}: {min_fairness:.4f}")

    print(f"\n=======================================================================")
    print(f"[MMF DEBUG] MMF computation process finished.")
    print(f"=======================================================================")
    return mmf_metrics