import numpy as np
from collections import Counter
import json
import pandas as pd

def compute_mmf(user_recommendations, topk, group_key, group_map, group_weights=None, csv=False):
    """
    Compute weighted Max-Min Fairness (MMF) with debugging and a detailed summary table.
    Calculates MMF for each k in topk and can generate a CSV report.

    Args:
        user_recommendations (dict): A dictionary where keys are integers (k) and values are lists of recommended items.
        topk (list): A list of integers (k) for which to compute the metrics.
        group_key (str): The name of the group being analyzed (e.g., 'artist', 'genre').
        group_map (dict): A dictionary mapping item IDs to their group ID.
        group_weights (dict, optional): Pre-computed weights for each group. If None, weights are calculated from catalog share. Defaults to None.
        csv (bool, optional): If True, generates a 'mmf_stats.csv' report of the MMF statistics. Defaults to False.

    Returns:
        dict: A dictionary containing the final MMF score for each k.
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
    all_k_summary_data = [] # To store data for the final CSV report

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

        summary_data_for_k = []
        for group in group_weights.keys():
            summary_data_for_k.append({
                "Group": group,
                "MMF Score": fairness_scores.get(group, 0.0),
                "Exposure Share": exposure_share.get(group, 0.0),
                "Catalog Share": group_weights.get(group, 0.0),
                "Exposure Count": group_exposures.get(group, 0),
                "Catalog Count": group_counts.get(group, 0)
            })

        summary_data_for_k.sort(key=lambda x: x['MMF Score'], reverse=True)

        if csv:
            for summary_item in summary_data_for_k:
                summary_item['K'] = k
                all_k_summary_data.append(summary_item)

        # Define table layout with updated header
        header = f"{'Group':<35} | {'MMF Score':<15} | {'Exposure (% / Count)':<28} | {'Catalog (% / Count)':<28}"

        # Print the summary table
        print(f"\n--- MMF Summary for {metric_label} ---")
        print(header)
        print("-" * len(header))
        for item in summary_data_for_k:
            # Swapped format to: Percentage% (Count)
            exp_share_str = f"{item['Exposure Share']*100:6.2f}% ({item['Exposure Count']})"
            cat_share_str = f"{item['Catalog Share']*100:6.2f}% ({item['Catalog Count']})"
            print(f"{str(item['Group']):<35} | {item['MMF Score']:<15.4f} | {exp_share_str:<28} | {cat_share_str:<28}")
        print("-" * len(header))
        print(f"--- Total Exposures Analyzed: {total_exposures} ---\n")

        # --- Report the final MMF score ---
        min_fairness = 0.0
        if summary_data_for_k:
            min_fairness = summary_data_for_k[-1]['MMF Score']

        mmf_metrics[metric_label] = min_fairness
        print(f"[MMF] -> FINAL RESULT FOR {metric_label}: {min_fairness:.4f}")

    # --- Generate CSV Report ---
    if csv and all_k_summary_data:
        print("\n[MMF] STEP 3: Generating CSV report 'mmf_stats.csv'...")
        try:
            df = pd.DataFrame(all_k_summary_data)
            df['Exposure Share'] = df['Exposure Share'] * 100
            df['Catalog Share'] = df['Catalog Share'] * 100
            df.rename(columns={
                'Exposure Share': 'Exposure Share (%)',
                'Catalog Share': 'Catalog Share (%)'
            }, inplace=True)


            csv_columns = [
                'K', 'Group', 'MMF Score',
                'Exposure Share (%)', 'Exposure Count',
                'Catalog Share (%)', 'Catalog Count'
            ]
            df = df[csv_columns]
            df.sort_values(by=['K', 'MMF Score'], ascending=[True, True], inplace=True)

            df.to_csv("mmf_stats.csv", index=False, float_format='%.4f')
            print("[MMF] -> SUCCESS: Report 'mmf_stats.csv' created.")

        except Exception as e:
            print(f"[MMF] -> ERROR: Failed to generate CSV report. Reason: {e}")
    elif csv:
        print("\n[MMF] STEP 3: No summary data was generated, skipping CSV report creation.")


    print(f"\n=======================================================================")
    print(f"[MMF DEBUG] MMF computation process finished.")
    print(f"=======================================================================")
    return mmf_metrics


def get_item_group_weights(hyper_params, data):
    """
    Computes a weight for each item based on its group's catalog share in a
    robust way, without assuming any item ID format.
    """
    print("[MMF SETUP] Computing item group weights for MMF regularization...")
    group_map = data.data.get("item_map_to_category")
    item_id_to_idx = data.data.get("item_map")
    # DEBUG
    # print("*"*50)
    # print("[MMF SETUP] group_map:", group_map)
    # print("[MMF SETUP] item_id_to_idx:", item_id_to_idx)
    # print("*"*50)

    if not group_map or not item_id_to_idx:
        print("[MMF SETUP] WARNING: 'item_map_to_category' or 'item_map' not found. Cannot compute MMF weights.")
        return None

    group_counts = Counter(group_map.values())
    total_items_in_catalog = sum(group_counts.values())

    if total_items_in_catalog == 0:
        print("[MMF SETUP] WARNING: Catalog is empty. Cannot compute MMF weights.")
        return None

    group_catalog_share = {group: count / total_items_in_catalog for group, count in group_counts.items()}
    num_items = hyper_params['num_items']
    item_weights = np.zeros(num_items)

    # Iterate through the provided group map directly
    for original_item_id, group in group_map.items():
        lookup_key = np.float64(original_item_id)
        
        # internal, 0-based index for the original item ID
        internal_idx = item_id_to_idx.get(lookup_key)

        # item exists in our dataset and has a valid group
        if internal_idx is not None and group is not None:
            item_weights[internal_idx] = group_catalog_share.get(group, 0.0)

    print(f"[MMF SETUP] Created item weight vector of shape {item_weights.shape}")
    print("[MMF SETUP] Item weights (non-zero):", item_weights[item_weights > 0])
    non_zero_positions = np.nonzero(item_weights)[0]
    print("[MMF SETUP] Non-zero item weights at positions:", non_zero_positions)
    return item_weights