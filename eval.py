import jax
import numpy as np
import jax.numpy as jnp
from numba import jit, float64
from scipy.sparse import csr_matrix, save_npz

import numpy as np
import os
import json
import yaml

from hyper_params import hyper_params
from metrics import compute_mmf

USE_GINI = hyper_params.get("use_gini", False)
USE_MMF = hyper_params.get("use_mmf", False)
POST_PROCESS = hyper_params.get("post_process", False)


class GiniCoefficient:
    """
    A class to calculate the Gini coefficient, a measure of income inequality.
    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    """

    def gini_coefficient(self, values):
        """
        Compute the Gini coefficient of array of values.
        For a frequency vector, G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mu)
        """
        print(f"[GINI] Computing Gini coefficient for {len(values)} values")
        arr = np.array(values, dtype=float)
        if arr.sum() == 0:
            print("[GINI] Sum of values is 0, returning 0.0")
            return 0.0
        # sort and normalize
        arr = np.sort(arr)
        n = arr.size
        cumvals = np.cumsum(arr)
        mu = arr.mean()
        # the formula simplifies to:
        # G = (1 / (n * mu)) * ( sum_i (2*i - n - 1) * arr[i] )
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
        print(f"[GINI] Computed Gini coefficient: {gini:.4f}")
        return gini

    def calculate_list_gini(self, articles, key="category"):
        """
        Given a list of article dicts and a key (e.g. 'category'), compute the
        Gini coefficient over the frequency distribution of that key.
        """
        print(f"[GINI] Calculating Gini for {len(articles)} articles using key '{key}'")
        # count frequencies
        freqs = {}
        for art in articles:
            val = art.get(key, None) or "UNKNOWN"
            freqs[val] = freqs.get(val, 0) + 1
        print(f"[GINI] Found {len(freqs)} unique {key} values")
        return self.gini_coefficient(list(freqs.values()))


INF = float(1e6)


def evaluate(
    hyper_params,
    kernelized_rr_forward,
    data,
    item_propensity,
    train_x,
    topk=[10, 100], #, 1000],
    test_set_eval=False,
    item_group_weights=None,
):
    print(
        f"\n[EVALUATE] Starting evaluation with topk={topk}, test_set_eval={test_set_eval}"
    )
    print(
        f"[EVALUATE] Hyperparameters: num_users={hyper_params['num_users']}, num_items={hyper_params['num_items']}, lambda={hyper_params['lamda']}"
    )

    preds, y_binary, metrics = [], [], {}
    for kind in ["HR", "NDCG", "PSP", "GINI"]:
        for k in topk:
            metrics["{}@{}".format(kind, k)] = 0.0

    # Train positive set -- these items will be set to -infinity while prediction on the val/test set
    train_positive_list = list(map(list, data.data["train_positive_set"]))
    print(f"[EVALUATE] Train positive set size: {len(train_positive_list)}")

    if test_set_eval:
        print(
            "[EVALUATE] Adding validation positive set to train positive set for test evaluation"
        )
        for u in range(len(train_positive_list)):
            train_positive_list[u] += list(data.data["val_positive_set"][u])

    # Train positive interactions (in matrix form) as context for prediction on val/test set
    eval_context = data.data["train_matrix"]
    if test_set_eval:
        print("[EVALUATE] Adding validation matrix to evaluation context")
        eval_context += data.data["val_matrix"]

    # What needs to be predicted
    to_predict = data.data["val_positive_set"]
    if test_set_eval:
        print("[EVALUATE] Using test positive set for prediction")
        to_predict = data.data["test_positive_set"]
    print(f"[EVALUATE] Prediction set size: {len(to_predict)}")

    # For GINI calculation - track item exposures across all recommendations
    item_exposures = np.zeros(hyper_params["num_items"])

    user_recommendations = {}

    # --- NEW: Create a placeholder for the full scores matrix ---
    if POST_PROCESS:
        full_ranking_scores = np.zeros((hyper_params["num_users"], hyper_params["num_items"]), dtype=np.float32)

    bsz = 140_000  # These many users
    print(f"[EVALUATE] Processing users in batches of {bsz}")

    for i in range(0, hyper_params["num_users"], bsz):
        batch_end = min(i + bsz, hyper_params["num_users"])
        print(
            f"[EVALUATE] Processing batch of users {i} to {batch_end-1} (total: {batch_end-i})"
        )

        print(f"[EVALUATE] Running forward pass for batch {i} to {batch_end-1}")
        temp_preds = kernelized_rr_forward(
            train_x,
            eval_context[i:batch_end].todense(),
            reg=hyper_params["lamda"],
            gini_reg=hyper_params.get("gini_reg", 0.0), # GINI regularization
            mmf_reg=hyper_params.get("mmf_reg", 0.0),   # MMF regularization
            item_group_weights=item_group_weights       # MMF group weights
        )
        print(
            f"[EVALUATE] Forward pass complete, prediction shape: {np.array(temp_preds).shape}"
        )

        # --- NEW: Store the batch predictions in the full matrix ---
        if POST_PROCESS:
            full_ranking_scores[i:batch_end] = np.array(temp_preds)

        print(f"[EVALUATE] Evaluating batch {i} to {batch_end-1}")
        metrics, temp_preds, temp_y, user_recommendations_batch = evaluate_batch(
            data.data["negatives"][i:batch_end],
            np.array(temp_preds),
            train_positive_list[i:batch_end],
            to_predict[i:batch_end],
            item_propensity,
            topk,
            metrics,
            data,
        )
        print(f"[EVALUATE] Batch evaluation complete")

        if USE_GINI:
            # Accumulate item exposures for GINI calculation
            for k in topk:
                if k not in user_recommendations:
                    user_recommendations[k] = []
                user_recommendations[k] += user_recommendations_batch[k]
                print(
                    f"[EVALUATE] Accumulated {len(user_recommendations_batch[k])} recommendations for k={k}"
                )

        preds += temp_preds
        y_binary += temp_y
        print(
            f"[EVALUATE] Accumulated {len(temp_preds)} predictions and {len(temp_y)} labels"
        )

    print(f"[EVALUATE] All batches processed, computing final metrics")
    y_binary, preds = np.array(y_binary), np.array(preds)
    if (True not in np.isnan(y_binary)) and (True not in np.isnan(preds)):
        metrics["AUC"] = round(fast_auc(y_binary, preds), 4)
        print(f"[EVALUATE] Computed AUC: {metrics['AUC']}")
    else:
        print(
            "[EVALUATE] Warning: NaN values detected in y_binary or preds, skipping AUC calculation"
        )
        # count how many NaN values are in y_binary and preds
        print(
            f"[EVALUATE] NaN count in y_binary: {np.isnan(y_binary).sum()}, preds: {np.isnan(preds).sum()}"
        )

    for kind in ["HR", "NDCG", "PSP"]:
        for k in topk:
            metrics["{}@{}".format(kind, k)] = round(
                float(100.0 * metrics["{}@{}".format(kind, k)])
                / hyper_params["num_users"],
                4,
            )
            print(f"[EVALUATE] {kind}@{k}: {metrics['{}@{}'.format(kind, k)]}")

    if USE_GINI:
        print("[EVALUATE] Computing GINI coefficients")
        for k in topk:
            print(
                f"[EVALUATE] Computing GINI@{k} with {len(user_recommendations[k])} recommendations"
            )
            metrics["GINI@{}".format(k)] = GiniCoefficient().calculate_list_gini(
                user_recommendations[k], key="category"
            )
            print(f"[EVALUATE] GINI@{k}: {metrics['GINI@{}'.format(k)]}")
    #### MMF Metrics ####
    if USE_MMF:
        item_map_to_category = data.data.get("item_map_to_category")
        print("[EVALUATE] Computing MMF metrics for categories.")
        if not user_recommendations:
            print("[EVALUATE] SKIPPING MMF: Full recommendation lists were not collected.")
            print("[EVALUATE] HINT: To compute MMF, 'USE_GINI' must be True so that recommendations are collected.")
        else:
            mmf_metrics = compute_mmf(
                user_recommendations=user_recommendations,
                topk=topk,
                group_key='category',
                group_map=item_map_to_category
            )
            metrics.update(mmf_metrics)
    ## End of MMF Metrics ####

    metrics["num_users"] = int(train_x.shape[0])
    metrics["num_interactions"] = int(jnp.count_nonzero(train_x.astype(np.int8)))
    print(
        f"[EVALUATE] Final metrics: num_users={metrics['num_users']}, num_interactions={metrics['num_interactions']}"
    )


    # POST PROCESSING FOR FAIR DIVERSE (make it compatible with FairDiverse)
    if POST_PROCESS:
        print("\n[POST-PROCESS] Starting post-processing and file generation...")
        dataset_name = hyper_params.get("dataset", "unknown_dataset")

        # Step 1: Create directories
        log_dir = f"FairDiverse/fairdiverse/recommendation/log/{dataset_name}"
        data_dir = f"FairDiverse/fairdiverse/recommendation/processed_dataset/{dataset_name}"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        print(f"[POST-PROCESS] Created directories: {log_dir} and {data_dir}")
        print("[POST-PROCESS] Establishing canonical item count and mappings...")
        item_map_from_data = data.data.get("item_map_to_category", {})
        if item_map_from_data:
            sample_key = next(iter(item_map_from_data.keys()))
            print(f"[POST-PROCESS] Diagnostic: A sample key from item_map_to_category is of type '{type(sample_key)}' (e.g., {sample_key})")
        else:
            print("[POST-PROCESS] WARNING: The 'item_map_to_category' dictionary is empty or missing.")


        num_model_items = hyper_params["num_items"]
        if num_model_items != full_ranking_scores.shape[1]:
            print(f"[POST-PROCESS] CRITICAL WARNING: Mismatch between hyper_params['num_items'] ({num_model_items}) "
                  f"and score matrix width ({full_ranking_scores.shape[1]}). Using matrix width as final authority.")
            num_model_items = full_ranking_scores.shape[1]
        
        print(f"[POST-PROCESS] Authoritative item count set to: {num_model_items}")

        original_to_new_dense_id_map = {i + 1: i for i in range(num_model_items)}
        item_remapping_path = os.path.join(data_dir, "item_id_remapping.json")
        with open(item_remapping_path, 'w') as f:
            json.dump(original_to_new_dense_id_map, f, indent=2)
        print(f"[POST-PROCESS] Saved original-to-new item ID map to: {item_remapping_path}")
        print("[POST-PROCESS] Saving ranking scores matrix...")
        ranking_scores_path = os.path.join(log_dir, "ranking_scores.npz")
        np.savez_compressed(ranking_scores_path, scores=full_ranking_scores)
        print(f"[POST-PROCESS] Saved ranking scores matrix (shape: {full_ranking_scores.shape}) to: {ranking_scores_path}")
        print("[POST-PROCESS] Preparing and remapping category mapping files...")

        sample_value = next(iter(item_map_from_data.values()), None)
        category_name_to_id_map = None
        if isinstance(sample_value, str):
            print("[POST-PROCESS] Found string categories. Creating new integer category IDs.")
            all_categories = sorted(list(set(item_map_from_data.values())))
            category_name_to_id_map = {name: i for i, name in enumerate(all_categories)}
            category_mapping_path = os.path.join(data_dir, "category_id_mapping.json")
            with open(category_mapping_path, 'w') as f: json.dump(category_name_to_id_map, f, indent=2)
            print(f"[POST-PROCESS] Saved category name-to-ID map to: {category_mapping_path}")

        final_remapped_iid2pid = {}
        for dense_id in range(num_model_items):
            original_id = dense_id + 1
            category_value = item_map_from_data.get(original_id)
            if category_value is None:
                category_value = item_map_from_data.get(str(original_id))

            final_category_id = -1 
            if category_value is not None:
                if category_name_to_id_map:
                    final_category_id = category_name_to_id_map.get(category_value, -1)
                elif isinstance(category_value, (int, float)):
                    final_category_id = int(category_value)

            final_remapped_iid2pid[str(dense_id)] = final_category_id

        iid2pid_path = os.path.join(data_dir, "iid2pid.json")
        with open(iid2pid_path, 'w') as f: json.dump(final_remapped_iid2pid, f)
        print(f"[POST-PROCESS] Saved final remapped item-to-group-ID map to: {iid2pid_path}")

        # --- Create Config Files ---
        print("[POST-PROCESS] Creating final configuration files...")
        num_users = hyper_params["num_users"]
        num_items = num_model_items 
        num_groups = len(set(val for val in final_remapped_iid2pid.values() if val != -1))
        
        # Add a check for num_groups if all items were -1
        if num_groups == 0 and final_remapped_iid2pid:
            print("[POST-PROCESS] WARNING: No valid group IDs found for any item. group_num will be 0.")

        config_data = { 'user_num': num_users, 'item_num': num_items, 'group_num': num_groups }

        process_config_path = os.path.join(data_dir, "process_config.yaml")
        with open(process_config_path, "w") as file:
            yaml.dump(config_data, file, sort_keys=False)
        print(f"[POST-PROCESS] Saved data processing config to: {process_config_path}")

        postprocessing_model_name = "CPFair"
        config_model = {
            "ranking_store_path": f"{dataset_name}", "model": f"{postprocessing_model_name}",
            "fair-rank": True, "log_name": f"{postprocessing_model_name}_without_fairdiverse_{dataset_name}",
            "topk": [5, 10, 20], "fairness_metrics": ["MinMaxRatio", "MMF", "GINI", "Entropy"],
            "fairness_type": "Exposure"
        }
        model_config_path = f"FairDiverse/fairdiverse/recommendation/postprocessing_without_fairdiverse.yaml"
        with open(model_config_path, "w") as file:
            yaml.dump(config_model, file, sort_keys=False)
        print(f"[POST-PROCESS] Saved post-processing model config to: {model_config_path}")
        print("[POST-PROCESS] Post-processing complete.")

    return metrics


def evaluate_batch(
    auc_negatives,
    logits,
    train_positive,
    test_positive_set,
    item_propensity,
    topk,
    metrics,
    data,
    train_metrics=False,
):
    print(f"[EVAL_BATCH] Starting batch evaluation with {len(logits)} users")

    # AUC Stuff
    temp_preds, temp_y = [], []
    for b in range(len(logits)):
        pos_count = len(test_positive_set[b])
        neg_count = len(auc_negatives[b])

        if pos_count == 0 or neg_count == 0:
            continue

        if b % 1000 == 0:  # Only print every 1000 users to avoid excessive output
            print(
                f"[EVAL_BATCH] User {b}: processing {pos_count} positive and {neg_count} negative examples"
            )

        temp_preds += np.take(logits[b], np.array(list(test_positive_set[b]))).tolist()
        temp_y += [1.0 for _ in range(pos_count)]

        temp_preds += np.take(logits[b], auc_negatives[b]).tolist()
        temp_y += [0.0 for _ in range(neg_count)]

    print(f"[EVAL_BATCH] Collected {len(temp_preds)} predictions for AUC calculation")

    # Marking train-set consumed items as negative INF
    print(f"[EVAL_BATCH] Marking train-set consumed items as negative infinity")
    for b in range(len(logits)):
        if b % 1000 == 0:  # Only print every 1000 users to avoid excessive output
            print(
                f"[EVAL_BATCH] User {b}: marking {len(train_positive[b])} train positive items as -INF"
            )
        logits[b][train_positive[b]] = -INF

    print(f"[EVAL_BATCH] Sorting indices for top-{max(topk)} recommendations")
    indices = (-logits).argsort()[:, : max(topk)].tolist()
    batch_exposures = {k: np.zeros(logits.shape[1]) for k in topk}

    user_recommendations = {}

    for k in topk:
        print(f"[EVAL_BATCH] Computing metrics for k={k}")
        user_recommendations[k] = []
        hr_sum, ndcg_sum, psp_sum = 0, 0, 0

        for b in range(len(logits)):
            
            if USE_GINI or USE_MMF or POST_PROCESS:
                # Update item exposures for this batch at this k
                for item_idx in indices[b][:k]:

                    try:
                        
                        user_recommendations[k].append(
                            {
                                "id": item_idx + 1,
                                "category": data.data["item_map_to_category"][item_idx + 1],
                            }
                        )
                    except:
                        pass

            num_pos = float(len(test_positive_set[b]))
            if num_pos == 0:
                continue
            hits = len(set(indices[b][:k]) & test_positive_set[b])

            if b % 1000 == 0:  # Only print every 1000 users to avoid excessive output
                print(
                    f"[EVAL_BATCH] User {b}, k={k}: {hits} hits out of {min(num_pos, k)} possible"
                )

            hr = float(hits) / float(min(num_pos, k))
            hr_sum += hr
            metrics["HR@{}".format(k)] += hr

            test_positive_sorted_psp = sorted(
                [item_propensity[x] for x in test_positive_set[b]]
            )[::-1]

            dcg, idcg, psp, max_psp = 0.0, 0.0, 0.0, 0.0
            for at, pred in enumerate(indices[b][:k]):
                if pred in test_positive_set[b]:
                    dcg += 1.0 / np.log2(at + 2)
                    psp += float(item_propensity[pred]) / float(min(num_pos, k))
                if at < num_pos:
                    idcg += 1.0 / np.log2(at + 2)
                    max_psp += test_positive_sorted_psp[at]

            ndcg = dcg / idcg if idcg > 0 else 0
            psp_norm = psp / max_psp if max_psp > 0 else 0

            ndcg_sum += ndcg
            psp_sum += psp_norm

            metrics["NDCG@{}".format(k)] += ndcg
            metrics["PSP@{}".format(k)] += psp_norm

        print(
            f"[EVAL_BATCH] k={k} metrics - Average HR: {hr_sum/len(logits):.4f}, Average NDCG: {ndcg_sum/len(logits):.4f}, Average PSP: {psp_sum/len(logits):.4f}"
        )
        if USE_GINI:
            print(
                f"[EVAL_BATCH] Collected {len(user_recommendations[k])} recommendations for k={k}"
            )
        
    print(
        f"[EVAL_BATCH] Batch evaluation complete, returning {len(temp_preds)} predictions"
    )
    # return metrics, temp_preds, temp_y, user_recommendations if USE_GINI else {}
    return metrics, temp_preds, temp_y, user_recommendations if (USE_GINI or USE_MMF or POST_PROCESS) else {}


@jit(float64(float64[:], float64[:]))
def fast_auc(y_true, y_prob):
    # Note: Can't add prints here because this function is JIT-compiled
    y_true = y_true[np.argsort(y_prob)]
    nfalse, auc = 0, 0
    for i in range(len(y_true)):
        nfalse += 1 - y_true[i]
        auc += y_true[i] * nfalse
    return auc / (nfalse * (len(y_true) - nfalse))