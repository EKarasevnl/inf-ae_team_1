from scipy.sparse import csr_matrix
import jax.numpy as jnp
import numpy as np
import copy
import h5py
import gc
import pandas as pd
import os
from tqdm import tqdm

tqdm.pandas()


class Dataset:
    def __init__(self, hyper_params):
        self.data = load_raw_dataset(
            hyper_params["dataset"],
            hyper_params["item_id"],
            hyper_params["category_id"],
            categories_to_retain=hyper_params.get("categories_to_retain", None),
            inter_item_col_name=hyper_params.get("inter_item_col_name", "product_id:token"),
        )
        self.set_of_active_users = list(set(self.data["train"][:, 0].tolist()))
        self.hyper_params = self.update_hyper_params(hyper_params)

    def update_hyper_params(self, hyper_params):
        updated_params = copy.deepcopy(hyper_params)

        self.num_users, self.num_items = self.data["num_users"], self.data["num_items"]
        self.num_interactions = self.data["num_interactions"]

        # Update hyper-params to have some basic data stats
        updated_params.update(
            {
                "num_users": self.num_users,
                "num_items": self.num_items,
                "num_interactions": self.num_interactions,
            }
        )

        return updated_params

    def sample_users(self, num_to_sample):
        if num_to_sample == -1:
            ret = self.data["train_matrix"]
        else:
            sampled_users = np.random.choice(
                self.set_of_active_users, num_to_sample, replace=False
            )
            sampled_interactions = self.data["train"][
                np.in1d(self.data["train"][:, 0], sampled_users)
            ]
            ret = csr_matrix(
                (
                    np.ones(sampled_interactions.shape[0]),
                    (sampled_interactions[:, 0], sampled_interactions[:, 1]),
                ),
                shape=(self.num_users, self.num_items),
            )

        # This just removes the users which were not sampled
        return jnp.array(ret[ret.getnnz(1) > 0].todense())


def load_raw_dataset(
    dataset,
    item_id,
    category_id,
    data_path=None,
    index_path=None,
    item_path=None,
    categories_to_retain=None, # list of categories to retain
    inter_item_col_name=None, # parameter for the .inter file's item column
):
    if data_path is None or index_path is None:
        data_path, index_path = [
            f"data/{dataset}/total_data.hdf5",
            f"data/{dataset}/index.npz",
        ]
        print(f"Using default paths: data_path={data_path}, index_path={index_path}")

    print(f"Loading data from {data_path}")
    with h5py.File(data_path, "r") as f:
        data = np.array(list(zip(f["user"][:], f["item"][:], f["rating"][:])))
    print(f"Loaded raw data with shape: {data.shape}")

    print(f"Loading index from {index_path}")
    index = np.array(np.load(index_path)["data"], dtype=np.int32)
    print(f"Loaded index with shape: {index.shape}")
    
    # Load item data first to prepare for filtering
    print("Loading item data")
    if item_path is None:
        item_path = f"data/{dataset}/{dataset}.item"
        print(f"Using default item_path: {item_path}")

    print(f"Reading item data from {item_path}")
    try:
        item_df = pd.read_csv(
            item_path, delimiter="\t", header=0, engine="python", encoding="latin-1"
        )
    except pd.errors.ParserError as e:
        print(f"Parser error with standard settings: {e}")
        print("Trying with on_bad_lines='warn' and encoding='utf-8'")
        item_df = pd.read_csv(
            item_path,
            delimiter="\t",
            header=0,
            engine="python",
            encoding="utf-8",
            on_bad_lines="warn",
        )
    print(f"Loaded item data with shape: {item_df.shape}")
    print(f"Item ID column: {item_id}")

    if category_id not in item_df.columns:
        print(f"Category ID '{category_id}' not found in item data. Creating dummy categories.")
        item_df[category_id] = np.random.choice(
            ["DummyA", "DummyB", "DummyC"], size=item_df.shape[0]
        )
    else:
        print(f"Category ID '{category_id}' found in item data")

    item_df = item_df[item_df[category_id].notna()]
    print(f"Filtered item data to {item_df.shape[0]} rows with non-NaN categories")
    item_df = item_df[item_df[item_id].notna()]
    print(f"Filtered item data to {item_df.shape[0]} rows with non-NaN item IDs")
    
    if categories_to_retain and isinstance(categories_to_retain, list) and category_id in item_df.columns:
        print(f"\n[FILTERING] Applying filter to retain {len(categories_to_retain)} explicitly defined categories from column '{category_id}'.")
        
        categories_to_filter_by = set(categories_to_retain)
        print(f"[FILTERING] Categories to retain: {list(categories_to_filter_by)}")

        original_rows = len(item_df)
        item_df = item_df[item_df[category_id].isin(categories_to_filter_by)].copy()
        print(f"[FILTERING] Item catalog filtered from {original_rows} to {len(item_df)} rows.")

        if item_path:
            base, ext = os.path.splitext(item_path)
            filtered_path = f"{base}_filtered{ext}"
            try:
                print(f"[FILTERING] Saving filtered item catalog to: {filtered_path}")
                item_df.to_csv(filtered_path, sep='\t', index=False, encoding='latin-1')
            except Exception as e:
                print(f"[FILTERING] Error saving filtered file: {e}")
        
        # Filter the .inter file using the configurable column name
        print("\n[INTERACTION FILE] Filtering the .inter file based on the filtered item catalog.")
        inter_path = f"data/{dataset}/{dataset}.inter"
        if not os.path.exists(inter_path):
            print(f"[INTERACTION FILE] Warning: {inter_path} not found. Skipping .inter file filtering.")
        # Check if the configurable column name was provided
        elif not inter_item_col_name:
            print(f"[INTERACTION FILE] Warning: 'inter_item_col_name' not configured. Skipping .inter file filtering.")
        else:
            try:
                print(f"[INTERACTION FILE] Loading raw interactions from {inter_path}")
                inter_df = pd.read_csv(inter_path, delimiter='\t', header=0, engine='python', encoding='latin-1')
                
                retained_item_ids_for_inter = set(item_df[item_id].tolist())
                
                original_inter_rows = len(inter_df)
                # Use the configurable column name for filtering
                filtered_inter_df = inter_df[inter_df[inter_item_col_name].isin(retained_item_ids_for_inter)].copy()
                print(f"[INTERACTION FILE] Filtered .inter file from {original_inter_rows} to {len(filtered_inter_df)} rows.")
                
                # --- Remap user_id to be contiguous ---
                user_col_name = 'user_id:token'
                print(f"[INTERACTION FILE] Remapping '{user_col_name}' to be contiguous.")
                
                # Create a mapping from old user IDs to new 0-indexed IDs
                unique_users = filtered_inter_df[user_col_name].unique()
                user_remapping_dict = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
                
                # Apply the mapping
                filtered_inter_df[user_col_name] = filtered_inter_df[user_col_name].map(user_remapping_dict)
                print(f"[INTERACTION FILE] Remapped {len(unique_users)} unique users.")
                # --- END NEW LOGIC ---

                # Format item ID column and save
                filtered_inter_df[inter_item_col_name] = filtered_inter_df[inter_item_col_name].astype('Int64')

                filtered_inter_path = f"data/{dataset}/{dataset}_filtered.inter"
                print(f"[INTERACTION FILE] Saving filtered interactions to {filtered_inter_path}")
                filtered_inter_df.to_csv(filtered_inter_path, sep='\t', index=False, encoding='latin-1')

            except KeyError as e:
                print(f"[INTERACTION FILE] An error occurred: Column '{e}' not found in {inter_path}.")
            except Exception as e:
                print(f"[INTERACTION FILE] An error occurred during .inter file processing: {e}")
        
        print("[FILTERING] Item catalog and interaction file filtering complete.\n")

    retained_item_ids = set(item_df[item_id].astype(int).tolist())
    if len(retained_item_ids) < item_df[item_id].nunique():
        print(f"[HDF5 FILTER] Using the {len(retained_item_ids)} items from the filtered catalog to filter HDF5 interactions.")
        original_interaction_count = data.shape[0]
        
        interaction_mask = np.isin(data[:, 1], list(retained_item_ids))
        
        data = data[interaction_mask]
        index = index[interaction_mask]
        
        print(f"[HDF5 FILTER] HDF5 interaction data filtered from {original_interaction_count} to {data.shape[0]} records.")

    def remap(data, index):
        print("Remapping user and item IDs based on filtered interactions")
        valid_users, valid_items = set(), set()

        print("Identifying valid users and items from the filtered set")
        for at, (u, i, r) in enumerate(tqdm(data, desc="Scanning for valid entries")):
            if index[at] != -1:
                valid_users.add(u)
                valid_items.add(i)

        print(
            f"Found {len(valid_users)} valid users and {len(valid_items)} valid items post-filtering."
        )

        user_map = dict(zip(list(valid_users), list(range(len(valid_users)))))
        valid_items_int = [int(i) for i in valid_items]
        item_map = dict(zip(valid_items_int, list(range(len(valid_items)))))
        print("User and item mapping created")

        return user_map, item_map

    user_map, item_map = remap(data, index)

    print("Creating new data and index arrays with mapped IDs")
    new_data, new_index = [], []
    for at, (u, i, r) in enumerate(tqdm(data, desc="Remapping data")):
        if index[at] == -1:
            continue
        if u in user_map and i in item_map:
            new_data.append([user_map[u], item_map[i], r])
            new_index.append(index[at])

    data = np.array(new_data, dtype=np.int32)
    index = np.array(new_index, dtype=np.int32)
    print(f"Remapped data shape: {data.shape}, index shape: {index.shape}")

    print(f"Category ID column: {category_id}")
    print(f"Number of unique categories in filtered catalog: {item_df[category_id].nunique()}")

    all_genres = [
        genre
        for genre_list in item_df[category_id].fillna("[Nan]")
        for genre in genre_list.strip("[]").split(", ")
    ]
    unique_genres_list = list(set(all_genres))
    item_map_to_category = dict(
        zip(item_df[item_id].astype(int), item_df[category_id])
    )

    def select(data, index, index_val):
        print(f"Selecting data with index value {index_val}")
        selected_indices = np.where(index == index_val)[0]
        print(f"Found {len(selected_indices)} entries with index value {index_val}")
        final = data[selected_indices]
        final[:, 2] = 1.0
        return final.astype(np.int32)

    print("Creating train/val/test splits")
    ret = {
        "item_map": item_map,
        "train": select(data, index, 0),
        "val": select(data, index, 1),
        "test": select(data, index, 2),
        "item_map_to_category": item_map_to_category,
    }
    print(
        f"Split sizes - Train: {len(ret['train'])}, Val: {len(ret['val'])}, Test: {len(ret['test'])}"
    )

    num_users = len(user_map)
    num_items = len(item_map)
    print(f"Dataset has {num_users} users and {num_items} items after filtering.")

    print("Cleaning up memory")
    del data, index
    gc.collect()

    def make_user_history(arr):
        print(f"Creating user history from array with shape {arr.shape}")
        ret = [set() for _ in range(num_users)]
        for u, i, r in tqdm(arr, desc="Building user history"):
            if u >= num_users or i >= num_items:
                continue
            ret[int(u)].add(int(i))

        # Log some statistics about the history
        history_sizes = [len(h) for h in ret]
        print(
            f"User history stats - Min: {min(history_sizes)}, Max: {max(history_sizes)}, "
            f"Avg: {sum(history_sizes)/len(history_sizes):.2f}"
        )
        return ret

    print("Creating positive sets for train/val/test")
    ret["train_positive_set"] = make_user_history(ret["train"])
    ret["val_positive_set"] = make_user_history(ret["val"])
    ret["test_positive_set"] = make_user_history(ret["test"])

    print("Creating sparse matrices")
    ret["train_matrix"] = csr_matrix(
        (
            np.ones(ret["train"].shape[0]),
            (ret["train"][:, 0].astype(np.int32), ret["train"][:, 1].astype(np.int32)),
        ),
        shape=(num_users, num_items),
    )
    print(
        f"Created train matrix with shape {ret['train_matrix'].shape} and {ret['train_matrix'].nnz} non-zeros"
    )

    ret["val_matrix"] = csr_matrix(
        (
            np.ones(ret["val"].shape[0]),
            (ret["val"][:, 0].astype(np.int32), ret["val"][:, 1].astype(np.int32)),
        ),
        shape=(num_users, num_items),
    )
    print(
        f"Created val matrix with shape {ret['val_matrix'].shape} and {ret['val_matrix'].nnz} non-zeros"
    )

    # Negatives will be used for AUC computation
    print("Generating negative samples for evaluation")
    ret["negatives"] = [set() for _ in range(num_users)]

    for u in tqdm(range(num_users), desc="Generating negatives"):
        attempts = 0
        while len(ret["negatives"][u]) < 50:
            attempts += 1
            if attempts > 1000:  # Safety check to avoid infinite loops
                logger.warning(
                    f"User {u} could not get 50 negatives after 1000 attempts"
                )
                break

            rand_item = np.random.randint(0, num_items)
            if rand_item in ret["train_positive_set"][u]:
                continue
            if rand_item in ret["test_positive_set"][u]:
                continue
            ret["negatives"][u].add(rand_item)
        ret["negatives"][u] = list(ret["negatives"][u])

    ret["negatives"] = np.array(ret["negatives"], dtype=np.int32)
    print(f"Generated negative samples with shape {ret['negatives'].shape}")

    ret.update(
        {
            "num_users": num_users,
            "num_items": num_items,
            "num_interactions": len(ret["train"]),
        }
    )

    # Log summary statistics
    print("Dataset loading complete. Summary:")
    print("# users:", num_users)
    print("# items:", num_items)
    print("# interactions:", len(ret["train"]))
    print("# unique genres:", len(unique_genres_list))

    return ret


if __name__ == "__main__":
    hyper_params = {
        "dataset": "steam", 
        "item_id": "id:token",                   # Item ID column in the .item file
        "inter_item_col_name": "product_id:token", # Item ID column in the .inter file
        "category_id": "developer:token",
        "categories_to_retain": [
            "Strategy First",
            "Malfador Machinations",
            "Sonalysts",
            "Introversion Software",
            "Outerlight Ltd.",
            "Darklight Games",
            "RavenSoft / id Software",
            "id Software",
            "Ritual Entertainment",
            "Valve",
        ]
    }

    data = Dataset(hyper_params)
    print("\nDataset object created successfully.")
    print(f"Number of users in final dataset: {data.hyper_params['num_users']}")
    print(f"Number of items in final dataset: {data.hyper_params['num_items']}")