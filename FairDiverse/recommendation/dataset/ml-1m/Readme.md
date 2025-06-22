### Preprocessing info

NOTE: Preprocessing was run with the following hyper_params:

```
hyper_params = {
	"dataset": "ml-1m", 
	"float64": False,

	"depth": 1,
	"grid_search_lamda": True,
	"lamda": 1.0, # Only used if grid_search_lamda == False,
	"use_gini": True,
	"use_mmf": True,
	"item_id": "item_id:token",  # configure it based on the .item file
    "category_id": "genre:token_seq",  # configure it based on the .item file
	"inter_item_col_name": "item_id:token",  # Column name for item IDs in the interaction file
	"use_first_category": True,  # If True, only the first tag in the category sequence is used
	"min_category_support": 10,  # Minimum support for a category to be retained
	# "mmf_reg": 2.0,  # MMF regularization strength
	# "gini_reg": 10.0,  # Gini regularization strength
	# Number of users to keep (randomly)
	"user_support": -1, # -1 implies use all users
	"seed": 42,
	
	"post_process": True # If True, apply post-processing the the model output for FairDiverse reranking 
}
```