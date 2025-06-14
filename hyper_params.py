hyper_params = {
	'dataset': 'ml-10m', 
	'float64': False,

	'depth': 1,
	'grid_search_lamda': True,
	'lamda': 1.0, # Only used if grid_search_lamda == False,
	'use_gini': True,
	'use_mmf': True,  # Set to True if you want to use MMF (Max-min Fairness)
	"item_id": "item_id:token",  # configure it based on the .item file
    "category_id": "type:token_seq",  # configure it based on the .item file


	# Number of users to keep (randomly)
	'user_support': -1, # -1 implies use all users
	'seed': 42,
}

# hyper_params = {
# 	'dataset': 'steam', 
# 	'float64': False,

# 	'depth': 1,
# 	'grid_search_lamda': True,
# 	'lamda': 1.0, # Only used if grid_search_lamda == False,
# 	'use_gini': True,
# 	'use_mmf': True,  # Set to True if you want to use MMF (Max-min Fairness)
# 	"item_id": "id:token",  # configure it based on the .item file
#     "category_id": "genres:token_seq",  # configure it based on the .item file


# 	# Number of users to keep (randomly)
# 	'user_support': -1, # -1 implies use all users
# 	'seed': 42,
# }
