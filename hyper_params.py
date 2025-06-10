hyper_params = {
	'dataset': 'ml-10m', 
	'float64': False,

	'depth': 1,
	'grid_search_lamda': True,
	'lamda': 1.0, # Only used if grid_search_lamda == False,
	'use_gini': True,
	"item_id": "item_id:token",  # configure it based on the .item file
    "category_id": "genre:token_seq",  # configure it based on the .item file


	# Number of users to keep (randomly)
	'user_support': -1, # -1 implies use all users
	'seed': 42,
}
