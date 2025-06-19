# hyper_params = {
# 	"dataset": "ml-1m", 
# 	"float64": False,

# 	"depth": 1,
# 	"grid_search_lamda": True,
# 	"lamda": 1.0, # Only used if grid_search_lamda == False,
# 	"use_gini": True,
# 	"item_id": "item_id:token",  # configure it based on the .item file
#     "category_id": "genre:token_seq",  # configure it based on the .item file


# 	# Number of users to keep (randomly)
# 	"user_support": -1, # -1 implies use all users
# 	"seed": 42,
# }

hyper_params = {
	"dataset": "steam", 
	"float64": False,

	"depth": 1,
	"grid_search_lamda": True,
	"lamda": 1.0, # Only used if grid_search_lamda == False,
	"use_gini": True,
	"use_mmf": True,
	"item_id": "id:token",  # configure it based on the .item file
    "category_id": "developer:token",  # configure it based on the .item file
	"categories_to_retain": [
		"Strategy First",
		"Malfador Machinations",
		"Sonalysts ",
		"Introversion Software",
		"Outerlight Ltd.",
		"Darklight Games",
		"RavenSoft / id Software",
		"id Software",
		"Ritual Entertainment",
		"Valve",
	 ],  # Categories to retain

	# Number of users to keep (randomly)
	"user_support": -1, # -1 implies use all users
	"seed": 42,
}
