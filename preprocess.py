import numpy as np
import h5py, sys, os

BASE_PATH = "data/"

# def prep_movielens(ratings_file_path):
#     f = open(ratings_file_path, "r")
#     users, items, ratings = [], [], []

#     # f.readline() # Skip the header
#     idx = 1
#     line = f.readline()

#     while line:
#         try:
#             u, i, r, _ = line.strip().split("::")
#         except:
#             print(line)
#             print(f"Line number: {idx}")

#         users.append(int(u))
#         items.append(int(i))
#         ratings.append(float(r))
#         line = f.readline()
#         idx += 1

#     min_user = min(users)
#     num_users = len(set(users))
#     print(f"min_user - {min_user} | num_users - {num_users} | number of users(+dupli) - {len(users)}")

#     data = [ [] for _ in range(num_users) ]
#     for i in range(len(users)):
#         try:
#             data[users[i] - min_user].append([ items[i], ratings[i] ])
#         except:
#             print(f"Faulty idx - {users[i] - min_user}")

#     return rating_data(data)

def prep_movielens(ratings_file_path):
    f = open(ratings_file_path, "r")
    users, items, ratings = [], [], []
    
    if "csv" in ratings_file_path:
        f.readline() # Uncomment if the file has a header

    idx = 1
    line = f.readline()

    while line:
        try:
            delimeter = "," if "csv" in ratings_file_path else "::"
            u, i, r, _ = line.strip().split(delimeter)
            users.append(int(u))
            items.append(int(i))
            ratings.append(float(r))
        except:
            print(f"Error parsing line {idx}: {line.strip()}")
            assert False

        line = f.readline()
        idx += 1

    unique_users = sorted(set(users))
    user_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}

    # print(f"min_user - {min(users)} | num_users - {len(unique_users)} | number of users(+dupli) - {len(users)}")

    data = [[] for _ in range(len(unique_users))]
    for i in range(len(users)):
        try:
            data[user_to_index[users[i]]].append([items[i], ratings[i]])
        except Exception as e:
            print(f"Error at i={i}, user={users[i]}, mapped_idx={user_to_index.get(users[i], 'N/A')} | {e}")
            assert False

    return rating_data(data)

class rating_data:
    def __init__(self, data):
        self.data = data

        self.index = [] # 0: train, 1: validation, 2: test, -1: removed due to user frequency < 3
        for user_data in self.data:
            for _ in range(len(user_data)): self.index.append(42)

    def train_test_split(self):
        at = 0

        for user in range(len(self.data)):
            first_split_point = int(0.8 * len(self.data[user]))
            second_split_point = int(0.9 * len(self.data[user]))

            indices = np.arange(len(self.data[user]))
            np.random.shuffle(indices)

            for timestep, (item, rating) in enumerate(self.data[user]):
                if len(self.data[user]) < 3: self.index[at] = -1
                else:
                    # Force atleast one element in user history to be in test
                    if timestep == indices[0]: self.index[at] = 2
                    else:
                        if timestep in indices[:first_split_point]: self.index[at] = 0
                        elif timestep in indices[first_split_point:second_split_point]: self.index[at] = 1
                        else: self.index[at] = 2
                at += 1

        assert at == len(self.index)
        self.complete_data_stats = None

    def save_index(self, path):
        os.makedirs(path, exist_ok = True)
        with open(path + "/index.npz", "wb") as f: np.savez_compressed(f, data = self.index)

    def save_data(self, path):
        flat_data = []
        for u in range(len(self.data)):
            flat_data += list(map(lambda x: [ u ] + x, self.data[u]))
        flat_data = np.array(flat_data)

        shape = [ len(flat_data) ]

        os.makedirs(path, exist_ok = True)
        with h5py.File(path + '/total_data.hdf5', 'w') as file:
            dset = {}
            dset['user'] = file.create_dataset("user", shape, dtype = 'i4', maxshape = shape, compression="gzip")
            dset['item'] = file.create_dataset("item", shape, dtype = 'i4', maxshape = shape, compression="gzip")
            dset['rating'] = file.create_dataset("rating", shape, dtype = 'f', maxshape = shape, compression="gzip")

            dset['user'][:] = flat_data[:, 0]
            dset['item'][:] = flat_data[:, 1]
            dset['rating'][:] = flat_data[:, 2]

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("This file needs the dataset name as the first argument...")
        exit(0)
    
    dataset = sys.argv[1]

    print("\n\n!!!!!!!! STARTED PROCESSING {} !!!!!!!!".format(dataset))

    if dataset in [ 'ml-1m' ]: total_data = prep_movielens(BASE_PATH + "/ml-1m/ratings.dat")
    if dataset in [ 'ml-10m' ]: total_data = prep_movielens(BASE_PATH + "/ml-10m/ratings.dat")
    if dataset in [ 'ml-20m' ]: total_data = prep_movielens(BASE_PATH + "/ml-20m/ratings.csv")

    total_data.save_data(BASE_PATH + "{}/".format(dataset))
    total_data.train_test_split()
    total_data.save_index(BASE_PATH + "{}/".format(dataset))
