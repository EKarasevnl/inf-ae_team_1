# @Time   : 2021/6/23
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/18
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

"""
recbole.evaluator.collector
################################################
"""

from recbole.evaluator.register import Register
import torch
import pandas as pd
import copy


class DataStruct(object):
    def __init__(self):
        self._data_dict = {}

    def __getitem__(self, name: str):
        return self._data_dict[name]

    def __setitem__(self, name: str, value):
        self._data_dict[name] = value

    def __delitem__(self, name: str):
        self._data_dict.pop(name)

    def __contains__(self, key: str):
        return key in self._data_dict

    def get(self, name: str):
        if name not in self._data_dict:
            raise IndexError("Can not load the data without registration !")
        return self[name]

    def set(self, name: str, value):
        self._data_dict[name] = value

    def update_tensor(self, name: str, value: torch.Tensor):
        if name not in self._data_dict:
            self._data_dict[name] = value.clone().detach()
        else:
            if not isinstance(self._data_dict[name], torch.Tensor):
                raise ValueError("{} is not a tensor.".format(name))
            self._data_dict[name] = torch.cat(
                (self._data_dict[name], value.clone().detach()), dim=0
            )

    def __str__(self):
        data_info = "\nContaining:\n"
        for data_key in self._data_dict.keys():
            data_info += data_key + "\n"
        return data_info


class Collector(object):
    """The collector is used to collect the resource for evaluator.
    As the evaluation metrics are various, the needed resource not only contain the recommended result
    but also other resource from data and model. They all can be collected by the collector during the training
    and evaluation process.

    This class is only used in Trainer.

    """

    def __init__(self, config):
        self.config = config
        self.data_struct = DataStruct()
        self.register = Register(config)
        self.full = "full" in config["eval_args"]["mode"]
        self.topk = self.config["topk"]
        self.device = self.config["device"]
        

        self.item_df = pd.read_csv(
            self.config["ITEM_PATH"],
            delimiter="\t",
            header=0,
            engine="python",
            encoding="utf-8",
            on_bad_lines="warn",
        )

        self.inter_df = pd.read_csv(self.config["INTER_PATH"], delimiter='\t', header=0, engine='python', encoding='latin-1')

        self.item_df_id = self.item_df[self.config["item_id_field"]].unique()
        print(f"Num unique ids in inter_item_id: {self.item_df_id.size} || {self.item_df_id}")
        
        self.item_map_to_category = dict(
                zip(self.item_df[self.config["item_id_field"]].astype(int), self.item_df[self.config["category_id"]])
                )

    def _map_internal_to_original_ids(self, internal_ids, dataset):
        """Map internal item IDs back to original item IDs.
        
        Args:
            internal_ids (torch.Tensor): Tensor of internal item IDs
            dataset: Dataset object containing the field2id_token mapping
            
        Returns:
            torch.Tensor: Tensor of original item IDs as integers
        """
        # Get the item field name
        item_field = dataset.iid_field
        
        # Use field2id_token to map internal IDs to original tokens
        # field2id_token[item_field] contains the mapping from internal ID to original token
        id_to_token = dataset.field2id_token[item_field]
        
        # Convert internal IDs to original IDs
        original_ids = []
        for idx in internal_ids.flatten():
            if idx.item() < len(id_to_token):
                original_token = id_to_token[idx.item()]
                # Skip special tokens like [PAD]
                if original_token.startswith('[') and original_token.endswith(']'):
                    original_ids.append(0)
                else:
                    original_id = int(float(original_token))
                    original_ids.append(original_id)
            else:
                original_ids.append(0)  # Fallback for invalid indices
        
        return torch.tensor(original_ids, dtype=torch.long, device=internal_ids.device).reshape(internal_ids.shape)

    def data_collect(self, train_data):
        """Collect the evaluation resource from training data.
        Args:
            train_data (AbstractDataLoader): the training dataloader which contains the training data.

        """
        # Store dataset reference for ID mapping
        self._dataset_ref = train_data.dataset
        
        if self.register.need("data.num_items"):
            item_id = self.config["ITEM_ID_FIELD"]
            self.data_struct.set("data.num_items", train_data.dataset.num(item_id))
        if self.register.need("data.num_users"):
            user_id = self.config["USER_ID_FIELD"]
            self.data_struct.set("data.num_users", train_data.dataset.num(user_id))
        if self.register.need("data.count_items"):
            self.data_struct.set("data.count_items", train_data.dataset.item_counter)
        if self.register.need("data.count_users"):
            self.data_struct.set("data.count_users", train_data.dataset.user_counter)

    def _average_rank(self, scores):
        """Get the ranking of an ordered tensor, and take the average of the ranking for positions with equal values.

        Args:
            scores(tensor): an ordered tensor, with size of (N, )

        Returns:
            torch.Tensor: average_rank

        Example:
            >>> average_rank(tensor([[1,2,2,2,3,3,6],[2,2,2,2,4,5,5]]))
            tensor([[1.0000, 3.0000, 3.0000, 3.0000, 5.5000, 5.5000, 7.0000],
            [2.5000, 2.5000, 2.5000, 2.5000, 5.0000, 6.5000, 6.5000]])

        Reference:
            https://github.com/scipy/scipy/blob/v0.17.1/scipy/stats/stats.py#L5262-L5352

        """
        length, width = scores.shape
        true_tensor = torch.full(
            (length, 1), True, dtype=torch.bool, device=self.device
        )

        obs = torch.cat([true_tensor, scores[:, 1:] != scores[:, :-1]], dim=1)
        # bias added to dense
        bias = (
            torch.arange(0, length, device=self.device)
            .repeat(width)
            .reshape(width, -1)
            .transpose(1, 0)
            .reshape(-1)
        )
        dense = obs.view(-1).cumsum(0) + bias

        # cumulative counts of each unique value
        count = torch.where(torch.cat([obs, true_tensor], dim=1))[1]
        # get average rank
        avg_rank = 0.5 * (count[dense] + count[dense - 1] + 1).view(length, -1)

        return avg_rank

    def eval_batch_collect(
        self,
        scores_tensor: torch.Tensor,
        interaction,
        positive_u: torch.Tensor,
        positive_i: torch.Tensor,
    ):
        """Collect the evaluation resource from batched eval data and batched model output.
        Args:
            scores_tensor (Torch.Tensor): the output tensor of model with the shape of (N, )
            interaction(Interaction): batched eval data.
            positive_u(Torch.Tensor): the row index of positive items for each user.
            positive_i(Torch.Tensor): the positive item id for each user.
        """
        if self.register.need("rec.items"):
            # get topk
            _, topk_idx = torch.topk(
                scores_tensor, max(self.topk), dim=-1
            )  # n_users x k

            # Map internal IDs to original IDs
            topk_original_ids = self._map_internal_to_original_ids(topk_idx, self._dataset_ref)

            # Store both internal indices and original IDs
            self.data_struct.update_tensor("rec.items.internal", topk_idx)
            self.data_struct.update_tensor("rec.items.original", topk_original_ids)
            
            # For backward compatibility
            self.data_struct.update_tensor("rec.items", topk_idx)
        
        if self.register.need("rec.topk"):
            _, topk_idx = torch.topk(
                scores_tensor, max(self.topk), dim=-1
            )  # n_users x k
            pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
            pos_matrix[positive_u, positive_i] = 1
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            self.data_struct.update_tensor("rec.topk", result)

        if self.register.need("rec.meanrank"):
            desc_scores, desc_index = torch.sort(scores_tensor, dim=-1, descending=True)

            # get the index of positive items in the ranking list
            pos_matrix = torch.zeros_like(scores_tensor)
            pos_matrix[positive_u, positive_i] = 1
            pos_index = torch.gather(pos_matrix, dim=1, index=desc_index)

            avg_rank = self._average_rank(desc_scores)
            pos_rank_sum = torch.where(
                pos_index == 1, avg_rank, torch.zeros_like(avg_rank)
            ).sum(dim=-1, keepdim=True)

            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            user_len_list = desc_scores.argmin(dim=1, keepdim=True)
            result = torch.cat((pos_rank_sum, user_len_list, pos_len_list), dim=1)
            self.data_struct.update_tensor("rec.meanrank", result)

        if self.register.need("rec.score"):
            self.data_struct.update_tensor("rec.score", scores_tensor)

        if self.register.need("data.label"):
            self.label_field = self.config["LABEL_FIELD"]
            self.data_struct.update_tensor(
                "data.label", interaction[self.label_field].to(self.device)
            )
        
        if self.register.need("data.group_key"):
            self.data_struct.set("data.group_key", self.config["group_key"])

        if self.register.need("data.group_map"):
            self.data_struct.set("data.group_map", self.item_map_to_category)
        
        if self.register.need("inter_id"):
            self.data_struct.set("inter_id", self.item_df_id)

        # if self.register.need("rec.label"):
        #     # Get global user ID for each row
        #     batch_user_id = interaction[self.config["USER_ID_FIELD"]].tolist()
        #     true_pos = {u: [] for u in batch_user_id}
        #     for u, i in zip(positive_u.tolist(), positive_i.tolist()):
        #         global_u = batch_user_id[u]
        #         true_pos[global_u].append(i)

        #     if "rec.label" not in self.data_struct:
        #         # One list per total user (known from training)
        #         num_total_users = self.data_struct.get("data.num_users")
        #         all_true_pos = [[] for _ in range(num_total_users)]
        #         for u, items in true_pos.items():
        #             all_true_pos[u] = items
        #         self.data_struct.set("rec.label", all_true_pos)
        #     else:
        #         existing = self.data_struct.get("rec.label")
        #         for u, items in true_pos.items():
        #             existing[u].extend(items)
        #         self.data_struct.set("rec.label", existing)

          # ------------------------------------------------------------------
        # Store each user's ground-truth items so metrics like PSP can
        # compute the per-user denominator (mPSP).  We build one list per
        # *global* user ID and keep extending it batch-by-batch.
        # ------------------------------------------------------------------
        if self.register.need("rec.label"):

            # ❶ Lazily create the master list-of-lists only once
            if "rec.label" not in self.data_struct:
                num_total_users = self.data_struct.get("data.num_users")
                self.data_struct.set("rec.label", [[] for _ in range(num_total_users)])

            rec_label = self.data_struct.get("rec.label")

            # ❷ Map local batch rows to global user IDs
            batch_user_id = interaction[self.config["USER_ID_FIELD"]].tolist()

            # ❸ Append the positive item of *every* user in this batch
            for local_u, item_id in zip(positive_u.tolist(), positive_i.tolist()):
                global_u = batch_user_id[local_u]
                rec_label[global_u].append(int(item_id))     # make sure it's plain int

            # ❹ Put it back (lists are mutable, but stay explicit)
            self.data_struct.set("rec.label", rec_label)


        # if self.register.need("rec.label"):
        #     # Get global user ID for each row
        #     batch_user_id = interaction[self.config["USER_ID_FIELD"]].tolist()
        #     true_pos = {u: [] for u in batch_user_id}
        #     for u, i in zip(positive_u.tolist(), positive_i.tolist()):
        #         global_u = batch_user_id[u]
        #         true_pos[global_u].append(i)

        #     if "rec.label" not in self.data_struct:
        #         # One list per total user (known from training)
        #         num_total_users = self.data_struct.get("data.num_users")
        #         all_true_pos = [[] for _ in range(num_total_users)]
        #         for u, items in true_pos.items():
        #             all_true_pos[u] = items
        #         self.data_struct.set("rec.label", all_true_pos)
        #     else:
        #         existing = self.data_struct.get("rec.label")
        #         for u, items in true_pos.items():
        #             existing[u].extend(items)
        #         self.data_struct.set("rec.label", existing)

          # ------------------------------------------------------------------
        # Store each user's ground-truth items so metrics like PSP can
        # compute the per-user denominator (mPSP).  We build one list per
        # *global* user ID and keep extending it batch-by-batch.
        # ------------------------------------------------------------------
        if self.register.need("rec.label"):

            # ❶ Lazily create the master list-of-lists only once
            if "rec.label" not in self.data_struct:
                num_total_users = self.data_struct.get("data.num_users")
                self.data_struct.set("rec.label", [[] for _ in range(num_total_users)])

            rec_label = self.data_struct.get("rec.label")

            # ❷ Map local batch rows to global user IDs
            batch_user_id = interaction[self.config["USER_ID_FIELD"]].tolist()

            # ❸ Append the positive item of *every* user in this batch
            for local_u, item_id in zip(positive_u.tolist(), positive_i.tolist()):
                global_u = batch_user_id[local_u]
                rec_label[global_u].append(int(item_id))     # make sure it's plain int

            # ❹ Put it back (lists are mutable, but stay explicit)
            self.data_struct.set("rec.label", rec_label)


    def model_collect(self, model: torch.nn.Module):
        """Collect the evaluation resource from model.
        Args:
            model (nn.Module): the trained recommendation model.
        """
        pass

    def eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor):
        """Collect the evaluation resource from total output and label.
        It was designed for those models that can not predict with batch.
        Args:
            eval_pred (torch.Tensor): the output score tensor of model.
            data_label (torch.Tensor): the label tensor.
        """
        if self.register.need("rec.score"):
            self.data_struct.update_tensor("rec.score", eval_pred)

        if self.register.need("rec.label"):
            self.label_field = self.config["LABEL_FIELD"]
            self.data_struct.update_tensor("rec.label", data_label.to(self.device))

    def get_data_struct(self):
        """Get all the evaluation resource that been collected.
        And reset some of outdated resource.
        """
        for key, value in list(self.data_struct._data_dict.items()):
            if hasattr(value, "cpu"):
                self.data_struct._data_dict[key] = value.cpu()

        returned_struct = copy.deepcopy(self.data_struct)
        for key in ["rec.topk", "rec.meanrank", "rec.score", "rec.items", "rec.items.internal", "rec.items.original", "data.label", "rec.label"]:
            if key in self.data_struct:
                del self.data_struct[key]
        return returned_struct
