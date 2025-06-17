# -*- encoding: utf-8 -*-
# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/08/12, 2021/8/29, 2020/9/16, 2021/7/2
# @Author  :   Kaiyuan Li, Zhichao Feng, Xingyu Pan, Zihan Lin
# @email   :   tsotfsk@outlook.com, fzcbupt@gmail.com, panxy@ruc.edu.cn, zhlin@ruc.edu.cn

r"""
recbole.evaluator.metrics
############################

Suppose there is a set of :math:`n` items to be ranked. Given a user :math:`u` in the user set :math:`U`,
we use :math:`\hat R(u)` to represent a ranked list of items that a model produces, and :math:`R(u)` to
represent a ground-truth set of items that user :math:`u` has interacted with. For top-k recommendation, only
top-ranked items are important to consider. Therefore, in top-k evaluation scenarios, we truncate the
recommendation list with a length :math:`K`. Besides, in loss-based metrics, :math:`S` represents the
set of user(u)-item(i) pairs, :math:`\hat r_{u i}` represents the score predicted by the model,
:math:`{r}_{u i}` represents the ground-truth labels.

"""

from logging import getLogger

import numpy as np
from collections import Counter
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import mean_absolute_error, mean_squared_error

from recbole.evaluator.utils import _binary_clf_curve
from recbole.evaluator.base_metric import AbstractMetric, TopkMetric, LossMetric
from recbole.utils import EvaluatorType

# TopK Metrics


class Hit(TopkMetric):
    r"""HR_ (also known as truncated Hit-Ratio) is a way of calculating how many 'hits'
    you have in an n-sized list of ranked items. If there is at least one item that falls in the ground-truth set,
    we call it a hit.

    .. _HR: https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870

    .. math::
        \mathrm {HR@K} = \frac{1}{|U|}\sum_{u \in U} \delta(\hat{R}(u) \cap R(u) \neq \emptyset),

    :math:`\delta(·)` is an indicator function. :math:`\delta(b)` = 1 if :math:`b` is true and 0 otherwise.
    :math:`\emptyset` denotes the empty set.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result("hit", result)
        return metric_dict

    def metric_info(self, pos_index):
        result = np.cumsum(pos_index, axis=1)
        return (result > 0).astype(int)


class MRR(TopkMetric):
    r"""The MRR_ (also known as Mean Reciprocal Rank) computes the reciprocal rank
    of the first relevant item found by an algorithm.

    .. _MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank

    .. math::
       \mathrm {MRR@K} = \frac{1}{|U|}\sum_{u \in U} \frac{1}{\operatorname{rank}_{u}^{*}}

    :math:`{rank}_{u}^{*}` is the rank position of the first relevant item found by an algorithm for a user :math:`u`.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result("mrr", result)
        return metric_dict

    def metric_info(self, pos_index):
        idxs = pos_index.argmax(axis=1)
        result = np.zeros_like(pos_index, dtype=np.float)
        for row, idx in enumerate(idxs):
            if pos_index[row, idx] > 0:
                result[row, idx:] = 1 / (idx + 1)
            else:
                result[row, idx:] = 0
        return result


class MAP(TopkMetric):
    r"""MAP_ (also known as Mean Average Precision) is meant to calculate
    average precision for the relevant items.

    Note:
        In this case the normalization factor used is :math:`\frac{1}{min(|\hat R(u)|, K)}`, which prevents your
        AP score from being unfairly suppressed when your number of recommendations couldn't possibly capture
        all the correct ones.

    .. _MAP: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms

    .. math::
       \mathrm{MAP@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{min(|\hat R(u)|, K)} \sum_{j=1}^{|\hat{R}(u)|} I\left(\hat{R}_{j}(u) \in R(u)\right) \cdot  Precision@j)

    :math:`\hat{R}_{j}(u)` is the j-th item in the recommendation list of \hat R (u)).
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("map", result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        pre = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
        sum_pre = np.cumsum(pre * pos_index.astype(np.float), axis=1)
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
        result = np.zeros_like(pos_index, dtype=np.float)
        for row, lens in enumerate(actual_len):
            ranges = np.arange(1, pos_index.shape[1] + 1)
            ranges[lens:] = ranges[lens - 1]
            result[row] = sum_pre[row] / ranges
        return result


class Recall(TopkMetric):
    r"""Recall_ is a measure for computing the fraction of relevant items out of all relevant items.

    .. _recall: https://en.wikipedia.org/wiki/Precision_and_recall#Recall

    .. math::
       \mathrm {Recall@K} = \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|R(u)|}

    :math:`|R(u)|` represents the item count of :math:`R(u)`.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("recall", result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


class NDCG(TopkMetric):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality,
    where positions are discounted logarithmically. It accounts for the position of the hit by assigning
    higher scores to hits at top ranks.

    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \mathrm {NDCG@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{\sum_{i=1}^{\min (|R(u)|, K)}
        \frac{1}{\log _{2}(i+1)}} \sum_{i=1}^{K} \delta(i \in R(u)) \frac{1}{\log _{2}(i+1)})

    :math:`\delta(·)` is an indicator function.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("ndcg", result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_index, dtype=np.float)
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(pos_index, dtype=np.float)
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

        result = dcg / idcg
        return result


class Precision(TopkMetric):
    r"""Precision_ (also called positive predictive value) is a measure for computing the fraction of relevant items
    out of all the recommended items. We average the metric for each user :math:`u` get the final result.

    .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision

    .. math::
        \mathrm {Precision@K} =  \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|\hat {R}(u)|}

    :math:`|\hat R(u)|` represents the item count of :math:`\hat R(u)`.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result("precision", result)
        return metric_dict

    def metric_info(self, pos_index):
        return pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)


# Custom PSP
# class PSP(TopkMetric):
#     metric_type = EvaluatorType.RANKING
#     metric_need = ["rec.topk"]  # ✅ only tensor input

#     def __init__(self, config):  # <-- add dataset param!
#         super().__init__(config)
#         self.item_count = config["psp_item_count"]

#     def used_info(self, dataobject):
#         rec_mat = dataobject.get("rec.topk")
#         topk_idx, pos_len_list = np.split(rec_mat, [max(self.topk)], axis=1)
#         return (
#             topk_idx.cpu().numpy(),
#             pos_len_list.squeeze(-1).cpu().numpy()
#         )

#     def metric_info(self, pos_index, topk_idx):
#         pop_weights = np.vectorize(lambda x: 1.0 / np.log1p(self.item_count.get(x, 1)))
#         weights = pop_weights(topk_idx)
#         weights *= pos_index
#         return weights.cumsum(axis=1) / np.arange(1, weights.shape[1] + 1)

#     def calculate_metric(self, dataobject):
#         topk_idx, pos_len_list = self.used_info(dataobject)
#         pos_index = topk_idx.astype(bool)
#         result = self.metric_info(pos_index, topk_idx)
#         return self.topk_result("psp", result)

# maybe version
# class PSP(TopkMetric):
#     r"""
#     PSP (Propensity Scored Precision) is a position-aware metric
#     that weights hits by item propensities (counts).

#     For each user:
#     .. math::
#         PSP@K(u) = \frac{
#             \sum_{i=1}^{K} \mathbb{1}[r_i \in R(u)] \cdot \text{propensity}(r_i)
#         }{
#             \sum_{i=1}^{\min(|R(u)|, K)} \text{top propensities}
#         }

#     Then,
#     .. math::
#         PSP@K = \frac{1}{|U|} \sum_{u \in U} PSP@K(u)

#     Uses `data.count_items` collected by Collector.
#     """

#     metric_type = EvaluatorType.RANKING
#     metric_need = ["rec.topk", "rec.items", "data.count_items", "rec.label"]

#     def __init__(self, config):
#         super().__init__(config)

#     def calculate_metric(self, dataobject):
#         pos_index, pos_len = self.used_info(dataobject)

#         propensities = dataobject["data.count_items"].numpy()
#         rec_items = dataobject["rec.items"].numpy()

#         result = self.metric_info(pos_index, pos_len, propensities, rec_items, dataobject)
#         metric_dict = self.topk_result("psp", result)
#         return metric_dict

#     def metric_info(self, pos_index, pos_len, propensities, rec_items, dataobject):
#         rec_prop = propensities[rec_items]

#         # Numerator: sum of propensities for hits
#         psp_numerator = np.cumsum(pos_index * rec_prop, axis=1)

#         # Denominator: top true propensities for each user
#         num_users, max_k = pos_index.shape
#         denom = np.zeros((num_users, max_k))

#         for u in range(num_users):
#             true_items = dataobject["rec.label"][u]
#             if len(true_items) > 0:
#                 true_props = np.sort(propensities[true_items])[::-1]
#             else:
#                 true_props = np.array([0.0])

#             top_props = np.cumsum(
#                 np.pad(true_props, (0, max(0, max_k - len(true_props))), constant_values=0)
#             )
#             if len(top_props) < max_k:
#                 top_props = np.concatenate([top_props, np.repeat(top_props[-1], max_k - len(top_props))])
#             denom[u, :] = top_props[:max_k]

#         denom[denom == 0] = 1.0
#         return psp_numerator / denom

class PSP(TopkMetric):
    r"""
    PSP (Propensity Scored Precision) is a position-aware metric 
    that weights hits by item propensities.

    For each user:
    .. math::
        PSP@K(u) = \frac{
            \sum_{i=1}^{K} \mathbb{1}[r_i \in R(u)] \cdot \text{propensity}(r_i)
        }{
            \sum_{i=1}^{\min(|R(u)|, K)} \text{top propensities}
        }

    Then,
    .. math::
        PSP@K = \frac{1}{|U|} \sum_{u \in U} PSP@K(u)

    It uses `data.count_items` collected by the `Collector` as the propensity.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.topk", "rec.items", "data.count_items", "rec.label"]

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        # 1️⃣ Standard: get hit indicator matrix and positive lengths
        pos_index, pos_len = self.used_info(dataobject)

        # 2️⃣ Robust: get the item count Counter and convert to dense array
        counter = dataobject["data.count_items"]  # this is a Counter
        num_items = max(counter.keys()) + 1
        propensities = np.zeros(num_items, dtype=np.float32)
        for item_id, count in counter.items():
            propensities[item_id] = count

        # 3️⃣ Get the recommended item IDs at each position
        rec_items = dataobject["rec.items"].numpy()

        # 4️⃣ Compute PSP per user
        result = self.metric_info(pos_index, pos_len, propensities, rec_items, dataobject)

        # 5️⃣ Standard topk output
        metric_dict = self.topk_result("psp", result)
        return metric_dict

    def metric_info(self, pos_index, pos_len, propensities, rec_items, dataobject):
        """
        pos_index: [num_users, max_k]  -> 1 if item at rank is correct
        propensities: [num_items] -> count of each item
        rec_items: [num_users, max_k] -> which item was recommended at each position
        """

        # Numerator: propensities of recommended hits
        rec_prop = propensities[rec_items]  # shape [num_users, max_k]
        psp_numerator = np.cumsum(pos_index * rec_prop, axis=1)

        # Denominator: for each user, the sum of top-K true item propensities
        num_users, max_k = pos_index.shape
        denom = np.zeros((num_users, max_k))

        for u in range(num_users):
            true_items = dataobject["rec.label"][u]  # list of true item IDs for user u
            if len(true_items) > 0:
                true_props = np.sort(propensities[true_items])[::-1]
                top_props = np.cumsum(
                    np.pad(true_props, (0, max(0, max_k - len(true_props))), constant_values=0)
                )
                if len(top_props) < max_k:
                    top_props = np.concatenate([top_props, np.repeat(top_props[-1], max_k - len(top_props))])
                denom[u, :] = top_props[:max_k]
            else:
                # If no true items, fallback to avoid divide by zero
                denom[u, :] = 1.0
                print("[DEBUG] Number of users with no true items:", np.sum(denom == 1.0))

        # Final safe division
        result = psp_numerator / denom
        return result


# CTR Metrics


class GAUC(AbstractMetric):
    r"""GAUC (also known as Grouped Area Under Curve) is used to evaluate the two-class model, referring to
    the area under the ROC curve grouped by user. We weighted the index of each user :math:`u` by the number of positive
    samples of users to get the final result.

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3219819.3219823>`__

    Note:
        It calculates the AUC score of each user, and finally obtains GAUC by weighting the user AUC.
        It is also not limited to k. Due to our padding for `scores_tensor` with `-np.inf`, the padding
        value will influence the ranks of origin items. Therefore, we use descending sort here and make
        an identity transformation  to the formula of `AUC`, which is shown in `auc_` function.
        For readability, we didn't do simplification in the code.

    .. math::
        \begin{align*}
            \mathrm {AUC(u)} &= \frac {{{|R(u)|} \times {(n+1)} - \frac{|R(u)| \times (|R(u)|+1)}{2}} -
            \sum\limits_{i=1}^{|R(u)|} rank_{i}} {{|R(u)|} \times {(n - |R(u)|)}} \\
            \mathrm{GAUC} &= \frac{1}{\sum_{u \in U} |R(u)|}\sum_{u \in U} |R(u)| \cdot(\mathrm {AUC(u)})
        \end{align*}

    :math:`rank_i` is the descending rank of the i-th items in :math:`R(u)`.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.meanrank"]

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        mean_rank = dataobject.get("rec.meanrank").numpy()
        pos_rank_sum, user_len_list, pos_len_list = np.split(mean_rank, 3, axis=1)
        user_len_list, pos_len_list = user_len_list.squeeze(-1), pos_len_list.squeeze(
            -1
        )
        result = self.metric_info(pos_rank_sum, user_len_list, pos_len_list)
        return {"gauc": round(result, self.decimal_place)}

    def metric_info(self, pos_rank_sum, user_len_list, pos_len_list):
        """Get the value of GAUC metric.

        Args:
            pos_rank_sum (numpy.ndarray): sum of descending rankings for positive items of each users.
            user_len_list (numpy.ndarray): the number of predicted items for users.
            pos_len_list (numpy.ndarray): the number of positive items for users.

        Returns:
            float: The value of the GAUC.
        """
        neg_len_list = user_len_list - pos_len_list
        # check positive and negative samples
        any_without_pos = np.any(pos_len_list == 0)
        any_without_neg = np.any(neg_len_list == 0)
        non_zero_idx = np.full(len(user_len_list), True, dtype=np.bool)
        if any_without_pos:
            logger = getLogger()
            logger.warning(
                "No positive samples in some users, "
                "true positive value should be meaningless, "
                "these users have been removed from GAUC calculation"
            )
            non_zero_idx *= pos_len_list != 0
        if any_without_neg:
            logger = getLogger()
            logger.warning(
                "No negative samples in some users, "
                "false positive value should be meaningless, "
                "these users have been removed from GAUC calculation"
            )
            non_zero_idx *= neg_len_list != 0
        if any_without_pos or any_without_neg:
            item_list = user_len_list, neg_len_list, pos_len_list, pos_rank_sum
            user_len_list, neg_len_list, pos_len_list, pos_rank_sum = map(
                lambda x: x[non_zero_idx], item_list
            )

        pair_num = (
            (user_len_list + 1) * pos_len_list
            - pos_len_list * (pos_len_list + 1) / 2
            - np.squeeze(pos_rank_sum)
        )
        user_auc = pair_num / (neg_len_list * pos_len_list)
        result = (user_auc * pos_len_list).sum() / pos_len_list.sum()
        return result


class AUC(LossMetric):
    r"""AUC_ (also known as Area Under Curve) is used to evaluate the two-class model, referring to
    the area under the ROC curve.

    .. _AUC: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

    Note:
        This metric does not calculate group-based AUC which considers the AUC scores
        averaged across users. It is also not limited to k. Instead, it calculates the
        scores on the entire prediction results regardless the users. We call the interface
        in `scikit-learn`, and code calculates the metric using the variation of following formula.

    .. math::
        \mathrm {AUC} = \frac {{{M} \times {(N+1)} - \frac{M \times (M+1)}{2}} -
        \sum\limits_{i=1}^{M} rank_{i}} {{M} \times {(N - M)}}

    :math:`M` denotes the number of positive items.
    :math:`N` denotes the total number of user-item interactions.
    :math:`rank_i` denotes the descending rank of the i-th positive item.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric("auc", dataobject)

    def metric_info(self, preds, trues):
        fps, tps = _binary_clf_curve(trues, preds)
        if len(fps) > 2:
            optimal_idxs = np.where(
                np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True]
            )[0]
            fps = fps[optimal_idxs]
            tps = tps[optimal_idxs]

        tps = np.r_[0, tps]
        fps = np.r_[0, fps]

        if fps[-1] <= 0:
            logger = getLogger()
            logger.warning(
                "No negative samples in y_true, "
                "false positive value should be meaningless"
            )
            fpr = np.repeat(np.nan, fps.shape)
        else:
            fpr = fps / fps[-1]

        if tps[-1] <= 0:
            logger = getLogger()
            logger.warning(
                "No positive samples in y_true, "
                "true positive value should be meaningless"
            )
            tpr = np.repeat(np.nan, tps.shape)
        else:
            tpr = tps / tps[-1]

        result = sk_auc(fpr, tpr)
        return result


# Loss-based Metrics


class MAE(LossMetric):
    r"""MAE_ (also known as Mean Absolute Error regression loss) is used to evaluate the difference between
    the score predicted by the model and the actual behavior of the user.

    .. _MAE: https://en.wikipedia.org/wiki/Mean_absolute_error

    .. math::
        \mathrm{MAE}=\frac{1}{|{S}|} \sum_{(u, i) \in {S}}\left|\hat{r}_{u i}-r_{u i}\right|

    :math:`|S|` represents the number of pairs in :math:`S`.
    """

    smaller = True

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric("mae", dataobject)

    def metric_info(self, preds, trues):
        return mean_absolute_error(trues, preds)


class RMSE(LossMetric):
    r"""RMSE_ (also known as Root Mean Squared Error) is another error metric like `MAE`.

    .. _RMSE: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    .. math::
       \mathrm{RMSE} = \sqrt{\frac{1}{|{S}|} \sum_{(u, i) \in {S}}(\hat{r}_{u i}-r_{u i})^{2}}
    """

    smaller = True

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric("rmse", dataobject)

    def metric_info(self, preds, trues):
        return np.sqrt(mean_squared_error(trues, preds))


class LogLoss(LossMetric):
    r"""Logloss_ (also known as logistic loss or cross-entropy loss) is used to evaluate the probabilistic
    output of the two-class classifier.

    .. _Logloss: http://wiki.fast.ai/index.php/Log_Loss

    .. math::
        LogLoss = \frac{1}{|S|} \sum_{(u,i) \in S}(-((r_{u i} \ \log{\hat{r}_{u i}}) + {(1 - r_{u i})}\ \log{(1 - \hat{r}_{u i})}))
    """

    smaller = True

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric("logloss", dataobject)

    def metric_info(self, preds, trues):
        eps = 1e-15
        preds = np.float64(preds)
        preds = np.clip(preds, eps, 1 - eps)
        loss = np.sum(-trues * np.log(preds) - (1 - trues) * np.log(1 - preds))
        return loss / len(preds)


class ItemCoverage(AbstractMetric):
    r"""ItemCoverage_ computes the coverage of recommended items over all items.

    .. _ItemCoverage: https://en.wikipedia.org/wiki/Coverage_(information_systems)

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/1864708.1864761>`__
    and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__.

    .. math::
       \mathrm{Coverage@K}=\frac{\left| \bigcup_{u \in U} \hat{R}(u) \right|}{|I|}
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "data.num_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        item_matrix = dataobject.get("rec.items")
        num_items = dataobject.get("data.num_items")
        return item_matrix.numpy(), num_items

    def calculate_metric(self, dataobject):
        item_matrix, num_items = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            key = "{}@{}".format("itemcoverage", k)
            metric_dict[key] = round(
                self.get_coverage(item_matrix[:, :k], num_items), self.decimal_place
            )
        return metric_dict

    def get_coverage(self, item_matrix, num_items):
        """Get the coverage of recommended items over all items

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            num_items(int): the total number of items.

        Returns:
            float: the `coverage` metric.
        """
        unique_count = np.unique(item_matrix).shape[0]
        return unique_count / num_items


class AveragePopularity(AbstractMetric):
    r"""AveragePopularity computes the average popularity of recommended items.

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.6700>`__
    and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__.

    .. math::
        \mathrm{AveragePopularity@K}=\frac{1}{|U|} \sum_{u \in U } \frac{\sum_{i \in R_{u}} \phi(i)}{|R_{u}|}

    :math:`\phi(i)` is the number of interaction of item i in training data.
    """

    metric_type = EvaluatorType.RANKING
    smaller = True
    metric_need = ["rec.items", "data.count_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and the popularity of items in training data"""
        item_counter = dataobject.get("data.count_items")
        item_matrix = dataobject.get("rec.items")
        return item_matrix.numpy(), dict(item_counter)

    def calculate_metric(self, dataobject):
        item_matrix, item_count = self.used_info(dataobject)
        result = self.metric_info(self.get_pop(item_matrix, item_count))
        metric_dict = self.topk_result("averagepopularity", result)
        return metric_dict

    def get_pop(self, item_matrix, item_count):
        """Convert the matrix of item id to the matrix of item popularity using a dict:{id,count}.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            item_count(dict): the number of interaction of items in training data.

        Returns:
            numpy.ndarray: the popularity of items in the recommended list.
        """
        value = np.zeros_like(item_matrix)
        for i in range(item_matrix.shape[0]):
            row = item_matrix[i, :]
            for j in range(row.shape[0]):
                value[i][j] = item_count.get(row[j], 0)
        return value

    def metric_info(self, values):
        return values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1)

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form

        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.

        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = "{}@{}".format(metric, k)
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict


class ShannonEntropy(AbstractMetric):
    r"""ShannonEntropy_ presents the diversity of the recommendation items.
    It is the entropy over items' distribution.

    .. _ShannonEntropy: https://en.wikipedia.org/wiki/Entropy_(information_theory)

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.6700>`__
    and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__

    .. math::
        \mathrm {ShannonEntropy@K}=-\sum_{i=1}^{|I|} p(i) \log p(i)

    :math:`p(i)` is the probability of recommending item i
    which is the number of item i in recommended list over all items.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the matrix of recommendation items."""
        item_matrix = dataobject.get("rec.items")
        return item_matrix.numpy()

    def calculate_metric(self, dataobject):
        item_matrix = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            key = "{}@{}".format("shannonentropy", k)
            metric_dict[key] = round(
                self.get_entropy(item_matrix[:, :k]), self.decimal_place
            )
        return metric_dict

    def get_entropy(self, item_matrix):
        """Get shannon entropy through the top-k recommendation list.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.

        Returns:
            float: the shannon entropy.
        """

        item_count = dict(Counter(item_matrix.flatten()))
        total_num = item_matrix.shape[0] * item_matrix.shape[1]
        result = 0.0
        for cnt in item_count.values():
            p = cnt / total_num
            result += -p * np.log(p)
        return result / len(item_count)


class GiniIndex(AbstractMetric):
    r"""GiniIndex presents the diversity of the recommendation items.
    It is used to measure the inequality of a distribution.

    .. _GiniIndex: https://en.wikipedia.org/wiki/Gini_coefficient

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3308560.3317303>`__.

    .. math::
        \mathrm {GiniIndex@K}=\left(\frac{\sum_{i=1}^{|I|}(2 i-|I|-1) P{(i)}}{|I| \sum_{i=1}^{|I|} P{(i)}}\right)

    :math:`P{(i)}` represents the number of times all items appearing in the recommended list,
    which is indexed in non-decreasing order (P_{(i)} \leq P_{(i+1)}).
    """

    metric_type = EvaluatorType.RANKING
    smaller = True
    metric_need = ["rec.items", "data.num_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        item_matrix = dataobject.get("rec.items")
        num_items = dataobject.get("data.num_items")
        return item_matrix.numpy(), num_items

    def calculate_metric(self, dataobject):
        item_matrix, num_items = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            key = "{}@{}".format("giniindex", k)
            metric_dict[key] = round(
                self.get_gini(item_matrix[:, :k], num_items), self.decimal_place
            )
        return metric_dict

    def get_gini(self, item_matrix, num_items):
        """Get gini index through the top-k recommendation list.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            num_items(int): the total number of items.

        Returns:
            float: the gini index.
        """
        item_count = dict(Counter(item_matrix.flatten()))
        sorted_count = np.array(sorted(item_count.values()))
        num_recommended_items = sorted_count.shape[0]
        total_num = item_matrix.shape[0] * item_matrix.shape[1]
        idx = np.arange(num_items - num_recommended_items + 1, num_items + 1)
        gini_index = np.sum((2 * idx - num_items - 1) * sorted_count) / total_num
        gini_index /= num_items
        return gini_index


class TailPercentage(AbstractMetric):
    r"""TailPercentage_ computes the percentage of long-tail items in recommendation items.

    .. _TailPercentage: https://en.wikipedia.org/wiki/Long_tail#Criticisms

    For further details, please refer to the `paper <https://arxiv.org/pdf/2007.12329.pdf>`__.

    .. math::
        \mathrm {TailPercentage@K}=\frac{1}{|U|} \sum_{u \in U} \frac{\sum_{i \in R_{u}} {\delta(i \in T)}}{|R_{u}|}

    :math:`\delta(·)` is an indicator function.
    :math:`T` is the set of long-tail items,
    which is a portion of items that appear in training data seldomly.

    Note:
        If you want to use this metric, please set the parameter 'tail_ratio' in the config
        which can be an integer or a float in (0,1]. Otherwise it will default to 0.1.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "data.count_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]
        self.tail = config["tail_ratio"]
        if self.tail is None or self.tail <= 0:
            self.tail = 0.1

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set."""
        item_matrix = dataobject.get("rec.items")
        count_items = dataobject.get("data.count_items")
        return item_matrix.numpy(), dict(count_items)

    def get_tail(self, item_matrix, count_items):
        """Get long-tail percentage through the top-k recommendation list.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            count_items(dict): the number of interaction of items in training data.

        Returns:
            float: long-tail percentage.
        """
        if self.tail > 1:
            tail_items = [item for item, cnt in count_items.items() if cnt <= self.tail]
        else:
            count_items = sorted(count_items.items(), key=lambda kv: (kv[1], kv[0]))
            cut = max(int(len(count_items) * self.tail), 1)
            count_items = count_items[:cut]
            tail_items = [item for item, cnt in count_items]
        value = np.zeros_like(item_matrix)
        for i in range(item_matrix.shape[0]):
            row = item_matrix[i, :]
            for j in range(row.shape[0]):
                value[i][j] = 1 if row[j] in tail_items else 0
        return value

    def calculate_metric(self, dataobject):
        item_matrix, count_items = self.used_info(dataobject)
        result = self.metric_info(self.get_tail(item_matrix, count_items))
        metric_dict = self.topk_result("tailpercentage", result)
        return metric_dict

    def metric_info(self, values):
        return values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1)

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form.

        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.

        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = "{}@{}".format(metric, k)
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict
