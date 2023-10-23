import torch
import torch.nn as nn
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.losses.base_metric_loss_function import  BaseMetricLossFunction
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
from pytorch_metric_learning.reducers import AvgNonZeroReducer

from pytorch_metric_learning.losses import SupConLoss

class FocalSupConLoss(BaseMetricLossFunction):
    def __init__(self, temperature=0.1, use_focal_scaling=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.temperature = temperature
        self.use_focal_scaling=use_focal_scaling

    def forward(self,
                embeddings, # (bs, h)
                labels, # (bs,)
                indices_tuple=None, 
                ref_emb=None, 
                ref_labels=None,
                bias_emb=None,): # (bs, d)
        self.reset_stats()
        c_f.check_shapes(embeddings, labels)
        if bias_emb != None:
            cos = CosineSimilarity()
            mat = 1 - cos(bias_emb)
            denominator = lmu.logsumexp(
                mat, add_one=False, dim=1
            )
            log_prob = mat - denominator
            bias_score = torch.exp(log_prob)
        else:
            bias_score = None

        if labels is not None:
            labels = c_f.to_device(labels, embeddings)
        ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        
        loss_dict = self.compute_loss(
            embeddings, labels, bias_score, indices_tuple, ref_emb, ref_labels
        )
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)
    
    def compute_loss(self, embeddings, labels, bias_score, indices_tuple, ref_emb, ref_labels):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels, ref_labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)
        return self.mat_based_loss(mat, indices_tuple, bias_score)
    
    def mat_based_loss(self, mat, indices_tuple, bias_score):
        a1, p, a2, n = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return self._compute_loss(mat, pos_mask, neg_mask, bias_score)
    
    def _compute_loss(self, mat, pos_mask, neg_mask, bias_score):
        if pos_mask.bool().any() and neg_mask.bool().any():
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                mat = -mat
            mat = mat / self.temperature
            mat_max, _ = mat.max(dim=1, keepdim=True)
            mat = mat - mat_max.detach()  # for numerical stability

            denominator = lmu.logsumexp(
                mat, keep_mask=(pos_mask + neg_mask).bool(), add_one=False, dim=1
            )
            log_prob = mat - denominator
            if bias_score != None:
                if self.use_focal_scaling:
                    p_t = torch.exp(log_prob)
                    focal_factor = (1 - p_t)**bias_score
                else:
                    focal_factor = bias_score
                log_prob = focal_factor * log_prob
                A = torch.isnan(log_prob)
                if not torch.all(~A):
                    breakpoint()

            mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
                pos_mask.sum(dim=1) + c_f.small_val(mat.dtype)
            )

            return {
                "loss": {
                    "losses": -mean_log_prob_pos,
                    "indices": c_f.torch_arange_from_size(mat),
                    "reduction_type": "element",
                }
            }
        return self.zero_losses()

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_default_distance(self):
        return CosineSimilarity()
