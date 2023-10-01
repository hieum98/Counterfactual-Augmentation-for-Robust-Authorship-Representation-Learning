
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint


class SelfAttention(nn.Module):
    """Implements Dot-Product Self-Attention as used in "Attention is all You Need".
    """
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, k, q, v):
        d_k = q.size(-1)
        scores = torch.matmul(k, q.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, v)