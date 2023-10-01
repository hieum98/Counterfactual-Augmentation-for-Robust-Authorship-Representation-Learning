from typing import List
import torch


def padding(tensors: List[torch.Tensor], pad_value):
    max_len_batch = max([t.size(-1) for t in tensors])
    padded = []
    for t in tensors:
        padded_tensor = torch.zeros((tensors[0].size(0), tensors[0].size(1), max_len_batch), dtype=torch.int) + pad_value
        padded_tensor[:, :, :t.size(-1)] = t
        padded.append(padded_tensor)
    return padded

