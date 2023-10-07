import faiss
import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

def compute_metrics(queries, targets) -> dict:
    query_authors = []
    q_list = []
    for item in queries:
        for i in range(item['ground_truth'].size(0)):
            query_authors.append(item['ground_truth'][i])
            q_list.append(item['embedding'][i])
    query_authors = torch.tensor(query_authors).cpu().numpy()
    q_list = torch.cat(q_list, dim=0).cpu().numpy()

    target_authors = []
    t_list = []
    for item in targets:
        for i in range(item['ground_truth'].size(0)):
            target_authors.append(item['ground_truth'][i])
            t_list.append(item['embedding'][i])
    target_authors = torch.tensor(target_authors).cpu().numpy()
    t_list = torch.cat(t_list, dim=0).cpu().numpy()
    
    metric_scores = {}
    metric_scores.update(ranking(q_list, t_list, query_authors, target_authors))
    
    return metric_scores

def ranking(queries, 
            targets,
            query_authors, 
            target_authors, 
            metric='cosine', 
):
    num_queries = len(query_authors)
    ranks = np.zeros((num_queries), dtype=np.float32)
    reciprocal_ranks = np.zeros((num_queries), dtype=np.float32)
    
    queries = queries.copy().astype(np.float32)
    targets = targets.copy().astype(np.float32)
    # distances = pairwise_distances(queries, Y=targets, metric=metric, n_jobs=1)
    dim = int(queries[0].size)
    faiss.normalize_L2(targets)
    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)
    assert len(target_authors) == targets.shape[0], f"length target authors: {len(target_authors)} target shape: {targets.shape}"
    index.add_with_ids(targets, np.array(range(len(target_authors))).astype(np.int64))
    faiss.normalize_L2(queries)
    D, I = index.search(queries, k=1000)
    for i in range(num_queries):
        sorted_target_authors = target_authors[I[i]]
        r = np.where(sorted_target_authors == query_authors[i])[0]
        
        if len(r) == 0:
            r = 10000
        else:
            r = min(r)
        ranks[i] = r
        reciprocal_ranks[i] = 1.0 / float(ranks[i] + 1)
        
    return_dict = {
        'R@8': np.sum(np.less_equal(ranks, 8)) / np.float32(num_queries),
        'R@16': np.sum(np.less_equal(ranks, 16)) / np.float32(num_queries),
        'R@32': np.sum(np.less_equal(ranks, 32)) / np.float32(num_queries),
        'R@64': np.sum(np.less_equal(ranks, 64)) / np.float32(num_queries),
        'MRR': np.mean(reciprocal_ranks)
    }

    return return_dict
