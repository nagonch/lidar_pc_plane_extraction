import numpy as np
from itertools import combinations
import torch


def get_gt_edges(gt_labels, overseg_labels):
    c = overseg_labels
    n_clusters = c.max() + 1
    l_edge = np.zeros((n_clusters, n_clusters))
    d = {}
    for i in set(gt_labels):
        d[i.item()] = set()
        
    offset = 2 ** 32
    s_combo = c + offset * gt_labels
    s_ = np.array([a.item() for a in set(s_combo)])
    s_gt = s_ // offset
    s_pred = s_ % offset
    
    for idx, i in enumerate(s_gt):
        d[i].add(s_pred[idx])
    
    for i_gt in d.keys():
        true_edge = [c for c in combinations(d[i_gt], 2)]
        for i, j in true_edge:
            l_edge[i][j] = 1
            l_edge[j][i] = 1
    n_clusters = l_edge.shape[0]
    edges = torch.zeros(((n_clusters ** 2 - n_clusters) // 2,))
    n = 0
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i < j:
                edges[n] = l_edge[i][j]
                n += 1
    return edges
