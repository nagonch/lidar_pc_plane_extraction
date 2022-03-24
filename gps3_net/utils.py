import numpy as np
from itertools import combinations
from torch_scatter import scatter_mean
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


def convert_to_net_data(batch, clusterer, spatial_shape=[480, 360, 32]):
    gt_labels = batch['pt_labs'][0].reshape(-1)
    mask = gt_labels >= 1
    gt_labels = gt_labels[mask]
    grid = batch['grid'][0][mask]
    pt_fea = batch['pt_fea'][0][mask]
    xyz = batch['xyz'][0][mask]
    
    cluster_labels = torch.tensor(clusterer.fit_predict(xyz))
    mask2 = cluster_labels >= 0
    gt_labels = gt_labels[mask2]
    cluster_labels = cluster_labels[mask2]
    grid = grid[mask2]
    pt_fea = pt_fea[mask2]
    xyz = xyz[mask2]
    
    features = torch.cat((xyz, torch.tensor(pt_fea)), axis=1)
    indices = torch.cat((cluster_labels[None].T, torch.tensor(grid)), axis=1)
    spatial_shape = torch.tensor(spatial_shape)
    
    node_centroids = scatter_mean(xyz,
                                  cluster_labels, out=torch.zeros_like(xyz), dim=0)[:cluster_labels.max() + 1, :]
    
    return xyz, features, indices, spatial_shape, gt_labels, node_centroids, batch['vox_coor'][0], cluster_labels
