import numpy as np
from itertools import combinations
from torch_scatter import scatter_mean
import torch


def pyramid_index(tensor):
    n = tensor.shape[0]
    tensor = torch.Tensor(tensor)
    index = torch.tril_indices(n, n)
    index = index.T[index[0] != index[1]].T
    result = tensor[index[0], index[1]]
    
    return result

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

    return pyramid_index(l_edge)


def convert_to_net_data(batch, clusterer, spatial_shape=[480, 360, 32]):
    result = []
    
    for i in range(len(batch['grid'])):
        gt_labels = batch['pt_labs'][i].reshape(-1)
        mask = gt_labels >= 1
        gt_labels = gt_labels[mask]
        grid = batch['grid'][i][mask]
        pt_fea = batch['pt_fea'][i][mask]
        xyz = batch['xyz'][i][mask]

        cluster_labels = torch.tensor(clusterer.fit_predict(xyz))
        mask2 = cluster_labels >= 0
        gt_labels = gt_labels[mask2]
        cluster_labels = cluster_labels[mask2]
        grid = grid[mask2]
        pt_fea = pt_fea[mask2]
        xyz = xyz[mask2]

        features = torch.cat((xyz, torch.tensor(pt_fea)), axis=1)
        indices = torch.cat((cluster_labels[None].T, torch.tensor(grid)), axis=1)
        spat_shape = torch.tensor(spatial_shape)

        node_centroids = scatter_mean(xyz,
                                      cluster_labels, out=torch.zeros_like(xyz), dim=0)[:cluster_labels.max() + 1, :]
        result.append([xyz.cuda(), features.cuda(), indices.cuda(), spat_shape.cuda(),
                      gt_labels.cuda(), node_centroids.cuda(), batch['vox_coor'][i].cuda(),
                      cluster_labels.cuda()])  
    return result

def create_mapping(index, preds):
    ma = {}
    for i, j in index[preds.argmax(-1) == 1]:
        to = min(i, j).item()
        from_ = max(i, j).item()
        from_stored = ma.get(from_)
        to_stored = ma.get(to)
        if from_stored is not None and to_stored is not None:
            continue
        if from_stored is not None:
            ma[to] = from_stored
        elif to_stored is not None:
            ma[from_] = to_stored
        else:
            ma[from_] = to
    return ma