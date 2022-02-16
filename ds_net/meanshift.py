import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_cluster import fps
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MeanShift


def meanshift_cluster(shifted_pcd, valid, bandwidth=1.0):
    embedding_dim = shifted_pcd.shape[1]
    clustered_ins_ids = np.zeros(shifted_pcd.shape[0], dtype=np.int32)
    valid_shifts = shifted_pcd[valid, :].reshape(-1, embedding_dim) if valid is not None else shifted_pcd
    if valid_shifts.shape[0] == 0:
        return clustered_ins_ids

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    try:
        ms.fit(valid_shifts)
    except Exception as e:
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(valid_shifts)
        print("\nException: {}.".format(e))
        print("Disable bin_seeding.")
    labels = ms.labels_ + 1
    assert np.min(labels) > 0
    if valid is not None:
        clustered_ins_ids[valid] = labels
        return clustered_ins_ids
    else:
        return labels
    

def cluster_batch(cart_xyz_list, shift_list, valid_list):
        bs = len(cart_xyz_list)
        pred_ins_ids_list = []
        for i in range(bs):
            i_clustered_ins_ids = meanshift_cluster(shift_list[i], valid_list[i])
            pred_ins_ids_list.append(i_clustered_ins_ids)
        return pred_ins_ids_list 


def pairwise_distance(x: torch.Tensor, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


class PytorchMeanshift(nn.Module):
    def __init__(self, bandwidth=[0.2, 1.7, 3.2], iteration=4, point_num_th=10000, init_size=32):
        super(PytorchMeanshift, self).__init__()
        self.bandwidth = bandwidth
        self.iteration = iteration
        self.data_mode = "offset"
        self.shift_mode = "matrix_flat_kernel_bandwidth_weight"
        self.down_sample_mode = "xyz"
        self.point_num_th = point_num_th
        self.init_size = init_size
        
        self.learnable_bandwidth_weights_layer_list = nn.ModuleList()
        for i in range(self.iteration):
            layer = nn.Sequential(
                nn.Linear(self.init_size, self.init_size, bias=True),
                nn.BatchNorm1d(self.init_size),
                nn.ReLU(),
                nn.Linear(self.init_size, len(self.bandwidth), bias=True),
            )
            self.learnable_bandwidth_weights_layer_list.append(layer)
            
    def calc_shifted_matrix_flat_kernel_bandwidth_weight(self, X, X_fea, iter_i):
        XT = X.T
        _weights = self.learnable_bandwidth_weights_layer_list[iter_i](X_fea).view(-1, len(self.bandwidth))
        weights = torch.softmax(_weights, dim=1)
        new_X_list = []
        dist = pairwise_distance(X)
        
        for bandwidth_i in range(len(self.bandwidth)):
            K = (dist <= self.bandwidth[bandwidth_i] ** 2).float()
            D = torch.matmul(K, torch.ones([X.shape[0], 1]).cuda()).view(-1)
            _new_X = torch.matmul(XT, K) / D
            new_X_list.append(_new_X * weights[:, bandwidth_i].view(-1))
        
        new_X = torch.sum(torch.stack(new_X_list), dim=0) / torch.sum(weights, dim=1).view(-1)
        
        return new_X.T, _weights
    
    def final_cluster(self, final_X, index, data, sampled_data, valid, batch_i, batch):
        # cluster for sampled_data
        sampled_labels = cluster_batch([None], [final_X.detach().cpu().numpy()], [None])[0].reshape(-1)

        if index is not None:
            # use NN to assign ins labels to all points in data
            nbrs = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(sampled_data)
            distances, idxs = nbrs.kneighbors(data)
            labels = sampled_labels[idxs.reshape(-1)]
        else:
            labels = sampled_labels

        # generate ins labels for all things and stuff points
        clustered_ins_ids = np.zeros(valid.shape[0], dtype=np.int32)
        clustered_ins_ids[valid.detach().cpu().numpy()] = labels

        return clustered_ins_ids
    
    def down_sample(self, data):
        ratio = float(self.point_num_th) / data.shape[0]
        if ratio >= 1.0:
            return None
        index = fps(data.cuda(), torch.zeros(data.shape[0]).cuda().long(), ratio=ratio, random_start=False)
        return index
    
    def forward(self, cartesian_xyz, regressed_centers, semantic_classes, batch, need_cluster=False):
        xyz = [points[classes] for points, classes in zip(cartesian_xyz, semantic_classes)]
        centers = [points[classes] for points, classes in zip(regressed_centers, semantic_classes)]
        index = [self.down_sample(point) for point in xyz]
        sampled_xyz = [(xyz_[index_.detach().cpu().numpy()] if index_ is not None else xyz_) for xyz_, index_ in zip(xyz, index)]
        sampled_centers = [(center[index_] if index_ is not None else center) for center, index_ in zip(centers, index)]
        
        batch_size = len(xyz)
        ins_id = []
        X_history = []
        for batch_i in range(batch_size):
            X = sampled_centers[batch_i]
            if X.shape[0] <= 1 and need_cluster:
                ins_id.append(np.zeros(semantic_classes[batch_i].shape[0], dtype=np.int32))
                continue
            elif X.shape[0] == 1 and not need_cluster:
                semantic_classes[batch_i] = np.zeros(semantic_classes[batch_i].shape, dtype=int)
                X = regressed_centers[batch_i][semantic_classes[batch_i]]
                
            iter_X_list = []
            bandwidth_list = []
            bandwidth_weight = None
            X_fea = batch['ins_fea_list'][batch_i][semantic_classes[batch_i]][index[batch_i]].reshape(-1, self.init_size)
            for iter_i in range(self.iteration):
                new_X, bandwidth_weight = self.calc_shifted_matrix_flat_kernel_bandwidth_weight(X, X_fea, iter_i)
                iter_X_list.append(new_X)
                bandwidth_list.append(bandwidth_weight)
                X = new_X
            if need_cluster:
                Id = self.final_cluster(X, index[batch_i], xyz[batch_i], sampled_xyz[batch_i], semantic_classes[batch_i], batch_i, batch)
                ins_id.append(Id)
            X_history.append(torch.stack(iter_X_list))
                    
        return ins_id, X_history, sampled_centers
