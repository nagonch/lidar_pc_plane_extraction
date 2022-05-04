import torch
import spconv.pytorch as spconv
from torch import nn
import hdbscan
from torch_scatter import scatter_mean
from dgl.nn import SAGEConv
import dgl
import numpy as np
from torch.nn.functional import normalize


def conv3x3x3(in_planes, out_planes, stride=1):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=stride,
                     padding=(1, 1, 1))

class SPConvnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = spconv.SparseSequential(
            conv3x3x3(11, 64),
            nn.ReLU(),
            conv3x3x3(64, 32),
            nn.ReLU(),
        ).cuda()
    
    def forward(self, features, indices,
                cluster_labels, spatial_shape):
        batch_size = indices[:, 0].max() + 1
        x = spconv.SparseConvTensor(features.to(torch.float32), indices.to(torch.int32),
                                    spatial_shape, batch_size)
        
        result = self.net(x).dense()
        indices = indices.long()
        feats = result[indices[:, 0], :, indices[:, 1], indices[:, 2], indices[:, 3]]
        labels = cluster_labels.repeat((32, 1)).T
        out = torch.zeros_like(feats)
        out = scatter_mean(feats, labels.to(torch.int64), out=out, dim=0)[:labels.max() + 1, :]
        
        return out
    
class EdgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(35, 64, 'mean').cuda()
        self.conv2 = SAGEConv(64, 32, 'mean').cuda()
        self.mlps = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
        )

    def pyramid_index(self, tensor):
        n = tensor.shape[0]
        index = torch.tril_indices(n, n)
        index = index.T[index[0] != index[1]].T
        result = tensor[index[0], index[1], :]
        
        return result

    def embed_handcalc(self, features, centroids):
        n_obj, _ = features.shape
        lhs = features.repeat(n_obj, 1, 1)
        rhs = features.repeat(n_obj, 1, 1).permute(1, 0, 2)
        result = normalize(lhs, dim=-1) * normalize(rhs, dim=-1)
        lhs = centroids.repeat(n_obj, 1, 1)
        rhs = centroids.repeat(n_obj, 1, 1).permute(1, 0, 2)
        centroids_dists = torch.abs(lhs - rhs)

        return self.pyramid_index(torch.cat((result, centroids_dists), dim=-1))

    def get_concat_features(self, features):
        n_obj, _ = features.shape
        lhs = features.repeat(n_obj, 1, 1)
        rhs = features.repeat(n_obj, 1, 1).permute(1, 0, 2)
        result = torch.cat((lhs, rhs), axis=-1)
        
        return self.pyramid_index(result)
    
    def get_graph(self, n_nodes):
        pairs = torch.tril_indices(n_nodes, n_nodes)
        pairs = pairs.T[pairs[0] != pairs[1]].T
        pairs = torch.from_numpy(np.array(pairs))
        g1, g2 = pairs
        graph = dgl.graph((g1, g2))
        
        return graph
        
    def forward(self, x, centroids):
        edge_weight = self.embed_handcalc(x, centroids).cuda().to(torch.float32)
        x = torch.cat((x, centroids), axis=1).double().to(torch.float32)
        
        n_clusters = x.shape[0]
        graph = self.get_graph(n_clusters).to(torch.device('cuda:0'))
        n_edges = (n_clusters ** 2 - n_clusters) // 2
        x = self.conv1(graph, x, edge_weight=edge_weight)
        x = self.conv2(graph, x)
        
        edge_features = self.get_concat_features(x)
        x = self.mlps(edge_features)
        
        return x
    
class GPS3Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.spconvnet = SPConvnet()
        self.edgenet = EdgeNet()
    
    def forward(self, x):
        scores = []
        for batch in x:
            xyz, features, indices, spatial_shape, gt_labels, node_centroids, vox_coor, cluster_labels = batch
            cluster_features = self.spconvnet(features, indices, cluster_labels, spatial_shape)
            scores.append(self.edgenet(cluster_features, node_centroids))
        
        return scores

class GPS3Net_inference():
    def __init__(self):
        self.net = GPS3Net()
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=2)

    def voselize(self, points, cluster_labels):
        raise NotImplementedError

    def predict(self, points):
        cluster_labels = torch.from_numpy(self.clusterer.fit_predict(points)).cuda() + 1
        features, indices, spatial_shape, node_centroids = self.voxelize(points, cluster_labels)
        pred = self.net(features, indices, cluster_labels, spatial_shape, node_centroids)
        
        return pred