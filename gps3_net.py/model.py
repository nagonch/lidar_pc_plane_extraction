import torch
import spconv.pytorch as spconv
from torch import nn
import hdbscan
from torch_scatter import scatter_mean
from dgl.nn import SAGEConv
import dgl
import numpy as np
from torch.autograd import Variable


def conv3x3x3(in_planes, out_planes, stride=1):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=stride,
                     padding=(1, 1, 1))

class SPConvnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = spconv.SparseSequential(
            conv3x3x3(24, 64),
            nn.ReLU(),
            conv3x3x3(64, 32),
            nn.ReLU(),
        ).cuda()
    
    def forward(self, features, indices,
                cluster_labels, spatial_shape):
        batch_size = indices[:, 0].max() + 1
        x = spconv.SparseConvTensor(features, indices,
                                    spatial_shape, batch_size)
        
        result = self.net(x).dense()
        indices = indices.long()
        feats = Variable(result[indices[:, 0], :, indices[:, 1], indices[:, 2], indices[:, 3]])
        labels = Variable(cluster_labels.repeat((32, 1)).T)
        out = Variable(torch.zeros_like(feats))
        out = scatter_mean(feats, labels, out=out, dim=0)[:labels.max() + 1, :]
        
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
            nn.Softmax(dim=-1),
        )
    
    def get_edge_features(self, cluster_features, cluster_centroids):
        cos_similarity = lambda a, b: (a * b) / (torch.norm(a) * torch.norm(b))
        n_nodes, features_shape = cluster_features.shape
        n_edges = (n_nodes ** 2 - n_nodes) // 2
        edge_features = torch.zeros(n_edges, features_shape + 3)
        n_edge = 0
        for i in range(cluster_features.shape[0]):
            for j in range(cluster_features.shape[0]):
                if i >= j:
                    continue
                edge_features[n_edge] = torch.cat((cos_similarity(cluster_features[i], cluster_features[j]),
                                    torch.abs(cluster_centroids[i]-cluster_centroids[j])))
                n_edge += 1
        return edge_features
    
    def get_graph(self, n_nodes):
        pairs = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i < j:
                    pairs.append([i, j])
        pairs = torch.from_numpy(np.array(pairs)).T
        g1, g2 = pairs
        graph = dgl.graph((g1, g2))
        
        return graph
        
    def forward(self, x, centroids):
        edge_weight = self.get_edge_features(x, centroids).cuda()
        x = torch.cat((x, centroids), axis=1)
        
        n_clusters = x.shape[0]
        graph = self.get_graph(n_clusters).to(torch.device('cuda:0'))
        n_edges = (n_clusters ** 2 - n_clusters) // 2
        x = self.conv1(graph, x, edge_weight=edge_weight)
        x = self.conv2(graph, x)
        
        edge_features = torch.zeros(n_edges, 64).cuda()
        n_edge = 0
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i < j:
                    edge_features[n_edge] = torch.cat((x[i], x[j]))
                    n_edge += 1
        x = self.mlps(edge_features)
        
        return x
    
class GPS3Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.spconvnet = SPConvnet()
        self.edgenet = EdgeNet()
    
    def forward(self, features, indices,
                cluster_labels, spatial_shape, node_centroids):
        cluster_features = self.spconvnet(features, indices, cluster_labels, spatial_shape)
        scores = self.edgenet(cluster_features, node_centroids)
        
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
