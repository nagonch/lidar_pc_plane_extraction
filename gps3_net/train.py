from .model import GPS3Net
from .utils import get_gt_edges
import hdbscan
from torch.nn import CrossEntropyLoss
import torch


clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
net = GPS3Net().cuda()

criterion = CrossEntropyLoss()

n_steps = 20
n_batches = 20

for step in range(n_steps):
    for batch in range(n_batches):
        print("batch: ", batch)
        M = 100 # num points
        N = 20 # num clusters

        points = torch.rand(M, 3)
        cluster_labels = torch.from_numpy(clusterer.fit_predict(points)).cuda() + 1

        features = torch.rand(M, 24).cuda()
        gt_labels = torch.randint(N, (M,)).cuda()
        indices = torch.cat((torch.randint(cluster_labels.max() + 1, (M,))[None].T, torch.randint(50, (M,))[None].T,
                             torch.randint(50, (M,))[None].T, torch.randint(20, (M,))[None].T), axis=1).to(torch.int32).cuda()
        spatial_shape = torch.tensor([50, 50, 20]).cuda()
        node_centroids = torch.rand(cluster_labels.max() + 1, 3).cuda()
        
        pred = net(features, indices, cluster_labels, spatial_shape, node_centroids)
        gt_graph = get_gt_edges(points, gt_labels, cluster_labels).cuda().long()
        
        loss = criterion(pred, gt_graph)
        loss.backward()
