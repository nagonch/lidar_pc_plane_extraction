import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from .model import GPS3Net
from .utils import get_gt_edges, convert_to_net_data
import hdbscan
from torch.nn import CrossEntropyLoss
import torch
from dataset import get_kitti_filepaths as get_karla_filepaths
from ds_net.dataset import build_dataloader
from dataset import KittiDataset
from hdbscan import HDBSCAN
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from datetime import datetime
from time import time
from tqdm import tqdm

RUN_ID = datetime.fromtimestamp(time()).strftime("%d-%m-%Y--%H-%M")
spatial_shape = [120, 90, 8]
model_save_path = f'/mnt/vol0/datasets/plane_extraction_model_states/saved_models/{RUN_ID}.pth'

train_data, val_data = get_karla_filepaths(0.7)
train_dataloader = build_dataloader(train_data, KittiDataset, 100000, batch_size=1, grid_size=spatial_shape)

N_STEPS = 300
LR = 1e-3

model = GPS3Net().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
torch_lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.96)
criterion = CrossEntropyLoss()
clusterer = HDBSCAN(min_cluster_size=10)

wandb.init(project="train", entity="skoltech-plane-extraction")
wandb.run.name = f'gps3_net--karla--{RUN_ID}'
wandb.config = {
    "learning_rate": LR,
    "epochs": N_STEPS,
    "batch_size": 1,
    "scene_size": -1,
}

for step in range(N_STEPS):
    with tqdm(train_dataloader, unit="step") as tepoch:
        for i, train_batch in enumerate(tepoch):
            tepoch.set_description(f"Step {step} / {N_STEPS}")
            optimizer.zero_grad()
            (xyz, features, indices, spatial_shape,
                 gt_labels, node_centroids, vox_coor, cluster_labels) = convert_to_net_data(train_batch, clusterer, spatial_shape=spatial_shape)
            output_scores = model(features.to(torch.float32).cuda(), indices.to(torch.int32).cuda(), cluster_labels.to(torch.int32).cuda(),
                                  spatial_shape.to(torch.int32).cuda(), node_centroids.cuda())
            gt_graph = get_gt_edges(gt_labels, cluster_labels).cuda().long()
            loss = criterion(output_scores, gt_graph)
            wandb.log({
                "pred_error": loss,
            })
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
    torch_lr_scheduler.step()
    torch.save(
            {
                'epoch': step,
                'model_state': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, model_save_path.format(RUN_ID),
    )