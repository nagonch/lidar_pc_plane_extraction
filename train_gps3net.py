from gps3_net.model import GPS3Net
from gps3_net.utils import get_gt_edges, convert_to_net_data
import hdbscan
from torch.nn import CrossEntropyLoss
from torch_scatter import scatter_mean
import torch
from dataset import get_kitti_filepaths as get_karla_filepaths
from ds_net.dataset import build_dataloader
from dataset import KittiDataset
from hdbscan import HDBSCAN
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
import wandb
from datetime import datetime
from time import time
from tqdm import tqdm
from IPython.display import clear_output

RUN_ID = datetime.fromtimestamp(time()).strftime("%d-%m-%Y--%H-%M")
spatial_shape = [120, 90, 8]
model_save_path = f'/mnt/vol0/datasets/plane_extraction_model_states/saved_models/{RUN_ID}.pth'
model_load_path = '/mnt/vol0/datasets/plane_extraction_model_states/saved_models/05-05-2022--02-30.pth'

train_data, val_data = get_karla_filepaths(0.7)
train_dataloader = build_dataloader(train_data, KittiDataset, 50000, batch_size=8, grid_size=spatial_shape)

clusterer = HDBSCAN(min_cluster_size=4)
N_STEPS = 150
LR = 1e-3

model = GPS3Net().cuda()
weights = torch.load(model_load_path)['model_state']
model.load_state_dict(weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
criterion = CrossEntropyLoss()
scheduler = LambdaLR(optimizer, lambda epoch: 0.9 ** epoch, verbose=True)

wandb.init(project="train", entity="skoltech-plane-extraction")
wandb.run.name = f'gps3_net--karla--{RUN_ID}'
wandb.config = {
    "learning_rate": LR,
    "epochs": N_STEPS,
    "batch_size": 1,
    "scene_size": -1,
}
wandb.watch(
    model,
    log = "gradients",
    log_freq = 20
)

for step in range(N_STEPS):
    with tqdm(train_dataloader, unit="step") as tepoch:
        for i, train_batch in enumerate(tepoch):
            try:
                batch = convert_to_net_data(train_batch, clusterer, spatial_shape=spatial_shape)
                
                tepoch.set_description(f"Step {step} / {N_STEPS}")
                optimizer.zero_grad()
                preds = model(batch)
                cluster_labels = [b[-1] for b in batch]
                gt_labels = [b[-4] for b in batch]
                gt_graphs = [get_gt_edges(gt, cl).cuda().long() for gt, cl in zip(gt_labels, cluster_labels)]
                gt_graphs = torch.cat(gt_graphs)
                loss = criterion(preds, gt_graphs)
                wandb.log({
                    "pred_error": loss,
                })
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    scheduler.step(loss)
    torch.save(
            {
                'epoch': step,
                'model_state': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, model_save_path.format(RUN_ID),
    )
