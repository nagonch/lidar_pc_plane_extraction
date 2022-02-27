from ds_net.instance import InstanceOffset, DSNet
import argparse
from time import time
from typing import Any
from logging_funtions import log_args
from dataset import get_filepaths
import torch
from model_hub import get_model
import wandb
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from tqdm import tqdm
from metrics.semantic import log_metrics
import torch.nn.functional as F
from datetime import datetime
from dataset import get_kitti_filepaths
from ds_net.dataset import build_dataloader
from dataset import KittiDataset
from ds_net.semantic import build_model
import numpy as np
from ds_net.modules.config import global_cfg
from ds_net.modules.train_utils import load_pretrained_model
from datetime import datetime
from time import time
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from ds_net.instance_losses import instance_loss
from ds_net.modules.train_utils import find_match_key


def load_pretrained_model(model, checkpoint, to_cpu=False):
    loc_type = torch.device('cpu') if to_cpu else None
    if checkpoint.get('model_state', None) is not None:
        checkpoint = checkpoint.get('model_state')
    elif checkpoint.get('model_statue_dict', None) is not None:
        checkpoint = checkpoint.get('model_statue_dict') # don't ask
        
    update_model_state = {}
    for key, val in checkpoint.items():
        match_key = find_match_key(key, model.state_dict())
        if match_key is None:
            print("Cannot find a matched key for {}".format(key))
            continue
        if model.state_dict()[match_key].shape == checkpoint[key].shape:
            update_model_state[match_key] = val

    state_dict = model.state_dict()
    state_dict.update(update_model_state)
    model.load_state_dict(state_dict)


device = torch.device('cuda:0')

master_weight = torch.load('/mnt/vol0/datasets/plane_extraction_model_states/saved_models/semantic-3class.pth')
init_w = master_weight['model_state']['fea_compression.0.weight']
offset_weight = torch.load('/mnt/vol0/datasets/plane_extraction_model_states/selected_models/offset_pretrain_pq_0.564.pth')
for key in offset_weight['model_state'].keys():
    if not key in master_weight['model_state'].keys():
        master_weight['model_state'][key] = offset_weight['model_state'][key]
assert (master_weight['model_state']['fea_compression.0.weight'] == init_w).all().item()
    
train_data, val_data = get_kitti_filepaths(0.7)
train_dataloader = build_dataloader(train_data, KittiDataset, 100000, return_instance=True)

model = InstanceOffset(global_cfg).to(device)
global_cfg.DATA_CONFIG.NCLASS = 2
load_pretrained_model(model, master_weight)
lr = 1e-3
n_steps = 100
batch_size = 1
scene_size = 100000

RUN_ID = datetime.fromtimestamp(time()).strftime("%d-%m-%Y--%H-%M")
model_save_path = f'/mnt/vol0/datasets/plane_extraction_model_states/saved_models/{RUN_ID}.pth'
wandb.init(project="train", entity="skoltech-plane-extraction")
wandb.run.name = f'ds-net-offset--kitti--{RUN_ID}'
wandb.config = {
    "learning_rate": lr,
    "epochs": n_steps,
    "batch_size": batch_size,
    "scene_size": scene_size,
}

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
torch_lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.98)
criterion = instance_loss

for step in range(n_steps):
    for i, train_batch in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
        centers = torch.stack([torch.from_numpy(item).cuda() for item in train_batch['centers']]).cuda()
        pred_offsets = model(train_batch)
        pred_error = criterion(centers, torch.stack(train_batch['pt_cart_xyz']).cuda(), torch.stack(pred_offsets).cuda())
        wandb.log({
            "pred_error": pred_error,
        })

        pred_error.backward()
        optimizer.step()
        torch.cuda.empty_cache()

    for p in model.parameters():
        break
    assert (p == init_w).all().item()
    torch_lr_scheduler.step()
    torch.save(
        {
            'epoch': step,
            'model_state': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': pred_error,
        }, model_save_path.format(RUN_ID),
    )
print("Training done")
print(f"State saved at {model_save_path.format(RUN_ID)}")
