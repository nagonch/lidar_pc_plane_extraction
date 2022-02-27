import torch
from ds_net.modules import PointNet
from ds_net.modules.main_models import PolarOffset
from ds_net.modules import spconv_unet
from ds_net.modules.config import global_cfg
from ds_net.modules.train_utils import find_match_key
from ds_net.meanshift import PytorchMeanshift
from ds_net.instance_losses import meanshift_loss
from dataset import get_kitti_filepaths
from ds_net.dataset import build_dataloader
from dataset import KittiDataset

global_cfg.DIST_TRAIN = None


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
    

class PolarOffsetSpconvMeanshift(PolarOffset):
    def __init__(self, cfg, only_offsets=False):
        super(PolarOffsetSpconvMeanshift, self).__init__(cfg, need_create_model=False)
        self.backbone = getattr(spconv_unet, cfg.MODEL.BACKBONE.NAME)(cfg)
        self.sem_head = getattr(spconv_unet, cfg.MODEL.SEM_HEAD.NAME)(cfg)
        self.vfe_model = getattr(PointNet, cfg.MODEL.VFE.NAME)(cfg)
        self.ins_head = getattr(spconv_unet, cfg.MODEL.INS_HEAD.NAME)(cfg)
        self.only_offsets = only_offsets
        if not self.only_offsets:
            self.pytorch_meanshift = PytorchMeanshift()
        

    def forward(self, batch, is_test=False):
        with torch.no_grad():
            coor, feature_3d = self.voxelize_spconv(batch)
            sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))
            sem_logits = self.sem_head(sem_fea)
        labels = []
        if is_test:
            grid_ind = batch['grid']
            for i in range(len(grid_ind)):
                labels.append(sem_logits[i, :, grid_ind[i][:, 0], grid_ind[i][:, 1], grid_ind[i][:, 2]])
            semantic_classes = torch.stack(labels).permute(0, -1, 1)
            semantic_classes = torch.argmax(semantic_classes, dim=-1)
        else:
            semantic_classes = batch['pt_labs']
 
        if self.only_offsets:
            pred_offsets, ins_fea_list = self.ins_head(ins_fea, batch)
            return pred_offsets
        
        with torch.no_grad():
            pred_offsets, ins_fea_list = self.ins_head(ins_fea, batch)
        batch['ins_fea_list'] = ins_fea_list
        regressed_centers = [offset.cuda() + xyz.cuda() for offset, xyz in zip(pred_offsets, batch['pt_cart_xyz'])]
        semantic_classes = [torch.squeeze(class_mask.to(torch.bool), dim=-1) for class_mask in semantic_classes]

        ins_id_preds, centers_history, sampled_centers = self.pytorch_meanshift(batch['pt_cart_xyz'], regressed_centers, semantic_classes, batch, need_cluster=is_test)

        return ins_id_preds, regressed_centers, centers_history, sampled_centers
    
master_weight = torch.load('/mnt/vol0/datasets/plane_extraction_model_states/selected_models/dsnet-instance-part1.pth')
ds_weight = torch.load('/mnt/vol0/datasets/plane_extraction_model_states/selected_models/dsnet_pretrain_pq_0.577.pth')
for key in ds_weight['model_state'].keys():
    master_weight['model_state'][key] = ds_weight['model_state'][key]

device = torch.device('cuda:0')
global_cfg.DATA_CONFIG.NCLASS = 2
model = PolarOffsetSpconvMeanshift(global_cfg, only_offsets=False).to(device)
load_pretrained_model(model, master_weight)

train_data, val_data = get_kitti_filepaths(0.7, return_instance=True)
train_dataloader = build_dataloader(train_data, KittiDataset, 100000, return_instance=True)

lr = 1e-3
n_steps = 100
batch_size = 1
scene_size = 100000

from datetime import datetime
from time import time
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


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
criterion = meanshift_loss

for step in tqdm(range(n_steps)):
    for i, train_batch in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
        centers = torch.stack([torch.from_numpy(item).cuda() for item in train_batch['centers']]).cuda()
        ins_id_preds, regressed_centers, centers_history, sampled_centers = model(train_batch)
        pred_error = criterion(sampled_centers, centers_history)
        # pred_error = criterion(centers, torch.stack(train_batch['pt_cart_xyz']).cuda(), torch.stack(pred_offsets).cuda())
        wandb.log({
            "pred_error": pred_error,
        })

        pred_error.backward()
        optimizer.step()
        torch.cuda.empty_cache()

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
