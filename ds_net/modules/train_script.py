import os
from main_models import PolarOffsetSpconv, PolarOffset
import PointNet
from config import global_cfg
from train_utils import load_pretrained_model
from dataset import build_dataloader, global_args, build_dataloader
from evaluate_panoptic import init_eval
from tqdm import tqdm
import torch
import spconv_unet
import clustering
from lovasz_losses import lovasz_hinge
from kitti_dataset import KittiDataset, spherical_dataset, build_dataloader
from classes_functions import get_dataset_elements_list
import wandb
from time import time
import torch.nn.functional as F


SCENE_SIZE = 20000
TRAIN_TS = str(int(time()))
BATCH_SIZE = 2
LR = 3e-4
N_STEPS = 100
MODEL_SAVE_PATH = 'saved_models/{}.pth'
SAVE_PERIOD = 10
SUBSET_SIZE = int(2000 * 16 / BATCH_SIZE)


wandb.config = {
  "learning_rate": LR,
  "epochs": N_STEPS,
  "batch_size": BATCH_SIZE,
  "scene_size": SCENE_SIZE,
}

LABELS_PATH = '/mnt/vol0/datasets/kitti_dataset/sequences/{}/labels'
SCENES_PATH = '/mnt/vol0/datasets/kitti_dataset/sequences/{}/velodyne'

train_drives = ['00', '01', '02', '03', '04', '05', '06', '07']

def get_dataset_filepaths(drives):
    labels = []
    scenes = []
    for drive in drives:
        folder_scenes = [SCENES_PATH.format(drive) + '/' + scene for scene in os.listdir(SCENES_PATH.format(drive))]
        folder_labels = [LABELS_PATH.format(drive) + '/' + label for label in os.listdir(LABELS_PATH.format(drive))]
        labels.append(folder_labels)
        scenes.append(folder_scenes)
        
    labels = [label for folder in labels for label in folder]
    scenes = [scene for folder in scenes for scene in folder]
    result = []
    for scene, label in zip(scenes, labels):
        result.append([scene, label])
    return result

train_loader = build_dataloader(get_dataset_filepaths(train_drives), scene_size=SCENE_SIZE, batch_size=BATCH_SIZE)

global_cfg.DIST_TRAIN = None
rank = global_cfg.LOCAL_RANK

class SemNetwork(PolarOffset):
    def __init__(self, cfg):
        super(PolarOffset, self).__init__(cfg)
        self.backbone = getattr(spconv_unet, cfg.MODEL.BACKBONE.NAME)(cfg)
        self.sem_head = getattr(spconv_unet, cfg.MODEL.SEM_HEAD.NAME)(cfg)
        self.vfe_model = getattr(PointNet, cfg.MODEL.VFE.NAME)(cfg)
        cluster_fn_wrapper = getattr(clustering, cfg.MODEL.POST_PROCESSING.CLUSTER_ALGO)
        self.cluster_fn = cluster_fn_wrapper(cfg)
        self.is_fix_semantic = False

        self.merge_func_name = cfg.MODEL.POST_PROCESSING.MERGE_FUNC
        
    def forward(self, batch):
        coor, feature_3d = self.voxelize_spconv(batch)
        sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))
        sem_logits = self.sem_head(sem_fea)
        
        grid_ind = batch['grid']
        result = []
        for i in range(len(grid_ind)):
            result.append(sem_logits[i, :, grid_ind[i][:, 0], grid_ind[i][:, 1], grid_ind[i][:, 2]])
        return torch.stack(result).permute(0, -1, 1)
    
device = torch.device("cuda:0")
model = SemNetwork(global_cfg).to(device)
model.load_state_dict(torch.load('saved_models/1642350882.pth')['model_statue_dict'])

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch: 0.92 ** epoch, verbose=True)
criterion = torch.nn.CrossEntropyLoss()

wandb.init(project="diploma-train", entity="nagonch")

for step in range(N_STEPS):
    with tqdm(train_loader, unit="step") as tepoch:
        for i, train_batch in enumerate(tepoch):
            tepoch.set_description(f"Step {step}")
            optimizer.zero_grad()
            output_scores = model(train_batch)
            target = torch.stack(train_batch['pt_labs']).to(device)
            pred_error = criterion(output_scores.permute(0, -1, 1), torch.squeeze(target, -1))

            total_error = pred_error
            wandb.log({
                "pred_error": pred_error,
                "total_loss": total_error,
            })

            total_error.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            
            if i >= 2000:
                break

    scheduler.step()
    if True:
        torch.save(
            {
                'epoch': step,
                'model_statue_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_error,
            }, MODEL_SAVE_PATH.format(TRAIN_TS),
        )
