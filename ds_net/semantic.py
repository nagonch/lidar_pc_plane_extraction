import torch
from .modules import PointNet
from .modules.main_models import PolarOffset
from .modules import spconv_unet
from .modules.config import global_cfg
from .modules.train_utils import load_pretrained_model

global_cfg.DIST_TRAIN = None


class PolarOffsetSpconv(PolarOffset):
    def __init__(self, cfg):
        super(PolarOffsetSpconv, self).__init__(cfg, need_create_model=False)
        self.backbone = getattr(spconv_unet, cfg.MODEL.BACKBONE.NAME)(cfg)
        self.sem_head = getattr(spconv_unet, cfg.MODEL.SEM_HEAD.NAME)(cfg)
        self.vfe_model = getattr(PointNet, cfg.MODEL.VFE.NAME)(cfg)
        

    def forward(self, batch):
        coor, feature_3d = self.voxelize_spconv(batch)
        sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))
        sem_logits = self.sem_head(sem_fea)
 
        grid_ind = batch['grid']
        result = []
        for i in range(len(grid_ind)):
            result.append(sem_logits[i, :, grid_ind[i][:, 0], grid_ind[i][:, 1], grid_ind[i][:, 2]])
        return torch.stack(result).permute(0, -1, 1)


def build_model(device_name, model_state_path, n_classes):
    device = torch.device(device_name)
    global_cfg.DATA_CONFIG.NCLASS = n_classes
    model = PolarOffsetSpconv(global_cfg).to(device)
    if model_state_path:
        load_pretrained_model(model, model_state_path)
    model = model.cuda()

    return model
    