import torch
from .modules import PointNet
from .modules.main_models import PolarOffset
from .modules import spconv_unet
from .modules.config import global_cfg
from .modules.train_utils import load_pretrained_model
from .meanshift import PytorchMeanshift


global_cfg.DIST_TRAIN = None


class PolarOffsetSpconvMeanshift(PolarOffset):
    def __init__(self, cfg):
        super(PolarOffsetSpconvMeanshift, self).__init__(cfg, need_create_model=False)
        self.backbone = getattr(spconv_unet, cfg.MODEL.BACKBONE.NAME)(cfg)
        self.sem_head = getattr(spconv_unet, cfg.MODEL.SEM_HEAD.NAME)(cfg)
        self.vfe_model = getattr(PointNet, cfg.MODEL.VFE.NAME)(cfg)
        self.ins_head = getattr(spconv_unet, cfg.MODEL.INS_HEAD.NAME)(cfg)
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
 
        pred_offsets, ins_fea_list = self.ins_head(ins_fea, batch)
        batch['ins_fea_list'] = ins_fea_list
        regressed_centers = [offset + torch.from_numpy(xyz).cuda() for offset, xyz in zip(pred_offsets, batch['pt_cart_xyz'])]

        ins_id_preds = self.pytorch_meanshift(batch['pt_cart_xyz'], regressed_centers, semantic_classes, batch, need_cluster=is_test)

        return ins_id_preds, regressed_centers

def build_model(device_name, model_state_path, n_classes):
    device = torch.device(device_name)
    global_cfg.DATA_CONFIG.NCLASS = n_classes
    model = PolarOffsetSpconvMeanshift(global_cfg).to(device)
    if model_state_path:
        load_pretrained_model(model, model_state_path)
    model = model.cuda()

    return model
