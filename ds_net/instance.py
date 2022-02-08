import torch
from .modules import PointNet
from .modules.main_models import PolarOffset
from .modules import spconv_unet
from .modules.config import global_cfg
from .modules.train_utils import load_pretrained_model
from .meanshift import PytorchMeanshift


global_cfg.DIST_TRAIN = None


class PolarOffsetSpconv(PolarOffset):
    def __init__(self, cfg, fix_semantic=True):
        super(PolarOffsetSpconv, self).__init__(cfg, need_create_model=False)
        self.backbone = getattr(spconv_unet, cfg.MODEL.BACKBONE.NAME)(cfg)
        self.sem_head = getattr(spconv_unet, cfg.MODEL.SEM_HEAD.NAME)(cfg)
        self.vfe_model = getattr(PointNet, cfg.MODEL.VFE.NAME)(cfg)
        self.ins_head = getattr(spconv_unet, cfg.MODEL.INS_HEAD.NAME)(cfg)
        self.pytorch_meanshift = PytorchMeanshift(cfg, self.ins_loss, self.cluster_fn)
        self.fix_semantic = fix_semantic
        

    def forward(self, batch, is_test=False):
        if self.fix_semantic:
            with torch.no_grad():
                coor, feature_3d = self.voxelize_spconv(batch)
                sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))
                sem_logits = self.sem_head(sem_fea)
        else:
            coor, feature_3d = self.voxelize_spconv(batch)
            sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))
            sem_logits = self.sem_head(sem_fea)
 
        pred_offsets, ins_fea_list = self.ins_head(ins_fea, batch)
        batch['ins_fea_list'] = ins_fea_list
        embedding = [offset + torch.from_numpy(xyz).cuda() for offset, xyz in zip(pred_offsets, batch['pt_cart_xyz'])]
        valid = batch['pt_valid']
        
        if is_test:
            pt_sem_preds = self.calc_sem_label(sem_logits, batch, need_add_one=False)
            valid = []
            for i in range(len(batch['grid'])):
                valid.append(np.isin(pt_sem_preds[i], valid_xentropy_ids).reshape(-1))

        pt_ins_ids_preds, meanshift_loss, bandwidth_weight_summary = self.pytorch_meanshift(batch['pt_cart_xyz'], embedding, valid, batch, need_cluster=is_test)
        grid_ind = batch['grid']
        result = []
        for i in range(len(grid_ind)):
            result.append(sem_logits[i, :, grid_ind[i][:, 0], grid_ind[i][:, 1], grid_ind[i][:, 2]])
        return torch.stack(result).permute(0, -1, 1)
