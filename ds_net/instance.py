from ds_net.semantic import PolarOffsetSpconv
from ds_net.meanshift import PytorchMeanshift
from ds_net.modules import spconv_unet
from ds_net.modules.train_utils import load_pretrained_model
import torch

class InstanceOffset(PolarOffsetSpconv):
    def __init__(self, cfg):
        super(InstanceOffset, self).__init__(cfg)
        for param in self.parameters():
            param.requires_grad = False
        self.ins_head = getattr(spconv_unet, cfg.MODEL.INS_HEAD.NAME)(cfg)
        
    def forward(self, batch):
        coor, feature_3d = self.voxelize_spconv(batch)
        sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))    
        pred_offsets, ins_fea_list = self.ins_head(ins_fea, batch)
        return pred_offsets


class DSNet(InstanceOffset):
    def __init__(self, cfg):
        super(DSNet, self).__init__(cfg)
        for param in self.parameters():
            param.requires_grad = False
        self.pytorch_meanshift = PytorchMeanshift()
    
    def forward(self, batch, is_test=False):
        coor, feature_3d = self.voxelize_spconv(batch)
        sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))    
        pred_offsets, ins_fea_list = self.ins_head(ins_fea, batch)
        if is_test:
            sem_logits = self.sem_head(sem_fea)

            grid_ind = batch['grid']
            result = []
            for i in range(len(grid_ind)):
                result.append(sem_logits[i, :, grid_ind[i][:, 0], grid_ind[i][:, 1], grid_ind[i][:, 2]])
            semantic_classes = torch.stack(result).argmax(-2)
        else:
            semantic_classes = batch['pt_labs']
        batch['ins_fea_list'] = ins_fea_list
        regressed_centers = [offset.cuda() + xyz.cuda() for offset, xyz in zip(pred_offsets, batch['pt_cart_xyz'])]
        semantic_classes = [torch.squeeze(class_mask.to(torch.bool), dim=-1) for class_mask in semantic_classes]
        
        ins_id_preds, centers_history, sampled_centers = self.pytorch_meanshift(batch['pt_cart_xyz'], regressed_centers, semantic_classes, batch, need_cluster=is_test)

        return ins_id_preds, regressed_centers, centers_history, sampled_centers, semantic_classes
