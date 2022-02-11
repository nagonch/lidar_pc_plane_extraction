import torch


def meanshift_loss(centers_gt, meanshift_history):
    meanshift_history = meanshift_history.permute(1, 0, 2, 3)
    diff = meanshift_history - centers_gt
    loss = torch.mean(torch.sum(torch.mean(torch.norm(diff, dim=-1, p=1), dim=-1), dim=0))
    
    return loss


def instance_loss(centers_gt, points_cart, offsets):
    loss = torch.mean(torch.mean(torch.norm(offsets - (centers_gt - points_cart), dim=-1, p=1), dim=-1))
    return loss