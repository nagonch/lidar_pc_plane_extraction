import torch


def meanshift_loss(centers_gt, meanshift_history):
    losses = []
    for gt_batch, history_batch in zip(centers_gt, meanshift_history):
        diff = history_batch - gt_batch
        loss = torch.sum(torch.mean(torch.norm(diff, dim=-1, p=1), dim=-1), dim=0)
        losses.append(loss)
    
    return torch.mean(torch.stack(losses))


def instance_loss(centers_gt, points_cart, offsets):
    loss = torch.mean(torch.mean(torch.norm(offsets - (centers_gt - points_cart), dim=-1, p=1), dim=-1))
    return loss