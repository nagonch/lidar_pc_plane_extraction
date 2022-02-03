from sklearn import metrics
import numpy as np
from time import time
import torch
from datetime import datetime
import wandb


RUN_ID = datetime.fromtimestamp(time()).strftime("%d-%m-%Y--%H-%M")


def log_metrics(preds, targets, run_id=RUN_ID):
    wandb.init(project="val", entity="skoltech-plane-extraction")
    wandb.run.name = f'{run_id}'

    labels = {
        0: {0: "planar", 1:"road"},
        1: {0:"non-planar", 1:"road"},
        2: {0: "non-planar", 1: "planar"},
    }
    
    preds_ref = preds.cpu().detach()
    targets_ref = targets.cpu().detach()
    
    for drop_class_label in range(1, 3):
        preds = torch.cat(
            (preds_ref[:, :, :, :drop_class_label],
            preds_ref[:, :, :, drop_class_label + 1:]),
            dim=-1,
        )
        preds = torch.softmax(preds, dim=-1)
        preds = preds.view(-1, 2)
        targets = targets_ref.view(-1)

        eval_inds = targets != drop_class_label
        preds = preds[eval_inds]
        targets = targets[eval_inds]
        
        classes_present = [i for i in range(3) if i != drop_class_label]
        targets[targets == min(classes_present)] = 0
        targets[targets == max(classes_present)] = 1

        plot_indices = torch.randperm(targets.shape[0])[:10000]
        targets = targets[plot_indices]
        preds = preds[plot_indices]

        suffix = " vs ".join(labels[drop_class_label].values())
        
        wandb.log({f"pr_curve_{suffix}": wandb.plot.pr_curve(targets, preds,
                         labels=labels[drop_class_label])})

        wandb.log({f"roc_{suffix}": wandb.plot.roc_curve(targets, preds,
                         labels=labels[drop_class_label])})