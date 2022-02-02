from sklearn import metrics
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from time import time
from datetime import datetie


RUN_ID = datetime.fromtimestamp(time()).strftime("%d-%m-%Y--%H-%M")


def process_cm(cm):
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]

    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    fpr = fp/(fp+tn)
    
    return recall, precision, fpr


def calculate_metrics(preds_probas, targets, threshold_steps=200):
    recalls, precisions, fprs = [], [], []
    recalls_no_road, precisions_no_road, fprs_no_road = [], [], []
    
    for threshold in tqdm(np.linspace(0, 1, threshold_steps)):
        pred = (preds_probas > threshold).astype(int)
        drop_indices = (pred == 1) & (targets == 2)
        preds_no_road = pred[~drop_indices]
        targets_no_road = targets[~drop_indices]
        
        targets = np.clip(targets, 0, 1).astype(int)
        targets_no_road = np.clip(targets_no_road, 0, 1).astype(int)
        
        cm = metrics.confusion_matrix(targets, pred)
        cm_no_road = metrics.confusion_matrix(targets_no_road, preds_no_road)
        
        recall, precision, fpr = process_cm(cm)
        recall_no_road, precision_no_road, fpr_no_road = process_cm(cm_no_road)
        
        recalls.append(recall)
        precisions.append(precision)
        fprs.append(fpr)
        recalls_no_road.append(recall_no_road)
        precisions_no_road.append(precision_no_road)
        fprs_no_road.append(fpr_no_road)
    
    tprs = recalls
    tprs_no_road = recalls_no_road
    
    auc = metrics.auc(fprs, tprs)
    auc_no_road = metrics.auc(fprs_no_road, tprs_no_road)
    
    return (
        recalls,
        precisions,
        fprs,
        tprs,
        auc,
        recalls_no_road,
        precisions_no_road,
        fprs_no_road,
        tprs_no_road,
        auc_no_road,
    )


def log_metrics(preds_probas, targets, threshold_steps=200):
    wandb.init(project="val", entity="skoltech-plane-extraction")
    wandb.run.name = f'{RUN_ID}'

    preds_probas = preds_probas.view(-1).cpu().detach().numpy()
    targets = targets.view(-1).cpu().detach().numpy()
    (
        recalls,
        precisions,
        fprs,
        tprs,
        auc,
        recalls_no_road,
        precisions_no_road,
        fprs_no_road,
        tprs_no_road,
        auc_no_road,
    ) = calculate_metrics(preds_probas, targets, threshold_steps=threshold_steps)

    data = [[x, y] for (x, y) in zip(recalls, precisions)]
    table = wandb.Table(data=data, columns = ["recall", "precision"])
    wandb.log({"precision_recall" : wandb.plot.line(table, "recall", "precision",
            title=f"Precision vs Recall")})
    
    data = [[x, y] for (x, y) in zip(fprs, tprs)]
    table = wandb.Table(data=data, columns = ["fpr", "tpr"])
    wandb.log({"roc_auc" : wandb.plot.line(table, "fpr", "tpr",
            title=f"ROC Curve, auc={auc}")})
    
    data = [[x, y] for (x, y) in zip(recalls_no_road, precisions_no_road)]
    table = wandb.Table(data=data, columns = ["recall_no_road", "precision_no_road"])
    wandb.log({"precision_recall_no_road" : wandb.plot.line(table, "recall_no_road", "precision_no_road",
            title=f"Precision vs Recall")})
    
    data = [[x, y] for (x, y) in zip(fprs_no_road, tprs_no_road)]
    table = wandb.Table(data=data, columns = ["fpr_no_road", "tpr_no_road"])
    wandb.log({"roc_auc_no_road" : wandb.plot.line(table, "fpr_no_road", "tpr_no_road",
            title=f"ROC Curve, auc={auc_no_road}")})


def plot_metrics(preds_probas, targets, threshold_steps=200):
    preds_probas = preds_probas.view(-1).cpu().detach().numpy()
    targets = targets.view(-1).cpu().detach().numpy()
    (
        recalls,
        precisions,
        fprs,
        tprs,
        auc,
        recalls_no_road,
        precisions_no_road,
        fprs_no_road,
        tprs_no_road,
        auc_no_road,
    ) = calculate_metrics(preds_probas, targets, threshold_steps=threshold_steps)

    plt.plot(fprs, tprs, label=f"AUC = {np.round(auc, 2)}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig('val_result/roc.png')
    plt.close()

    plt.plot(fprs_no_road, tprs_no_road, label=f"AUC = {np.round(auc_no_road, 2)}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic (no road)")
    plt.legend(loc="lower right")
    plt.savefig('val_result/roc_no_road.png')
    plt.close()

    plt.plot(recalls, precisions)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision recall curve")
    plt.savefig('val_result/pr.png')
    plt.close()

    plt.plot(recalls_no_road, precisions_no_road)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision recall curve (no road)")
    plt.savefig('val_result/pr_no_road.png')
    plt.close()

    print("Find val results in val_result folder")
