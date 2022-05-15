import numpy as np


def metric(gt, pred, beta=1):
  assert gt.shape[0] == pred.shape[0]
  n_gt_clusters, n_pred_clusters = np.unique(gt).shape[0], np.unique(pred).shape[0]
  m_gt = np.zeros((n_gt_clusters, n_pred_clusters))
  m_pred = np.zeros((n_pred_clusters, n_gt_clusters))

  for i, it in enumerate(gt):
    vals, counts = np.unique(pred[gt == i], return_counts=True)
    for val, count in zip(vals, counts):
      m_gt[i][val] = count / len(pred[gt == i])

  for i, it in enumerate(pred):
    vals, counts = np.unique(gt[pred == i], return_counts=True)
    for val, count in zip(vals, counts):
      m_pred[i][val] = count / len(gt[pred == i])
  
  gt_vector = np.max(m_gt, axis=1)
  pred_vector = np.max(m_pred, axis=1)
  overcluster_factor = gt_vector.mean()
  undercluster_factor = pred_vector.mean()
  
  return overcluster_factor, undercluster_factor, (1 + beta ** 2) * (overcluster_factor * undercluster_factor) / (undercluster_factor + overcluster_factor * beta ** 2)