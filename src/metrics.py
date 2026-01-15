import numpy as np
from sklearn.metrics import roc_curve

def fpr(labels, anomaly_scores, percentile):
    fpr_list, tpr_list, thresholds = roc_curve(labels, anomaly_scores)
    index = np.argmax(tpr_list >= 1 - (percentile / 100))
    real_fpr = fpr_list[index]
    return real_fpr