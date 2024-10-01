import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, recall_score, precision_score
import quiche as qu

def roc_curve(y_true, mdata, thresholds):
    """
    Computes the False Positive Rate (FPR) and True Positive Rate (TPR) for various thresholds.
    
    Parameters:
        y_true (array-like): Array of true binary labels, where 1 represents the positive class.
        mdata (dict): A dictionary containing 'milo' as a key with an AnnData object as its value. The AnnData object should have 'SpatialFDR' and 'logFC' in its variables.
        thresholds (list or array-like): List or array of thresholds to evaluate FPR and TPR.
    
    Returns:
        tuple: Two lists containing the FPR and TPR for each threshold in the given list of thresholds.
    """

    fpr = []
    tpr = []

    for threshold in thresholds:

        y_pred = np.select([(mdata['milo'].var['SpatialFDR'] < threshold) & (mdata['milo'].var['logFC'] > 0),
                            (mdata['milo'].var['SpatialFDR'] < threshold) & (mdata['milo'].var['logFC'] <= 0)],
                            [1, 1], default=0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return fpr, tpr

def precision_recall_curve(y_true, mdata, thresholds):
    """
    Computes precision and recall for various thresholds used to determine the prediction labels based on 'SpatialFDR' and 'logFC'.
    
    Parameters:
        y_true (array-like): Array of true binary labels, where 1 represents the positive class.
        mdata (dict): A dictionary containing 'milo' as a key with an AnnData object as its value. The AnnData object should have 'SpatialFDR' and 'logFC' in its variables.
        thresholds (list or array-like): List or array of thresholds to evaluate precision and recall.
    
    Returns:
        tuple: Two lists containing precision and recall values for each threshold.
    """

    precision = []
    recall = []

    for threshold in thresholds:
        y_pred = np.select([(mdata['milo'].var['SpatialFDR'] < threshold) & (mdata['milo'].var['logFC'] > 0),
                            (mdata['milo'].var['SpatialFDR'] < threshold) & (mdata['milo'].var['logFC'] <= 0)],
                            [1, 1], default=0)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        # Calculate precision and recall
        precision.append(tp / (tp + fp) if (tp + fp) != 0 else 0)
        recall.append(tp / (tp + fn) if (tp + fn) != 0 else 0)

    return precision, recall

def compute_milo_recall(mdata, threshold, y_true):
    """
    Computes the recall score for a specific threshold using 'SpatialFDR', 'logFC', and a condition on 'DA_group_center'.
    
    Parameters:
        mdata (dict): A dictionary containing 'milo' as a key with an AnnData object as its value. The AnnData should have 'SpatialFDR', 'logFC', and 'DA_group_center' in its variables.
        threshold (float): The threshold value to apply to 'SpatialFDR' for defining positive predictions.
        y_true (array-like): Array of true binary labels, where 1 represents the positive class.
    
    Returns:
        float: The recall score calculated for the given threshold.
    """
    da_nhoods = np.select([(mdata['milo'].var['SpatialFDR'] < threshold) & (mdata['milo'].var['logFC'] > 0 & (mdata['milo'].var['DA_group_center'] != 'random')),
                        (mdata['milo'].var['SpatialFDR'] < threshold) & (mdata['milo'].var['logFC'] <= 0) & (mdata['milo'].var['DA_group_center'] != 'random')],
                        [1, 1], default=0)

    return recall_score(y_true, da_nhoods)

# # Calculate AUC
# idx_pred = ~mdata['milo'].var['SpatialFDR'].isnull()
# thresholds = np.quantile(mdata['milo'].var['SpatialFDR'][idx_pred].values, np.arange(1e-8, 1 - 1e-8, 0.01))
# y_true = mdata['rna'].obs['ground_labels'].values
# fpr, tpr = qu.tl.roc_curve(y_true, mdata, thresholds)
# roc_auc = auc(fpr, fpr)

# # Plot ROC curve
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate (FPR)')
# plt.ylabel('True Positive Rate (TPR)')
# plt.title('ROC Curve')
# plt.legend(loc='lower right')
# plt.show()

def compute_metrics(score_df, p = 90, y_df = None):
    """
    Computes recall, precision, and F1-score based on a percentile threshold of 'logFC'.
    
    Parameters:
        score_df (pd.DataFrame): DataFrame containing 'logFC' values from which to compute the threshold for prediction.
        p (int, optional): Percentile used to determine the threshold for 'logFC'. Default is 90.
        y_df (pd.DataFrame, optional): DataFrame with true labels ('y_true'). Must also be modified in-place to include 'y_pred'.
    
    Returns:
        tuple: Three values representing the recall, precision, and F1-score for the computed predictions.
    """
    perc  = qu.tl.compute_percentile(np.abs(score_df['logFC']), p = p)
    y_pred = list(score_df.iloc[np.where((score_df['logFC'] < -perc) | (score_df['logFC'] > perc))[0], :].index)
    y_df['y_pred'] = 0
    y_df['y_pred'][np.isin(y_df.index, y_pred)] = 1
    recall = recall_score(y_df['y_true'], y_df['y_pred'])
    precision = precision_score(y_df['y_true'], y_df['y_pred'])
    f1 = f1_score(y_df['y_true'], y_df['y_pred'])
    return recall, precision, f1