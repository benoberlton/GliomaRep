import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, recall_score, precision_score
import quiche as qu
import re
import itertools
import os

#evaluate precision, recall, AUPRC
def evaluate_precision_recall(mdata = None,
                                scores_df = None,
                                thresholds = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                                labels_key = 'mask_name',
                                ground_truth_key = 'ground_truth_DA',
                                group = 'immune1',
                                feature_key = 'quiche',
                                **args):
    y_true = _access_ytrue(mdata = mdata, ground_truth_key = ground_truth_key, labels_key = labels_key, group = group, feature_key = feature_key)
    precision = []
    recall = []
    for threshold in thresholds:
        y_pred = np.select([((mdata[feature_key].var['SpatialFDR'] <= threshold) & (mdata[feature_key].var[labels_key] == group))],
                            [1], default=0)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        # Calculate precision and recall
        precision.append(tp / (tp + fp) if (tp + fp) != 0 else 0)
        recall.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
    
    eval_df =  pd.DataFrame(thresholds, columns = ['sig_cutoff'])
    eval_df['precision'] = precision
    eval_df['recall'] = recall
    eval_df = eval_df.melt(id_vars = 'sig_cutoff')
    prc_auc = auc(recall, precision)
    prc_auc = pd.DataFrame([np.nan, 'AUPRC', prc_auc], index = ['sig_cutoff', 'variable', 'value']).transpose()
    eval_df = pd.concat([eval_df, prc_auc], axis = 0)

    return eval_df

def _access_ytrue(mdata = None, ground_truth_key = 'ground_truth_DA', labels_key = 'mask_name', group = 'immune1', feature_key = 'milo'):
    try:
        mdata[feature_key].var['SpatialFDR'][mdata[feature_key].var['SpatialFDR'].isna()] = 1.0
    except:
        pass
    mdata[feature_key].var[ground_truth_key] = mdata['expression'].obs[ground_truth_key]
    mdata[feature_key].var[labels_key] = mdata['expression'].obs[labels_key].values

    idx_true = np.where((mdata['expression'].obs[ground_truth_key] == 1) &(mdata['expression'].obs[labels_key] == group))[0]
    y_true = np.zeros(len(mdata['expression'].obs.index))
    y_true[idx_true] = 1
    return y_true

def plot_precision_recall(scores_df, figsize = (4,4), xlim = [-0.02, 1.02], ylim = [-0.02, 1.02], save_directory = None, filename_save = None):
    fig, ax = plt.subplots(figsize=figsize)
    recall = scores_df[scores_df['variable'] == 'recall']['value'].values
    precision = scores_df[scores_df['variable'] == 'precision']['value'].values
    prc_auc = scores_df[scores_df['variable'] == 'AUPRC']['value'][0]
    ax.plot(recall, precision, lw=2, label=f'AUPRC {prc_auc:0.2f})', color='navy', marker = '.')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc="upper right", bbox_to_anchor=(1.0,0.95))
    if filename_save is not None:
        plt.savefig(os.path.join(save_directory, filename_save + '.pdf'), bbox_inches = 'tight')

def _normalize_niche(niche):
    parts = niche.split("__")
    parts.sort()
    if len(parts) == 1:
        parts = re.sub(r'[\d\.]*_', '', parts[0])
        parts = [parts, parts]
    return "__".join(parts)

def evaluate_jaccard(scores_df = None,
                     y_true = None,
                    cell_types = None,
                    **args):
    pairwise_lcd = list(itertools.combinations_with_replacement(cell_types, 2))
    pairwise_lcd = ["__".join(combination) for combination in pairwise_lcd]
    pairwise_lcd = [_normalize_niche(niche) for niche in pairwise_lcd]
    y_pred = [_normalize_niche(niche) for niche in list(scores_df.index)]
    y_true = [_normalize_niche(niche) for niche in y_true]

    y_pred_simplified = []
    for y_pred_i in y_pred:
        for lcd_i in pairwise_lcd:
            if lcd_i in y_pred_i:
                y_pred_simplified.append(lcd_i)

    inter = len(set(y_pred_simplified).intersection(set(y_true)))
    union = len(set(y_pred_simplified).union(set(y_true)))
    jaccard_index = inter / union
    eval_df = pd.DataFrame([jaccard_index], columns = ['jaccard_index'])
    eval_df = eval_df.melt()
    return eval_df

def compute_purity_score(adata_niche,
                        annot_key = 'kmeans_cluster_10',
                        labels_key = 'mask_name',
                        fov_key = 'Patient_ID',
                        condition_key = 'condition'):

    df = adata_niche.obs.copy()
    #index for counting cells
    df['cell_count'] = 1

    count_df = df.groupby([fov_key, annot_key, labels_key]).cell_count.sum()

    total_cells = df.groupby([fov_key, annot_key]).cell_count.sum()

    #frequency of cell types per cluster per patient
    proportion_df = count_df/total_cells

    count_df = count_df.reset_index()
    proportion_df = proportion_df.reset_index()

    #frequency of cells in each condition per cluster
    total_counts = pd.DataFrame(df.groupby([condition_key, annot_key]).cell_count.sum().unstack().sum(0)).transpose()
    purity_score = df.groupby([condition_key, annot_key]).cell_count.sum().unstack() / total_counts.values

    return count_df, proportion_df, purity_score

def evaluate_purity(mdata,
                    scores_df,
                    annot_key = 'kmeans_cluster_10',
                    labels_key = 'mask_name',
                    fov_key = 'Patient_ID',
                    condition_key = 'condition',
                    feature_key = 'spatial_nhood'):
    try:
        _, _, purity_score = compute_purity_score(mdata[feature_key], annot_key = annot_key, labels_key = labels_key, fov_key = fov_key, condition_key = condition_key)
        avg_purity = purity_score.loc[:, scores_df.index].max().mean()
    except:
        avg_purity = np.nan
    eval_df = pd.DataFrame([avg_purity], columns = ['avg_purity'])
    eval_df = eval_df.melt()
    return eval_df