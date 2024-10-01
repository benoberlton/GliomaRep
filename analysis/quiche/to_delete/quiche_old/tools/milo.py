import scanpy as sc
import pertpy as pt
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from abc import abstractmethod
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV, ShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, balanced_accuracy_score, confusion_matrix
import scipy
from rpy2.robjects import pandas2ri
from scipy.sparse import csr_matrix
import anndata
import quiche
pandas2ri.activate()

def run_quiche(adata,
                   n_neighbors = 100,
                   neighbors_key = None,
                   design = '~condition',
                   model_contrasts = 'conditionA-conditionB',
                   anno_col = 'cell_cluster',
                   patient_key = 'Patient_ID'):
    """
    Run QUICHE analysis on spatial transcriptomics data.

    Args:
    - adata (anndata.AnnData): Anndata object containing the spatial transcriptomics data.
    - n_neighbors (int): Number of neighbors.
    - neighbors_key (str or None): Key to store neighbor information in adata.
    - design (str): Design formula for QUICHE analysis.
    - model_contrasts (str): Contrasts to test in QUICHE analysis.
    - anno_col (str): Column containing cluster annotations.
    - patient_key (str): Key to identify patients.

    Returns:
    - mdata (MultiAnnData): MultiAnnData object containing the QUICHE analysis results.
    """
    #fixed number of cells per neighborhood
    milo = pt.tl.Milo()
    mdata = milo.load(adata)
    if neighbors_key is None:
        sc.pp.neighbors(mdata['rna'], n_neighbors = n_neighbors)
    mdata['rna'].uns["nhood_neighbors_key"] = None
    mdata = quiche.tl.build_milo_graph(mdata)
    mdata = milo.count_nhoods(mdata, sample_col = patient_key)
    milo.da_nhoods(mdata,
                design=design,
                model_contrasts=model_contrasts)
    milo.annotate_nhoods(mdata, anno_col = anno_col)
    return mdata

def build_nhood_graph(mdata, basis = "X_umap", feature_key = "rna"):
    """Build graph of neighbourhoods used for visualization of DA results

    Args:
        mdata: MuData object
        basis: Name of the obsm basis to use for layout of neighbourhoods (key in `adata.obsm`). Defaults to "X_umap".
        feature_key: If input data is MuData, specify key to cell-level AnnData object. Defaults to 'rna'.

    Returns:
        - `milo_mdata['milo'].varp['nhood_connectivities']`: graph of overlap between neighbourhoods (i.e. no of shared cells)
        - `milo_mdata['milo'].var["Nhood_size"]`: number of cells in neighbourhoods
    """
    adata = mdata[feature_key]
    # # Add embedding positions
    mdata["milo"].varm["X_milo_graph"] = adata[adata.obs["nhood_ixs_refined"] == 1].obsm[basis]
    # Add nhood size
    mdata["milo"].var["Nhood_size"] = np.array(adata.obsm["nhoods"].sum(0)).flatten()
    # Add adjacency graph
    mdata["milo"].varp["nhood_connectivities"] = adata.obsm["nhoods"].T.dot(adata.obsm["nhoods"])
    mdata["milo"].varp["nhood_connectivities"].setdiag(0)
    mdata["milo"].varp["nhood_connectivities"].eliminate_zeros()
    mdata["milo"].uns["nhood"] = {
        "connectivities_key": "nhood_connectivities",
        "distances_key": "",
    }

def compute_spatialDA(adata,
                    radius = 0.1,
                    p = 2,
                    k = 10,
                    khop = None,
                    labels_key = 'cell_cluster',
                    spatial_key = 'X_spatial',
                    fov_key = 'Patient_ID',
                    patient_key = 'Patient_ID',
                    min_cell_threshold = 0,
                    phenotypic_markers = None,
                    k_sim = 100,
                    design = '~condition',
                    model_contrasts = 'conditionA-conditionB',
                    dist_key = 'euclidean_dist',
                    sketch_size = None,
                    n_jobs = -1):
    """
    Compute spatial differential abundance (DA) analysis using QUICHE.

    Args:
        adata (anndata.AnnData): Annotated data object containing spatial transcriptomics data.
        radius (float): Radius parameter for computing spatial niches.
        p (int): Parameter for computing spatial niches.
        k (int): Number of neighbors for constructing spatial similarity graph.
        khop (int or None): Number of hops for defining connectivity in the graph.
        labels_key (str): Key in adata.obs containing the cell cluster labels.
        spatial_key (str): Key in adata.obsm containing the spatial coordinates.
        fov_key (str): Key in adata.obs identifying the field of view (FOV) or patient.
        patient_key (str): Key in adata.obs identifying the patient.
        min_cell_threshold (int): Minimum number of cells required for a neighborhood to be considered.
        phenotypic_markers (list or None): List of phenotypic markers for QUICHE analysis.
        k_sim (int): Number of neighbors for QUICHE analysis.
        design (str): Design formula for QUICHE analysis.
        model_contrasts (str): Contrasts to test in QUICHE analysis.
        dist_key (str): Key for storing distance information in the adata object.
        sketch_size (int or None): Size of the sketch subsample. If None, no subsampling is performed.
        n_jobs (int): Number of parallel jobs to run. -1 means using all available CPU cores.

    Returns:
        mdata (MultiAnnData): MultiAnnData object containing the results of the spatial DA analysis.
    """
    
    adata_niche = quiche.tl.compute_spatial_niches(adata, radius = radius, p = p, k = k, khop = khop, labels_key = labels_key,
                                                spatial_key = spatial_key, fov_key = fov_key, min_cell_threshold = min_cell_threshold,  n_jobs = n_jobs)
    if sketch_size is None:
        adata_subsample = adata_niche.copy()
    else:
        _, adata_subsample = sketch(adata_niche, sample_set_key = patient_key, gamma = 1, num_subsamples = sketch_size, frequency_seed = 0, n_jobs = n_jobs)
    
    adata_subsample = quiche.tl.construct_niche_similarity_graph(adata_subsample, adata[adata_subsample.obs_names, phenotypic_markers], dist_key = dist_key, k = k_sim, n_jobs = n_jobs, filename_save = False)
    mdata = run_quiche(adata_subsample, n_neighbors = k_sim, neighbors_key = dist_key, design = design, model_contrasts = model_contrasts, patient_key = patient_key)
    return mdata

def annotate_nhoods(
    mdata,
    anno_col: str,
    feature_key = "rna",
    nlargest = 2
):
    """Assigns a categorical label to neighbourhoods, based on the most frequent label among cells in each neighbourhood. This can be useful to stratify DA testing results by cell types or samples.

    Args:
        mdata: MuData object
        anno_col: Column in adata.obs containing the cell annotations to use for nhood labelling
        feature_key: If input data is MuData, specify key to cell-level AnnData object. Defaults to 'rna'.
        nlargest (int): Number of top labels to consider.

    Returns:
        None. Adds in place:
        - `milo_mdata['milo'].var["nhood_annotation"]`: assigning a label to each nhood
        - `milo_mdata['milo'].var["nhood_annotation_frac"]` stores the fraciton of cells in the neighbourhood with the assigned label
        - `milo_mdata['milo'].varm['frac_annotation']`: stores the fraction of cells from each label in each nhood
        - `milo_mdata['milo'].uns["annotation_labels"]`: stores the column names for `milo_mdata['milo'].varm['frac_annotation']`

    """
    try:
        sample_adata = mdata["milo"]
    except KeyError:
        print(
            "milo_mdata should be a MuData object with two slots: feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
        )
        raise
    adata_run = mdata[feature_key].copy()

    # Check value is not numeric
    if pd.api.types.is_numeric_dtype(adata_run.obs[anno_col]):
        raise ValueError(
            "adata.obs[anno_col] is not of categorical type - please use milopy.utils.annotate_nhoods_continuous for continuous variables"
        )

    anno_dummies = pd.get_dummies(adata_run.obs[anno_col])
    anno_count = adata_run.obsm["nhoods"].T.dot(csr_matrix(anno_dummies.values))
    anno_count_dense = anno_count.toarray()
    anno_sum = anno_count_dense.sum(1)
    anno_frac = np.divide(anno_count_dense, anno_sum[:, np.newaxis])
    anno_frac_dataframe = pd.DataFrame(anno_frac, columns=anno_dummies.columns, index=sample_adata.var_names)

    sample_adata.varm["frac_annotation"] = anno_frac_dataframe.values
    sample_adata.uns["annotation_labels"] = anno_frac_dataframe.columns
    sample_adata.uns["annotation_obs"] = anno_col
    sample_adata.var["nhood_annotation"] = anno_frac_dataframe.apply(top_labels_with_condition, nlargest = nlargest, axis=1) #anno_frac_dataframe.idxmax(1)
    sample_adata.var["nhood_annotation_frac"] = anno_frac_dataframe.max(1)

def top_labels_with_condition(row, nlargest):
    """
    Returns a sorted string of top labels based on the specified fraction threshold.

    Args:
        row (pd.Series): Series containing label fractions.
        nlargest (int): Number of top labels to consider.

    Returns:
        sorted_labels (str): String of sorted top labels separated by '__'.
    """

    top_labels = row.nlargest(nlargest)
    top_labels = top_labels.index[top_labels > 0.10]
    sorted_labels = '__'.join(sorted(top_labels))
    return sorted_labels

class BaseLabelPropagation:
    """Class for performing label propagation

    Parameters
    W: ndarray
        adjacency matrix to compute label propagation on
    ----------

    Returns
    ----------
    """
    def __init__(self, W):
        """
        Initialize the BaseLabelPropagation object.

        Args:
            W (ndarray): Adjacency matrix to compute label propagation on.
        """
        self.W_norm = self._normalize(W)
        self.n_nodes = np.shape(W)[0]
        self.indicator_labels = None
        self.n_classes = None
        self.labeled_mask = None
        self.predictions = None

    @staticmethod
    @abstractmethod
    def _normalize(W):
        """
        Abstract method to compute row-normalized adjacency matrix.

        Args:
            W (ndarray): Adjacency matrix.

        Returns:
            ndarray: Normalized adjacency matrix.
        """
        raise NotImplementedError("_normalize must be implemented")

    @abstractmethod
    def _propagate(self):
        """
        Abstract method for label propagation.
        """
        raise NotImplementedError("_propagate must be implemented")


    def _encode(self, labels):
        """
        One-hot encode labeled data instances and zero rows corresponding to unlabeled instances.

        Args:
            labels (ndarray): Array of labels for every node.

        Returns:
            None
        """
        # Get the number of classes
        classes = np.unique(labels)
        classes = classes[classes != -1] #-1 are unlabeled nodes so we'll exclude them
        self.n_classes = np.shape(classes)[0]
        # One-hot encode labeled data instances and zero rows corresponding to unlabeled instances
        unlabeled_mask = (labels == -1)
        labels = labels.copy()
        labels[unlabeled_mask] = 0
        onehot_encoder = OneHotEncoder(sparse_output=False)
        self.indicator_labels = labels.reshape(len(labels), 1)
        self.indicator_labels = onehot_encoder.fit_transform(self.indicator_labels)
        self.indicator_labels[unlabeled_mask, 0] = 0

        self.labeled_mask = ~unlabeled_mask

    def fit(self, labels, max_iter, tol):
        """Fits semisupervised label propagation model

        Parameters
        labels: ndarray
            labels for every node, where -1 indicates unlabeled nodes
        max_iter: int (default = 10000)
            maximum number of iterations before stopping prediction
        tol: float (default = 1e-3)
            float referring to the error tolerance between runs. If unchanging, stop prediction
        """
        self._encode(labels)

        self.predictions = self.indicator_labels.copy()
        prev_predictions = np.zeros((self.n_nodes, self.n_classes), dtype = np.float)

        for i in range(max_iter):
            # Stop iterations if the system is considered at a steady state
            variation = np.abs(self.predictions - prev_predictions).sum().item()

            if variation < tol:
                print(f"The method stopped after {i} iterations, variation={variation:.4f}.")
                break

            prev_predictions = self.predictions
            self._propagate()

    def predict(self):
        """
        Predicts labels.

        Returns:
            ndarray: Predicted labels.
        """
        return self.predictions

    def predict_labels(self):
        """
        Returns
        predicted_labels: ndarray
            array of predicted labels according to the maximum probability
        predicted_scores: ndarray
            array of probability scores with dimensions n x nclasses
        uncertainty: ndarray
            array 1 - max of predictions

        ----------
        """
        predicted_labels = np.argmax(self.predictions, axis = 1)
        predicted_scores = self.predictions
        uncertainty = 1 - np.max(predicted_scores, 1)

        return predicted_labels, predicted_scores, uncertainty

class LabelPropagation(BaseLabelPropagation):
    """
    Class for performing label propagation.

    Parameters:
        W (ndarray): Adjacency matrix to compute label propagation on.
    """
    def __init__(self, W):
        """
        Initialize the LabelPropagation object.

        Args:
            W (ndarray): Adjacency matrix to compute label propagation on.
        """
        super().__init__(W)

    @staticmethod
    def _normalize(W):
        """ Computes row normalized adjacency matrix: D^-1 * W"""
        d = W.sum(axis=0).getA1()
        d = 1/d
        D = scipy.sparse.diags(d)

        return D @ W

    def _propagate(self):
        """
        Propagates labels.

        Returns:
            None
        """
        self.predictions = self.W_norm @ self.predictions

        # Put back already known labels
        self.predictions[self.labeled_mask] = self.indicator_labels[self.labeled_mask]

    def fit(self, labels, max_iter = 500, tol = 1e-3):
        """
        Fits semi-supervised label propagation model.

        Args:
            labels (ndarray): Labels for every node.
            max_iter (int): Maximum number of iterations before stopping prediction.
            tol (float): Error tolerance between runs. If unchanging, stop prediction.

        Returns:
            None
        """
        super().fit(labels, max_iter, tol)

def annotate_nhoods_label_prop(
    mdata,
    n_splits = 20,
    anno_col = None,
    feature_key = "rna",
    train_size = 0.6,
    nlargest = 2,
    label_key = 'nhood_annotation', 
):
    """Assigns a categorical label to neighbourhoods, based on the most frequent label among cells in each neighbourhood. This can be useful to stratify DA testing results by cell types or samples.

    Args:
        mdata: MuData object
        n_splits (int): Number of splits for StratifiedShuffleSplit.
        anno_col: Column in adata.obs containing the cell annotations to use for nhood labelling
        feature_key: If input data is MuData, specify key to cell-level AnnData object. Defaults to 'rna'.
        train_size (float): Fraction of samples to be used as training data.
        nlargest (int): Number of top labels to consider.
        label_key (str): Key for storing neighborhood annotations.

    Returns:
        tuple: Contains three elements:
        - mdata (MuData): The MuData object updated in place with neighbourhood annotations.
        - predicted_scores_avg (numpy array): The average prediction scores for each label across all splits. Shape is (n_samples, n_labels).
        - uncertainty (numpy array): The uncertainty associated with the predicted labels. Shape matches `predicted_scores_avg`.

    Updates the MuData object in-place, adding the following:
    - `mdata['milo'].var[label_key]`: assigns a label to each neighbourhood.
    - `mdata['milo'].var[f'{label_key}_frac']`: stores the fraction of cells in the neighbourhood with the assigned label.
    - `mdata['milo'].varm['frac_annotation']`: stores the fraction of cells from each label in each neighbourhood.
    - `mdata['milo'].uns["annotation_labels"]`: stores the column names for `mdata['milo'].varm['frac_annotation']`.
    """
    try:
        sample_adata = mdata["milo"]
    except KeyError:
        print(
            "milo_mdata should be a MuData object with two slots: feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
        )
        raise
    adata_run = mdata[feature_key].copy()

    # Check value is not numeric
    if pd.api.types.is_numeric_dtype(adata_run.obs[anno_col]):
        raise ValueError(
            "adata.obs[anno_col] is not of categorical type - please use milopy.utils.annotate_nhoods_continuous for continuous variables"
        )

    anno_dummies = pd.get_dummies(adata_run.obs[anno_col])
    anno_count = adata_run.obsm["nhoods"].T.dot(csr_matrix(anno_dummies.values))
    anno_count_dense = anno_count.toarray()
    anno_sum = anno_count_dense.sum(1)
    anno_frac = np.divide(anno_count_dense, anno_sum[:, np.newaxis])
    anno_frac_dataframe = pd.DataFrame(anno_frac, columns=anno_dummies.columns, index=sample_adata.var_names)

    sample_adata.varm["frac_annotation"] = anno_frac_dataframe.values
    sample_adata.uns["annotation_labels"] = anno_frac_dataframe.columns
    sample_adata.uns["annotation_obs"] = anno_col
    og_df = pd.DataFrame(mdata['rna'].X, columns = mdata['rna'].var_names)
    y_max = og_df.apply(top_labels_with_condition, nlargest = nlargest, axis=1) #anno_frac_dataframe.idxmax(1)
    y_max[np.isin(list(y_max.values), list(y_max.value_counts()[y_max.value_counts() < n_splits].index))] = 'unidentified'
    le = LabelEncoder()
    y = le.fit_transform(y_max).astype(int)
    # y.loc[y.index[y.value_counts() < 10]] = -1. #if barely any cells in this category, we say unconfident annotation and learn their labels based on neighbors
    n_nodes = np.shape(y)[0]
    X = mdata['rna'].X
    # X = anno_frac_dataframe.values
    sss = StratifiedShuffleSplit(n_splits = n_splits, train_size = train_size, random_state = 0)
    sss.get_n_splits(X, y = y)
    i=0
    predicted_scores_total = 0
    # sc.pp.neighbors(adata_run, n_neighbors = 50)
    for train_index, test_index in sss.split(X, y):
        print(i)
        i=i+1
        y_t = np.full(n_nodes, -1.)
        y_t[train_index] = y[train_index].copy()
        label_propagation = LabelPropagation(mdata['rna'].obsp['connectivities'])
        label_propagation.fit(y_t)
        _, predicted_scores, uncertainty = label_propagation.predict_labels()
        predicted_scores_total += predicted_scores
    predicted_scores_avg = predicted_scores_total / n_splits
    predicted_labels_avg = np.argmax(predicted_scores_avg, axis = 1)   
    predicted_labels_avg = le.inverse_transform(predicted_labels_avg)

    sample_adata.var[label_key] = predicted_labels_avg
    sample_adata.var[f'{label_key}_frac'] = anno_frac_dataframe.max(1)
    return mdata, predicted_scores_avg, uncertainty

def top_labels_with_condition(row, nlargest):
    """
    Identifies and returns a string of sorted labels from a row of data that are the largest and above a specified threshold.
    
    Parameters:
        row (pd.Series): A pandas Series containing numeric data from which to find the top labels.
        nlargest (int): The number of largest values to consider from the row.
    
    Returns:
        str: A string concatenation of sorted labels that are the largest and meet the threshold condition (> 0.10).
    """

    top_labels = row.nlargest(nlargest)
    top_labels = top_labels.index[top_labels > 0.10]
    sorted_labels = '__'.join(sorted(top_labels))
    return sorted_labels

def label_niches_abundance(mdata, feature_key = 'rna', anno_col = 'cell_cluster', nlargest = 3, annotation_key = 'original_annotations'):
    """
    Labels niches in multi-dimensional data based on abundance, updating the dataset with new annotations.
    
    Parameters:
        mdata (dict): A dictionary of AnnData objects containing multi-omics data.
        feature_key (str): Key to select the feature space within `mdata`. Default is 'rna'.
        anno_col (str): Column name in the observations of AnnData to be used for annotations.
        nlargest (int): Number of largest elements to consider in labeling.
        annotation_key (str): Key to store the resulting annotations in the variable annotations of AnnData.
    
    Returns:
        dict: Updated dictionary containing multi-omics data with new niche annotations.
    """
    feature_key = 'rna'
    anno_col = 'cell_cluster'
    sample_adata = mdata["milo"]

    adata_run = mdata[feature_key].copy()
    anno_dummies = pd.get_dummies(adata_run.obs[anno_col])
    anno_count = adata_run.obsm["nhoods"].T.dot(csr_matrix(anno_dummies.values))
    anno_count_dense = anno_count.toarray()
    anno_sum = anno_count_dense.sum(1)
    anno_frac = np.divide(anno_count_dense, anno_sum[:, np.newaxis])
    anno_frac_dataframe = pd.DataFrame(anno_frac, columns=anno_dummies.columns, index=sample_adata.var_names)
    idx_empty = np.where(anno_frac_dataframe.sum(1) == 1)[0]
    largest_group_annotation = anno_frac_dataframe.apply(top_labels_with_condition, nlargest = nlargest, axis=1) #anno_frac_dataframe.idxmax(1)
    largest_group_annotation[idx_empty] = 'empty'
    mdata['milo'].var[annotation_key] = largest_group_annotation.values
    return mdata

def percent_change_directional(before, after):
    """
    Calculates the directional percent change between two values, adjusting the direction of the change.
    
    Parameters:
        before (float): The initial value.
        after (float): The subsequent value after some changes.
    
    Returns:
        float: The directional percent change indicating an increase or decrease.
    """
    if (before < 0 and after < before) or (before > 0 and after > before):
        perc_change = abs(((after - before) / abs(before)) * 100)
    elif before == after:
        perc_change = 0
    else: 
        perc_change = -abs(((after - before) / abs(before)) * 100)
    return perc_change
    
def label_niches_aggregate(mdata, annotation_key = 'original_annotations', aggregate_key = 'mapped_annotations', lfc_pct = 15):
    """
    Aggregates niche labels based on log-fold change comparisons and updates the data annotations accordingly.
    
    Parameters:
        mdata (dict): Dictionary containing AnnData objects.
        annotation_key (str): The key in AnnData.var where initial annotations are stored.
        aggregate_key (str): The key to store the aggregated annotations in AnnData.var.
        lfc_pct (int): The log-fold change percentage threshold to determine significant changes.
    
    Returns:
        tuple: A tuple containing the updated multi-omics data dictionary and a dictionary of the new annotations.
    """
    second_round_annotations = {}
    for group in np.unique(mdata['milo'].var[annotation_key]):
        try:
            comparison_groups = np.unique([i for i in np.unique(mdata['milo'].var[annotation_key]) if np.isin(i.split('__'), group.split('__')).all()])
            idx_group = np.where(mdata['milo'].var[annotation_key] == group)[0]
            logfc_group = mdata['milo'].var.iloc[idx_group, :].loc[:, 'logFC'].mean()
            logfc_comparisons_before = []
            logfc_comparisons_after = []
            for comparison_group in comparison_groups:
                idx_comparison_before = np.where(mdata['milo'].var[annotation_key] == comparison_group)[0]
                logfc_comparisons_before.append(mdata['milo'].var.iloc[idx_comparison_before, :].loc[:, 'logFC'].mean())

                tmp_mapping = pd.DataFrame(mdata['milo'].var[annotation_key]).copy()
                tmp_mapping[np.isin(tmp_mapping[annotation_key], comparison_group)] = group
                idx_comparison_after = np.where(tmp_mapping == group)[0]
                logfc_comparisons_after.append(mdata['milo'].var.iloc[idx_comparison_after, :].loc[:, 'logFC'].mean())

            log_fc_group_list = np.repeat(logfc_group, len(logfc_comparisons_after))
            
            perc_change = [percent_change_directional(log_fc_group_list[i], logfc_comparisons_after[i]) for i in range(0, len(log_fc_group_list))]
            idx_group_comparison = np.where(comparison_groups != group)[0]
            if np.array(perc_change).max() >= lfc_pct:
                second_round_annotations[group] = comparison_groups[np.where(np.array(perc_change) == np.array(perc_change).max())[0]][0]
            elif (abs(np.array(perc_change))[idx_group_comparison] < lfc_pct).any():
                second_round_annotations[group] = comparison_groups[np.where(np.abs(np.array(perc_change)) == np.abs(np.array(perc_change)[idx_group_comparison]).min())[0]][0]
            else:
                second_round_annotations[group] = group
            
            #compare logFC of group to all comparison groups 
        except:
            second_round_annotations[group] = group #worse then keep   
    mdata['milo'].var[aggregate_key] = pd.Series(mdata['milo'].var[annotation_key]).map(second_round_annotations)
    return mdata, second_round_annotations

def get_niche_expression_ind(mdata, adata, sig_niches, nn_dict, annotation_key, cluster_key, fov_key, patient_key):
    """
    Extracts specific niche expression data from given data subsets and constructs a new AnnData object from them.
    
    Parameters:
        mdata (dict): Dictionary of multi-omics data, including AnnData objects.
        adata (AnnData): AnnData object containing detailed expression data.
        sig_niches (list): List of significant niches to be analyzed.
        nn_dict (dict): Dictionary mapping field of views (FOVs) to nearest neighbors (nn) data.
        annotation_key (str), cluster_key (str), fov_key (str), patient_key (str): Keys for various annotations and identifiers in `adata`.
    
    Returns:
        AnnData: An AnnData object containing niche-specific expression data with additional metadata.
    """
    #gives only the cell types that make up the max cell types of the niche for all of the niches
    subset_df = pd.DataFrame()
    for niche in sig_niches:
        niche_df = mdata['rna'].obs[np.isin(mdata['milo'].var[annotation_key], niche)].copy() #subset data based on niche of interest
        fov_list = np.unique(niche_df[fov_key])
        for fov in fov_list: #for every FOV that contains the niche of interest
            cells = niche_df[niche_df[fov_key] == fov].index
            adata_subset = adata[adata.obs[fov_key] == fov].copy()
            nn_subset = nn_dict[fov][np.where(np.isin(adata_subset.obs_names, cells))[0]] #find the nn of index cells that are enriched 
            for nn in nn_subset:
                idx = [i for i in range(0, len(list(adata_subset[nn].obs[cluster_key].values))) if list(adata_subset[nn].obs[cluster_key].values)[i] in niche.split('__')] #return the cell types within each niche that make up the niche
                func_markers = adata_subset[nn][idx].to_df()
                func_markers[patient_key] = adata_subset[nn][idx].obs[patient_key].values
                func_markers[fov_key] = fov
                func_markers[annotation_key] = niche
                subset_df = pd.concat([subset_df, func_markers], axis = 0)

    other_metadata = pd.merge(subset_df.reset_index().loc[:, ['index', annotation_key]], adata.obs.reset_index(), on = 'index')
    adata_runner = anndata.AnnData(subset_df.drop(columns = [patient_key, fov_key, annotation_key]))
    adata_runner.obs = other_metadata.copy()
    return adata_runner



def binarize_functional_expression(adata_func, threshold_list):
    """
    Converts continuous functional expression data into binary format based on provided thresholds.
    
    Parameters:
        adata_func (AnnData): An AnnData object containing functional expression data to be binarized.
        threshold_list (list of tuples): List of tuples containing marker names and their corresponding thresholds.
    
    Returns:
        AnnData: An AnnData object with binarized expression data.
    """
    adata_func_binary = adata_func.to_df()
    for marker, threshold in threshold_list:
        adata_func_binary[marker] = (adata_func_binary[marker].values >= threshold).astype('int')

    adata_func_binary = anndata.AnnData(adata_func)
    adata_func_binary.obs = adata_func.obs
    return adata_func_binary

def relabel_niches_celltype(adata, annotation_key, cluster_key, sig_niches):
    """
    Relabels niches by combining annotations with cell type data, focusing on significant niches.
    
    Parameters:
        adata (AnnData): AnnData object to be relabeled.
        annotation_key (str): Key for niche annotations.
        cluster_key (str): Key for cell type cluster information.
        sig_niches (list): List of significant niches to focus on.
    
    Returns:
        list: A list of newly mapped niche labels sorted according to significant niche groups.
    """
    adata.obs['niche_cell_type'] = pd.DataFrame(list(adata.obs[annotation_key].values)) + ' niche: ' + pd.DataFrame(list(adata.obs[cluster_key].values))
    niche_dict = dict(zip(list(adata.obs[annotation_key].values), list(adata.obs['niche_cell_type'].values)))
    groups = [i.split(':')[0] for i in pd.Series(sig_niches).map(niche_dict)]
    mapped_niches = list(dict.fromkeys([i for g in groups for i in np.unique(adata.obs['niche_cell_type']) if g in i]).keys())
    sorted_mapped_niches = [item for group in groups for item in mapped_niches if item.startswith(group)]
    return sorted_mapped_niches
