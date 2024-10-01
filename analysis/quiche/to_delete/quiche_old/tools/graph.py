import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import igraph as ig
import scipy
from scipy.sparse import *
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from itertools import combinations
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from scipy.spatial.distance import pdist, squareform
import pyemd
import phate
import scanpy as sc
import timeit
import anndata
import logging

def get_igraph(W = None,
               directed: bool = None):
    """Converts adjacency matrix into igraph object

    Parameters
    W: (default = None)
        adjacency matrix
    directed: bool (default = None)
        whether graph is directed or not
    ----------

    Returns
    g: ig.Graph
        graph of adjacency matrix
    ----------
    """
    sources, targets = W.nonzero()
    weights = W[sources, targets]
    if type(weights) == np.matrix:
        weights = weights.A1 #flattens 
    g = ig.Graph(directed = directed)
    g.add_vertices(np.shape(W)[0])
    g.add_edges(list(zip(sources, targets)))
    g.es['weight'] = weights  
    return g

def heat_kernel(dist = None,
                radius = 3):
    """Transforms distances into weights using heat kernel
    Parameters
    dist: np.ndarray (default = None)
        distance matrix (dimensions = cells x k)
    radius: np.int (default = 3)
        defines the per-cell bandwidth parameter (distance to the radius nn)
    ----------
    Returns
    s: np.ndarray
        array containing between cell similarity (dimensions = cells x k)
    ----------
    """         
    sigma = dist[:, [radius]]  # per cell bandwidth parameter (distance to the radius nn)
    s = np.exp(-1 * (dist**2)/ (2.*sigma**2)) # -||x_i - x_j||^2 / 2*sigma_i**2
    return s

def construct_affinity(X = None,
                        k: int = 10,
                        radius: int = 3,
                        n_pcs = None,
                        random_state: int = 0, 
                        n_jobs: int = -1):
    """Computes between cell affinity knn graph using heat kernel
    Parameters
    X: np.ndarray (default = None)
        Data (dimensions = cells x features)
    k: int (default = None)
        Number of nearest neighbors
    radius: int (default = 3)
        Neighbor to compute per cell distance for heat kernel bandwidth parameter
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    n_jobs: int (default = -1)
        Number of tasks  
    ----------
    Returns
    W: np.ndarray
        sparse symmetric matrix containing between cell similarity (dimensions = cells x cells)
    ----------
    """
    if n_pcs is not None:
        n_comp = min(n_pcs, X.shape[1])
        pca_op = PCA(n_components=n_comp, random_state = random_state)
        X_ = pca_op.fit_transform(X)
    else:
        X_ = X.copy()

    # find kNN
    knn_tree = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='euclidean', n_jobs=n_jobs).fit(X_)
    dist, nn = knn_tree.kneighbors()  # dist = cells x knn (no self interactions)

    # transform distances using heat kernel
    s = heat_kernel(dist, radius = radius) # -||x_i - x_j||^2 / 2*sigma_i**2
    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = nn.reshape(-1)
    W = scipy.sparse.csr_matrix((s.reshape(-1), (rows, cols)), shape=(X.shape[0], X.shape[0]))

    # make symmetric
    bigger = W.transpose() > W
    W = W - W.multiply(bigger) + W.transpose().multiply(bigger)
    return W

def spatial_neighbors_fov(fov, adata, radius, p, labels_key, spatial_key, fov_key):
    """
    Compute spatial neighbors for cells within a specific field of view (FOV).

    Parameters:
    -----------
    fov : str
        Identifier for the field of view (FOV) for which spatial neighbors are computed.
    adata : anndata.AnnData
        Annotated data object containing spatial information and cell cluster labels.
    radius : int
        Radius for defining spatial neighbors.
    p : int
        Minkowski distance parameter for spatial neighbor calculation.
    labels_key : str
        Key in adata.obs to retrieve cell cluster labels.
    spatial_key : str
        Key in adata.obsm to retrieve spatial coordinates.
    fov_key : str
        Key in adata.obs to retrieve field of view identifiers.

    Returns:
    --------
    prop_df : pandas.DataFrame
        DataFrame containing spatial neighbor proportions for each cell within the specified FOV.
    nn : list
        List of nearest neighbor indices for each cell within the specified FOV.

    Notes:
    ------
    This function computes spatial neighbors for cells within a specific field of view (FOV) in an AnnData object.
    It retrieves the subset of data corresponding to the specified FOV, constructs a spatial KD-tree,
    and calculates spatial neighbors for each cell within the FOV based on the specified radius and distance metric.
    The resulting spatial neighbor proportions are stored in prop_df, and the nearest neighbor indices are stored in nn.
    """
    adata_fov = adata[np.where(adata.obs[fov_key] == fov)[0], :].copy()
    spatial_kdTree = cKDTree(adata_fov.obsm[spatial_key])
    nn = spatial_kdTree.query_ball_point(adata_fov.obsm[spatial_key], r=radius, p=p)  # no self interactions

    prop_df = pd.DataFrame()
    for i in range(0, nn.shape[0]):
        counts = adata_fov.obs[labels_key].iloc[nn[i]].value_counts().copy()
        prop_df_ = pd.DataFrame(counts / counts.sum()).transpose()
        prop_df = pd.concat([prop_df, prop_df_], axis=0)
    prop_df.index = adata_fov.obs_names

    return prop_df, nn

def compute_spatial_neighbors_parallel(adata, radius=200, p=2, min_cell_threshold=5, labels_key='cell_cluster', spatial_key='X_spatial', fov_key='fov', n_jobs = -1):
    """
    Compute spatial neighbors for cells in an AnnData object in parallel.

    Parameters:
    -----------
    adata : anndata.AnnData
        Annotated data object containing spatial information and cell cluster labels.
    radius : int, optional (default=200)
        Radius for defining spatial neighbors.
    p : int, optional (default=2)
        Minkowski distance parameter for spatial neighbor calculation.
    min_cell_threshold : int, optional (default=5)
        Minimum number of cells required to compute spatial neighbors for a field of view.
    labels_key : str, optional (default='cell_cluster')
        Key in adata.obs to retrieve cell cluster labels.
    spatial_key : str, optional (default='X_spatial')
        Key in adata.obsm to retrieve spatial coordinates.
    fov_key : str, optional (default='fov')
        Key in adata.obs to retrieve field of view identifiers.
    n_jobs : int, optional (default=-1)
        Number of parallel jobs to run. Set to -1 to use all available CPU cores.

    Returns:
    --------
    prop_df : pandas.DataFrame
        DataFrame containing spatial neighbor proportions for each cell.
    nn_list : list
        List of nearest neighbor indices for each cell, ordered by field of view.

    Notes:
    ------
    This function computes spatial neighbors for cells in an AnnData object in parallel across multiple field of views (FOVs).
    It uses the multiprocessing module to distribute computations across multiple CPU cores.
    The spatial neighbor calculation is based on the radius and Minkowski distance parameter provided.
    The resulting neighbor proportions are stored in prop_df, and the nearest neighbor indices are stored in nn_list.
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif n_jobs < -1:
        n_jobs = mp.cpu_count() + 1 + n_jobs

    p = mp.Pool(n_jobs)

    fovs = adata.obs[fov_key].unique().tolist() #should preserve original order
    n_fovs = len(fovs)

    prop_df = []
    nn_list = []
    for result in tqdm(p.imap(partial(spatial_neighbors_fov, adata=adata, radius=radius, p = p, labels_key=labels_key, spatial_key=spatial_key, fov_key=fov_key), fovs), total=n_fovs, desc='computing spatial niches'):
        prop_df.append(result[0])
        nn_list.append(result[1])

    prop_df = pd.concat(prop_df, axis=0)
    prop_df = prop_df.fillna(0).copy()
    prop_df = prop_df.loc[adata.obs_names]

    return prop_df, nn_list

def spatial_niches_radius(adata,
                        radius = 200,
                        p = 2,
                        min_cell_threshold= 0,
                        labels_key = 'cell_cluster',
                        spatial_key = 'X_spatial',
                        fov_key = 'fov',
                        n_jobs = -1):
    """
    Compute spatial niches using a radius-based approach.

    Parameters:
    -----------
    adata : anndata.AnnData
        Annotated data object containing spatial information and cell cluster labels.
    radius : int, optional (default=200)
        Radius for defining spatial neighbors.
    p : int, optional (default=2)
        Minkowski distance parameter for spatial neighbor calculation.
    min_cell_threshold : int, optional (default=0)
        Minimum number of cells required to compute spatial neighbors for a field of view.
    labels_key : str, optional (default='cell_cluster')
        Key in adata.obs to retrieve cell cluster labels.
    spatial_key : str, optional (default='X_spatial')
        Key in adata.obsm to retrieve spatial coordinates.
    fov_key : str, optional (default='fov')
        Key in adata.obs to retrieve field of view identifiers.
    n_jobs : int, optional (default=-1)
        Number of parallel jobs to run. Set to -1 to use all available CPU cores.

    Returns:
    --------
    niche_df : pandas.DataFrame
        DataFrame containing spatial niche proportions for each cell.
    nn_dict : dict
        Dictionary containing nearest neighbor indices for each cell, indexed by field of view.

    Notes:
    ------
    This function computes spatial niches for cells in an AnnData object using a radius-based approach.
    It iterates over each field of view (FOV) and calculates spatial neighbors for cells within the specified radius.
    The resulting niche proportions are stored in niche_df, and the nearest neighbor indices are stored in nn_dict.
    """
    niche_df = []
    cells2remove = []
    nn_dict = {}
    for fov in np.unique(adata.obs[fov_key]):
        adata_fov = adata[np.where(adata.obs[fov_key] == fov)[0], :].copy()
        spatial_kdTree = cKDTree(adata_fov.obsm[spatial_key])
        nn = spatial_kdTree.query_ball_point(adata_fov.obsm[spatial_key], r=radius, p = p, workers = n_jobs) #no self interactions
        nn_dict[fov] = nn
        niche_df_fov = pd.DataFrame()
        for i in range(0, nn.shape[0]):
            if len(nn[i]) < min_cell_threshold:
                cells2remove.append(adata_fov.obs_names[i])
            else:
                counts = adata_fov.obs[labels_key].iloc[nn[i]].value_counts().copy()
                niche_df_fov_ = pd.DataFrame(counts / counts.sum()).transpose()
                niche_df_fov = pd.concat([niche_df_fov, niche_df_fov_], axis = 0)
        niche_df_fov.index = adata_fov.obs_names[~np.isin(adata_fov.obs_names, cells2remove)]
        niche_df.append(niche_df_fov)
    try:
        cells2remove = np.concatenate(cells2remove)    
    except:
        pass
    niche_df = pd.concat(niche_df, axis = 0)
    niche_df = niche_df.fillna(0).copy()
    niche_df = niche_df.loc[adata.obs_names[~np.isin(adata.obs_names, list(set(cells2remove)))]]
    return niche_df, nn_dict

def spatial_niches_khop(adata,
                        radius = 200,
                        p = 2,
                        k = 10,
                        khop = 3, 
                        labels_key = 'cell_cluster',
                        spatial_key = 'X_spatial',
                        fov_key = 'fov',
                        min_cell_threshold= 0,
                        n_jobs = -1):
    """
    Compute spatial niches using a k-hop neighborhood approach.

    Parameters:
    -----------
    adata : anndata.AnnData
        Annotated data object containing spatial information and cell cluster labels.
    radius : int, optional (default=200)
        Radius for defining spatial neighbors.
    p : int, optional (default=2)
        Minkowski distance parameter for spatial neighbor calculation.
    k : int, optional (default=10)
        Number of nearest neighbors to consider when building the k-hop neighborhood.
    khop : int, optional (default=3)
        Number of hops to consider in the k-hop neighborhood.
    labels_key : str, optional (default='cell_cluster')
        Key in adata.obs to retrieve cell cluster labels.
    spatial_key : str, optional (default='X_spatial')
        Key in adata.obsm to retrieve spatial coordinates.
    fov_key : str, optional (default='fov')
        Key in adata.obs to retrieve field of view identifiers.
    min_cell_threshold : int, optional (default=0)
        Minimum number of cells required to compute spatial neighbors for a field of view.
    n_jobs : int, optional (default=-1)
        Number of parallel jobs to run. Set to -1 to use all available CPU cores.

    Returns:
    --------
    niche_df : pandas.DataFrame
        DataFrame containing spatial niche proportions for each cell.
    nn_dict : dict
        Dictionary containing nearest neighbor indices for each cell, indexed by field of view.

    Notes:
    ------
    This function computes spatial niches for cells in an AnnData object using a k-hop neighborhood approach.
    It iterates over each field of view (FOV) and builds the k-hop neighborhood graph.
    The resulting niche proportions are stored in niche_df, and the nearest neighbor indices are stored in nn_dict.
    """
    niche_df = []
    cells2remove = []
    nn_dict = {}
    for fov in np.unique(adata.obs[fov_key]):
        adata_fov = adata[np.where(adata.obs[fov_key] == fov)[0], :].copy()
        if adata_fov.shape[0] < k*khop: #not enough neighbors
            logging.info(f'{fov} is less than khop neighborhood size {k*khop}. Removing FOV.')
            cells2remove.append(list(adata_fov.obs_names))
        else:
            niche_df_fov = pd.DataFrame()
            knn_kdtree = NearestNeighbors(n_neighbors= khop*k, p = p, algorithm='kd_tree', metric='euclidean', n_jobs=n_jobs).fit(adata_fov.obsm[spatial_key])
            dist, nn = knn_kdtree.kneighbors()
            nn_dict[fov] = nn
            for i in range(0, nn.shape[0]):
                niche_nn = nn[i, :][dist[i, :] < radius]
                if len(niche_nn) < min_cell_threshold:
                    cells2remove.append(adata_fov.obs_names[i])
                else:
                    counts = adata_fov.obs[labels_key].iloc[niche_nn].value_counts().copy()
                    niche_df_fov_ = pd.DataFrame(counts / counts.sum()).transpose()
                    niche_df_fov = pd.concat([niche_df_fov, niche_df_fov_], axis = 0)
            niche_df_fov.index = adata_fov.obs_names[~np.isin(adata_fov.obs_names, cells2remove)]
            niche_df.append(niche_df_fov)
    try:
        cells2remove = np.concatenate(cells2remove)    
    except:
        pass
    niche_df = pd.concat(niche_df, axis = 0)
    niche_df = niche_df.fillna(0).copy()
    niche_df = niche_df.loc[adata.obs_names[~np.isin(adata.obs_names, list(set(cells2remove)))]]
    return niche_df, nn_dict

def compute_spatial_niches(adata,
                            radius = 200,
                            p = 2,
                            k = 10,
                            khop = 3, 
                            labels_key = 'cell_cluster',
                            spatial_key = 'X_spatial',
                            fov_key = 'fov',
                            min_cell_threshold = 0, 
                            n_jobs = -1):
    """
    Compute spatial niches for cells in an AnnData object.

    Parameters:
    -----------
    adata : anndata.AnnData
        Annotated data object containing spatial information and cell cluster labels.
    radius : int, optional (default=200)
        Radius for defining spatial neighbors.
    p : int, optional (default=2)
        Minkowski distance parameter for spatial neighbor calculation.
    k : int, optional (default=10)
        Number of nearest neighbors to consider when building the k-hop neighborhood.
    khop : int, optional (default=3)
        Number of hops to consider in the k-hop neighborhood.
    labels_key : str, optional (default='cell_cluster')
        Key in adata.obs to retrieve cell cluster labels.
    spatial_key : str, optional (default='X_spatial')
        Key in adata.obsm to retrieve spatial coordinates.
    fov_key : str, optional (default='fov')
        Key in adata.obs to retrieve field of view identifiers.
    min_cell_threshold : int, optional (default=0)
        Minimum number of cells required to compute spatial neighbors for a field of view.
    n_jobs : int, optional (default=-1)
        Number of parallel jobs to run. Set to -1 to use all available CPU cores.

    Returns:
    --------
    adata_niche : anndata.AnnData
        Annotated data object containing spatial niche proportions for each cell.
    nn_dict : dict
        Dictionary containing nearest neighbor indices for each cell, indexed by field of view.

    Notes:
    ------
    This function computes spatial niches for cells in an AnnData object.
    It chooses between a radius-based approach (`spatial_niches_radius`) and a k-hop neighborhood approach (`spatial_niches_khop`)
    based on the value of `khop`. The choice of approach affects the parameter requirements and computational strategy.
    """
    
    if khop is not None:
        niche_df, nn_dict = spatial_niches_khop(adata, radius = radius, p = p, k = k, khop = khop, min_cell_threshold = min_cell_threshold, labels_key = labels_key, spatial_key = spatial_key, fov_key = fov_key, n_jobs = n_jobs)
    else:
        niche_df, nn_dict = spatial_niches_radius(adata, radius = radius, p = p, labels_key = labels_key, min_cell_threshold = min_cell_threshold, spatial_key = spatial_key, fov_key = fov_key, n_jobs = n_jobs)
    adata_niche = anndata.AnnData(niche_df)
    adata_niche.obs = adata.obs.loc[niche_df.index, :]
    return adata_niche, nn_dict


def filter_fovs(adata, patient_key, threshold):  
    """
    Filter out field of views (FOVs) from an AnnData object based on the number of cells associated with each patient.

    Parameters:
    -----------
    adata : anndata.AnnData
        Annotated data object containing field of view (FOV) information.
    patient_key : str
        Key in adata.obs to retrieve patient identifiers.
    threshold : int
        Minimum number of cells required for a patient to retain their FOVs in the filtered dataset.

    Returns:
    --------
    adata_filtered : anndata.AnnData
        Annotated data object with FOVs filtered based on the specified threshold.

    Notes:
    ------
    This function filters out field of views (FOVs) from an AnnData object based on the number of cells associated with each patient
    """
    n_niches = adata.obs[patient_key].value_counts(sort=False)
    adata = adata[~np.isin(adata.obs[patient_key], n_niches[n_niches < threshold].index)]    
    return adata

def construct_niche_similarity_graph(adata, expression_mat, dist_key = 'emd_dist', k = 100, phate_knn = 30, n_jobs = -1, precomputed = False, filename_save = None):
    """
    Constructs a similarity graph for the given AnnData object based on either precomputed distances or calculated Earth Mover's Distance (EMD) between samples.

    Parameters:
    - adata (AnnData): An AnnData object containing the dataset to analyze. It must have `.var_names` to use as clusters.
    - expression_mat (array-like or sparse matrix): The expression matrix used to compute the ground distances if not precomputed.
    - dist_key (str, optional): Key in `adata.obsp` where the distance matrix is stored or will be stored. Defaults to 'emd_dist'.
    - k (int, optional): The number of nearest neighbors to consider in the nearest neighbors graph. Defaults to 100.
    - phate_knn (int, optional): The number of nearest neighbors to use for PHATE if computing the ground distance. Defaults to 30.
    - n_jobs (int, optional): The number of parallel jobs to run. `-1` means using all processors. Defaults to -1.
    - precomputed (bool, optional): Flag to indicate whether the distance matrix is precomputed. If `True`, the function will use the existing matrix in `adata.obsp[dist_key]`. Defaults to False.
    - filename_save (str, optional): If provided, the modified AnnData object will be saved to this file. The filename should not include an extension; `.h5ad` will be appended automatically.

    Returns:
    - AnnData: The input AnnData object updated with a nearest neighbors graph stored in `adata.obsp['connectivities']` and `adata.obsp['distances']`. 
    Additional metadata about the neighbors computation is stored in `adata.uns['neighbors']`.

    This function updates the input AnnData object with new fields that represent the connectivities and distances between samples based on the specified distance metric. 
    It can handle both cases where the distance matrix is precomputed or needs to be computed using the given expression matrix and clustering information.
    """
    if (dist_key == 'emd_dist') and (precomputed == False):
        clusters = adata.var_names
        ground_dist = compute_ground_dist(adata, expression_mat, phate_knn, 'cell_cluster', clusters)
        emd_dist = compute_EMD(adata.X, ground_dist, -1, 'pyemd')
        adata.obsp[dist_key] = emd_dist
        nn = NearestNeighbors(n_neighbors = k, metric = 'precomputed', n_jobs = n_jobs).fit(adata.obsp[dist_key])
    elif (dist_key == 'emd_dist') and (precomputed == True):
        nn = NearestNeighbors(n_neighbors = k, metric = 'precomputed', n_jobs = n_jobs).fit(adata.obsp[dist_key])
    else:
        nn = NearestNeighbors(n_neighbors = k, algorithm = 'kd_tree', n_jobs = n_jobs).fit(adata.X)

    connectivities = nn.kneighbors_graph(mode = 'connectivity')
    distances = nn.kneighbors_graph(mode = 'distance')            
    adata.obsp['connectivities'] = connectivities
    adata.obsp['distances'] = distances
    adata.uns["nhood_neighbors_key"] = None
    adata.uns['neighbors'] = {'connectivities_key': 'connectivities',
                            'distances_key': 'distances',
                            'params': {'n_neighbors': k,
                                    'method': 'umap',
                                    'distance': dist_key}}
    # if filename_save is not None:
    #     adata.write(f'{filename_save}.h5ad')
    return adata

def build_milo_graph(mdata):
    """
    Build the MILO graph based on the input MuData object.

    This function constructs the MILO graph, which represents the neighborhood structure of cells.
    It defines neighborhood indices, creates a binary k-nearest neighbors (KNN) graph,
    computes the distance matrix, and stores relevant information in the input MuData object.

    Parameters:
    -----------
    mdata : MuData
        MuData object.

    Returns:
    --------
    mdata : MuData
        Updated MuData object with the MILO graph information added to the 'rna' layer.

    Notes:
    ------
    The MILO graph captures the local neighborhood relationships between cells,
    which is essential for downstream analysis and visualization tasks.
    """
    #define idxs as all cells
    mdata['rna'].obs['nhood_ixs_random'] = 1 
    mdata['rna'].obs['nhood_ixs_refined'] = 1
    #binary knn graph
    knn_graph = mdata['rna'].obsp['connectivities'].copy()
    knn_graph[knn_graph!=0] =1 #binarize
    mdata['rna'].obsm['nhoods'] = knn_graph.copy()
    #distance matrix
    knn_dists = mdata['rna'].obsp["distances"]
    knn_dists = knn_dists.max(1).toarray().ravel()
    mdata['rna'].obs["nhood_kth_distance"] = knn_dists
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
