import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
import squidpy as sq
from tqdm import tqdm
import os
## pairwise enrichment
import squidpy as sq
from scipy import stats
import statsmodels.stats.multitest as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pertpy as pt
import scanpy as sc
import quiche
from rpy2.robjects import pandas2ri
pandas2ri.activate()

def run_milo(adata,
             n_neighbors = 100,
             design = '~condition',
             model_contrasts = 'conditionA-conditionB',
             patient_key = 'Patient_ID',
             prop = 0.1):
    """
    Runs the Milo analysis pipeline on single-cell data to detect differentially abundant neighborhoods under various conditions.
    
    Parameters:
        adata (AnnData): Annotated data matrix with single-cell data.
        n_neighbors (int): Number of nearest neighbors to consider for the neighborhood graph.
        design (str): Design formula for the statistical model.
        model_contrasts (str): Contrast specification for the model to assess differences.
        patient_key (str): Key in adata.obs used to identify patient samples.
        prop (float): Proportion of total cells used to subsample neighborhoods.
        
    Returns:
        mdata (dict): A dictionary containing modified AnnData objects after Milo analysis.
    """
    #neighborhoods are subsampled milo way
    milo = pt.tl.Milo()
    mdata = milo.load(adata)
    sc.pp.neighbors(mdata['rna'], n_neighbors = n_neighbors)
    milo.make_nhoods(mdata['rna'], prop = prop)
    mdata = quiche.tl.build_milo_graph(mdata)
    mdata = milo.count_nhoods(mdata, sample_col = patient_key)
    milo.da_nhoods(mdata,
                design=design,
                model_contrasts=model_contrasts)
    milo.annotate_nhoods(mdata, anno_col = 'cell_cluster')
    return mdata

def graphcompass_interaction(adata,
                             patient_key = 'Patient_ID',
                             labels_key = 'cell_cluster',
                             spatial_key = 'X_spatial'):
    """
    Computes the cell type interactions within neighborhoods identified in spatial transcriptomics data.

    Parameters:
        adata (AnnData): Annotated data matrix with spatial transcriptomics data.
        patient_key (str): Column in adata.obs that identifies patient or sample IDs.
        labels_key (str): Column in adata.obs used to label cell clusters or types.
        spatial_key (str): Key used to access spatial coordinates of cells in adata.obsm.
    
    Returns:
        AnnData: Updated adata object with uns['celltype_interaction'] containing interaction matrices.
    """
    sq.gr.spatial_neighbors(adata, library_key=patient_key, spatial_key = spatial_key, coord_type='generic', delaunay=True)
    adata.obs['library_id'] = adata.obs[patient_key].copy()
    cell_type_levels = adata.obs[labels_key].cat.categories

    count_list = []
    for i, name in tqdm(enumerate(adata.obs_names)):
        row, col = adata.obsp['spatial_connectivities'][i, :].nonzero()
        count = adata.obs[labels_key][col].value_counts()
        count_list.append(count)

    neighborhood_composition = pd.DataFrame(count_list, index=adata.obs_names)
    adata.uns['neighborhood_composition'] = neighborhood_composition

    zscore_list = []
    count_list = []
    celltype_names = []

    for i in adata.obs.library_id.cat.categories:
        adata_sub = adata[adata.obs.library_id == i].copy() #for each patient 
        adata_sub.obs[labels_key] = adata_sub.obs[labels_key].astype('category')
        ##compute pairwise enrichment between all cell types
        zscore, count = sq.gr.nhood_enrichment(
            adata_sub,
            cluster_key=labels_key,
            copy=True,
            show_progress_bar=False,
        )
        
        ct_labels = adata_sub.obs[labels_key].cat.categories
        del adata_sub
        
        celltype_names.append(ct_labels)
        zscore_list.append(zscore)
        count_list.append(count)

    cell_type_combinations = pd.DataFrame()

    cell_types = pd.Series(cell_type_levels)
    cell_type_combinations['cell_type'] = cell_types.repeat(len(cell_types))
    cell_type_combinations['interactor'] = np.repeat([cell_types], len(cell_type_levels), axis=0).flatten()
    cell_type_combinations['combination'] = cell_type_combinations['cell_type'].add('_' + cell_type_combinations['interactor'])
    cell_type_combinations['dummy'] = cell_type_combinations['combination'].factorize()[0]
    n_celltypes = len(adata.obs[labels_key].cat.categories)

    arr = np.array(cell_type_combinations.dummy).reshape(n_celltypes, n_celltypes)
    celltypexcelltype = pd.DataFrame(arr, index=adata.obs[labels_key].cat.categories,
                                columns=adata.obs[labels_key].cat.categories)

    df_list = []
    sample_id = adata.obs.library_id.cat.categories
    quant_data = zscore_list
    for i in range(len(quant_data)):
        df = pd.DataFrame()
        a = quant_data[i]
        upper = a[np.triu_indices(a.shape[0])] #take upper triangular portion of pairwise enrichment 
        values = np.array(upper.reshape(-1,1)) #flatten it into n x 1 array
        df['values'] = values.ravel()
        dummy_vars = celltypexcelltype.loc[celltypexcelltype.index.isin(celltype_names[i]),celltype_names[i]]
        dummy_vars = np.array(dummy_vars)
        dummy_vars = np.array(dummy_vars[np.triu_indices(dummy_vars.shape[0])].reshape(-1,1)).ravel()
        df['interaction_id'] = dummy_vars
        df.index = np.repeat(sample_id[i], df.shape[0]) 
        df_list.append(df)
        
    df_long = pd.concat(df_list) #concatenate by 0

    # replace dummy factors with cell type labels
    label_dict = {'interaction_id': dict(zip(cell_type_combinations['dummy'], cell_type_combinations['combination']))}
    df_long.replace(label_dict, inplace=True)

    # melt the matrix  
    df_wide = df_long.pivot(columns='interaction_id', values='values')
    df_wide[np.isnan(df_wide)] = 0

    adata.uns['celltype_interaction'] = df_wide
    return adata
    
def evaluate_graphcompass(adata,
                          patient_key = 'Patient_ID',
                          condition_key = 'Status',
                          ref = 'normal'):
    """
    Integrates R function evaluate_graphcompass to analyze cell type interactions across different conditions using data stored in an AnnData object.

    Parameters:
        adata (AnnData): Annotated data matrix with single-cell data.
        patient_key (str): Column in adata.obs used for patient IDs.
        condition_key (str): Column in adata.obs defining the condition of the samples (e.g., disease status).
        ref (str): Reference condition against which other conditions are compared.
    
    Returns:
        DataFrame: Results from the R function evaluate_graphcompass, typically including statistical metrics.
    """
    group = adata.obs[['library_id', patient_key, condition_key]].copy()
    group.drop_duplicates(inplace=True)
    group.set_index('library_id', inplace=True)
    group

    # condition = group.StatusMap
    condition = group[condition_key]
    patient = group[patient_key]

    interaction_mat = adata.uns['celltype_interaction']
    interaction_mat['condition'] = condition
    interaction_mat['subject_id'] = patient
    interaction_mat = pd.melt(interaction_mat, id_vars = ['condition','subject_id'])

    r = robjects.r
    r['source'](os.path.join('spatialDA', 'tools', 'evaluate.r'))
    evaluate_graphcompass_r = robjects.globalenv['evaluate_graphcompass']
    result_r = evaluate_graphcompass_r(pandas2ri.py2rpy(condition), pandas2ri.py2rpy(interaction_mat), ref)
    result = robjects.conversion.rpy2py(result_r)
    return result

def pairwise_enrichment(adata, patient_IDs, patient_key, cluster_key):
    """
    Calculates spatial neighborhood enrichment and plots the results for specified patient IDs.

    Parameters:
        adata (AnnData): Annotated data matrix with single-cell data.
        patient_IDs (list): List of patient IDs to analyze.
        patient_key (str): Key in adata.obs that identifies patient samples.
        cluster_key (str): Key in adata.obs used for clustering cell types.
    
    Returns:
        tuple: Lists of z-scores and counts of neighborhood enrichments for each patient ID.
    """
    z_score_list = []
    count_list = []
    for id in patient_IDs:
        adata_run = adata[np.isin(adata.obs[patient_key], id)].copy()
        sq.gr.spatial_neighbors(adata_run, library_key = patient_key, n_neighs=50, radius = 0.1, coord_type = 'generic')
        sq.gr.nhood_enrichment(adata_run, cluster_key=cluster_key)
        sq.pl.nhood_enrichment(adata_run, cluster_key=cluster_key, cmap = 'RdBu_r')

        z_score_list.append(adata_run.uns['cell_cluster_nhood_enrichment']['zscore'])
        count_list.append(adata_run.uns['cell_cluster_nhood_enrichment']['count'])
    return z_score_list, count_list


def compute_pairwise_significance(z_score_listA, z_score_listB, count_A, count_B, cell_types, save_directory, filename_save):
    """
    Computes and visualizes statistical differences in spatial interactions between two conditions.

    Parameters:
        z_score_listA (list): List of z-score matrices from condition A.
        z_score_listB (list): List of z-score matrices from condition B.
        count_A (list): List of count matrices corresponding to z-score matrices from condition A.
        count_B (list): List of count matrices corresponding to z-score matrices from condition B.
        cell_types (list): List of cell types considered in the analysis.
        save_directory (str): Directory path to save the output plots.
        filename_save (str): Filename to save the output plots.

    Returns:
        DataFrame: DataFrame containing log fold changes, raw p-values, and adjusted p-values for each cell type interaction.
    """

    # Example: Replace this with your list of cell types
    cell_types = ['A', 'B', 'C', 'D', 'E']

    # Create a list of combinations for cell types, including self-interactions
    cell_combinations = [(cell_types[i], cell_types[j]) for i in range(len(cell_types)) for j in range(len(cell_types))]

    Z_min = np.floor(np.min([np.min(z_score_listA), np.min(z_score_listB)]))-2
    Z_max = np.ceil(np.max([np.max(z_score_listA), np.max(z_score_listB)]))+2

    # Create a subplot for each combination of cell types
    fig, axes = plt.subplots(ncols=len(cell_combinations)//5, nrows=5, figsize=(15, 15))
    lfc_arr =[]
    pval_arr = []
    cell_type_list  = []
    for idx, (cell_type1, cell_type2) in enumerate(cell_combinations):
        row = idx // 5
        col = idx % 5
        if col >= row:  # Only plot lower triangle
            cell_type_list.append(cell_type1+'_'+cell_type2)
            # Extract the z-scores for the given cell types from each matrix
            Z_A = [matrix[cell_types.index(cell_type1), cell_types.index(cell_type2)] for matrix in z_score_listA]
            Z_B = [matrix[cell_types.index(cell_type1), cell_types.index(cell_type2)] for matrix in z_score_listB]

            C_A = [matrix[cell_types.index(cell_type1), cell_types.index(cell_type2)] for matrix in count_A]
            C_B = [matrix[cell_types.index(cell_type1), cell_types.index(cell_type2)] for matrix in count_B]
            
            # Perform Wilcoxon rank-sum test
            _, p_value = stats.ranksums(C_A, C_B)
            pval_arr.append(p_value)
            
            # Calculate log fold change
            mean_A = np.mean(C_A)
            mean_B = np.mean(C_B)
            
            log_fold_change = np.log2((mean_A / mean_B))
            lfc_arr.append(log_fold_change)
                        
            # Create a violin plot
            df = pd.DataFrame({'condition1': Z_A, 'condition2': Z_B})
            g = sns.violinplot(x='variable', y='value', data=df.melt(), ax=axes[row, col], inner='point', hue='variable', palette='Set1')
            g.set_xlabel('', fontsize=12)
            g.set_ylabel('Z-score', fontsize=12)
            g.tick_params(labelsize=12)
            g.set_ylim(Z_min, Z_max)
            axes[row, col].get_legend().remove()
            axes[row, col].set_title(f'{cell_type1}_{cell_type2}')
        else:
            # Hide axes for upper triangle
            axes[row, col].axis('off')

    # Adjust layout
    plt.tight_layout()
    if filename_save is not None:
        plt.savefig(os.path.join(save_directory, filename_save+'.pdf'), bbox_inches = 'tight')
    else:
        plt.show()
    adj_pval_arr = sm.multipletests(pval_arr, method='fdr_bh')[1] 
    scores_df = pd.DataFrame(lfc_arr, index = cell_type_list, columns = ['logFC'])
    scores_df['pval'] = pval_arr
    scores_df['adj_pval'] = adj_pval_arr
    return scores_df

    