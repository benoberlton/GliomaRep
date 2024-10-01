import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import random
import os
import anndata
from sketchKH import *
from scipy.spatial.distance import cdist
import pertpy as pt
import scanpy as sc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import squidpy as sq
from scipy import stats
import statsmodels.stats.multitest as sm
from rpy2.robjects import pandas2ri
import quiche as qu
pandas2ri.activate()

def plot_grid(df):
    """
    Plot a grid on a scatter plot based on given DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing 'x' and 'y' coordinates for scatter plot.

    Returns:
        None
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    # Scatter plot of x-y points
    axes.scatter(x = df['x'], y = df['y'], c='#D5859B', alpha=0.5)

    # Add grid lines
    num_grids_x = 8  # Adjust the number of grids in x direction
    num_grids_y = 5  # Adjust the number of grids in y direction

    # Calculate grid size
    grid_size_x = 1 / num_grids_x
    grid_size_y = 1 / num_grids_y

    # Plot vertical grid lines
    for i in range(1, num_grids_x):
        axes.axvline(i * grid_size_x, color='k', linestyle='--', linewidth=0.5)

    # Plot horizontal grid lines
    for i in range(1, num_grids_y):
        axes.axhline(i * grid_size_y, color='k', linestyle='--', linewidth=0.5)

    axes.tick_params(labelsize=10)
    # Set axis labels and legend
    plt.xlabel('X', fontsize = 12)
    plt.ylabel('Y', fontsize = 12)
    plt.xlim(0, 1)
    plt.ylim(0,1)
    # Show the plot
    plt.show()

def simulate_spatial(num_grids_x = 5, num_grids_y = 5, n_regions = 4, n_niches = 1000, da_vec = ['A', 'C', 'E'], seed = 0,
                     colors_dict = {'A': '#B46CDA','B': '#78CE8B', 'C': '#FF8595', 'D': '#1885F2', 'E': '#D78F09'}, hex = '#e41a1c',
                     fc_dict = {'A': 1,'B': 3, 'C': 5, 'D': 1.5, 'E': 2}, show_grid = False,save_directory = None, filename_save = None):
    """
    Simulate spatial distribution of points with specified parameters.

    Parameters:
        num_grids_x (int): Number of grid cells along the x-axis.
        num_grids_y (int): Number of grid cells along the y-axis.
        n_regions (int): Number of regions to simulate.
        n_niches (int): Total number of points to distribute.
        da_vec (list): List of cell types to distribute in each region.
        seed (int): Seed for random number generation.
        colors_dict (dict): Dictionary mapping cell types to colors.
        hex (str): Color code for grid lines.
        fc_dict (dict): Dictionary mapping cell types to fold change values.
        show_grid (bool): Whether to show grid lines on the plot.
        save_directory (str or None): Directory to save the plot. If None, plot is not saved.
        filename_save (str or None): Filename for the saved plot.

    Returns:
        DataFrame: DataFrame containing simulated spatial data.
        dict: Dictionary containing simulation parameters.
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    # Create a sample DataFrame with x, y, and fold_change columns
    np.random.seed(seed)
    data = {
        'x': np.random.rand(n_niches),
        'y': np.random.rand(n_niches)}
    
    n_niches_cell_type = int(n_niches/5)
    scale_dict = {'A': n_niches_cell_type, 'B': n_niches_cell_type, 'C': n_niches_cell_type, 'D': n_niches_cell_type, 'E': n_niches_cell_type}

    df = pd.DataFrame(data)

    # Calculate grid size
    grid_size_x = 1 / num_grids_x
    grid_size_y = 1 / num_grids_y

    if show_grid == True:
    # # Plot vertical grid lines
        for i in range(1, num_grids_x):
            plt.axvline(i * grid_size_x, color='gray', linestyle='--', linewidth=0.5)

        # Plot horizontal grid lines
        for i in range(1, num_grids_y):
            plt.axhline(i * grid_size_y, color='gray', linestyle='--', linewidth=0.5)

    # Initialize the 'label' column
    df['group'] = np.nan
    df['group'] = df['group'].astype('object')
    
    df['DA_group'] = 'random'
    df['DA_group'] = df['DA_group'].astype('object')

    df['DA_group_center'] = 'random'
    df['DA_group_center'] = df['DA_group_center'].astype('object')

    available_grids = set([(x, y) for x in range(1, num_grids_x + 1) for y in range(1, num_grids_y + 1)])
    grids = []
    while len(grids) < n_regions:
        # Randomly select a grid cell
        grid_ = random.sample(available_grids, 1)[0]

        # Check if the selected grid is not adjacent to any of the previously selected grids
        if all((abs(grid_[0] - x) > 1 or abs(grid_[1] - y) > 1) for (x, y) in grids):
            grids.append(grid_)

    # grids = random.sample(list(available_grids), n_regions)
    
    selected_grid_list = []
    for grid in grids:
        # Randomly select a grid cell
        selected_grid_x = grid[0]
        selected_grid_y = grid[1]
        selected_grid_list.append((grid[0], grid[1]))

        # Select all cells in the chosen grid cell
        selected_locations = df[
            (df['x'] >= (selected_grid_x - 1) * grid_size_x) & (df['x'] < selected_grid_x * grid_size_x) &
            (df['y'] >= (selected_grid_y - 1) * grid_size_y) & (df['y'] < selected_grid_y * grid_size_y)
        ].index

        # Reassign labels for the selected locations
        df.loc[selected_locations, 'group'] = np.random.choice(da_vec, size=len(selected_locations))
        df.loc[selected_locations, 'DA_group'] = '_'.join(da_vec)
    
            # Find the centroid of the selected grid
        centroid = (
            (selected_grid_x - 1 + selected_grid_x) * 0.5 * grid_size_x,
            (selected_grid_y - 1 + selected_grid_y) * 0.5 * grid_size_y
        )

        # Calculate distances from each cell to the centroid
        distances = cdist(df.loc[selected_locations, ['x', 'y']], [centroid])

        # Find the indices of the 5 closest cells
        closest_indices = np.argsort(distances.flatten())[0]

        # Assign DA_group to the 5 closest cells
        df.loc[selected_locations[closest_indices], 'DA_group_center'] = '_'.join(da_vec)

        # Outline the selected grid box in red
        selected_grid_rect = Rectangle(
            ((selected_grid_x - 1) * grid_size_x, (selected_grid_y - 1) * grid_size_y),
            grid_size_x, grid_size_y,
            edgecolor=hex, linewidth=1.5, fill=False
        )
        plt.gca().add_patch(selected_grid_rect)

    labels = df.loc[:, 'group'].value_counts()
    scale_dict = {'A': n_niches_cell_type, 'B': n_niches_cell_type, 'C': n_niches_cell_type, 'D': n_niches_cell_type, 'E': n_niches_cell_type}
    for i, v in labels.items():
        scale_dict[i] = scale_dict[i] - v
    relabel_arr = np.concatenate([[v]*k for i, (v, k) in enumerate(scale_dict.items())])

    df.loc[df['group'].isnull(), 'group'] = np.random.choice(relabel_arr, size = len(relabel_arr), replace=False)
    df['foldchange'] = pd.Series(df['group']).map(fc_dict)
    df.index = [f'Loc{i}' for i in range(1, df.shape[0]+1)]
    # Plot the points with colors representing the labels
    sns.scatterplot(x = 'x', y = 'y', hue = 'group', data = df, alpha=0.5, palette=colors_dict, hue_order = ['A', 'B', 'C', 'D', 'E'])

    axes.tick_params(labelsize=10)
    # Set axis labels and legend
    plt.xlabel('Y', fontsize = 12)
    plt.ylabel('X', fontsize = 12)
    plt.xlim(0, 1)
    plt.ylim(0,1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    if save_directory is not False:
        plt.savefig(os.path.join(save_directory, filename_save+'.pdf'), bbox_inches = 'tight')

    param_dict = {'seed': seed,
                  'num_grids_x': num_grids_x,
                  'num_grids_y': num_grids_y,
                  'n_regions': n_regions,
                  'da_vec': da_vec,
                  'n_niches': n_niches,
                  'selected_grids': selected_grid_list}
    
    return df, param_dict

def simulate_spatial_multi(num_grids_x=5, num_grids_y=5, n_region_list=[4, 3], n_niches=1000,
                            da_vec_list=[['A', 'C', 'E'], ['B']], seed=0, hex = '#377eb8',
                            colors_dict={'A': '#B46CDA', 'B': '#78CE8B', 'C': '#FF8595', 'D': '#1885F2', 'E': '#D78F09'},
                            fc_dict={'A': 1, 'B': 3, 'C': 5, 'D': 1.5, 'E': 2}, show_grid = False,
                            save_directory = None, filename_save = None):
    """
    Simulate spatial distribution of points across multiple sets of regions with specified parameters.

    Parameters:
        num_grids_x (int): Number of grid cells along the x-axis.
        num_grids_y (int): Number of grid cells along the y-axis.
        n_region_list (list): List of number of regions for each set.
        n_niches (int): Total number of points to distribute.
        da_vec_list (list): List of lists, each containing cell types to distribute in each set of regions.
        seed (int): Seed for random number generation.
        hex (str): Color code for grid lines.
        colors_dict (dict): Dictionary mapping cell types to colors.
        fc_dict (dict): Dictionary mapping cell types to fold change values.
        show_grid (bool): Whether to show grid lines on the plot.
        save_directory (str or None): Directory to save the plot. If None, plot is not saved.
        filename_save (str or None): Filename for the saved plot.

    Returns:
        DataFrame: DataFrame containing simulated spatial data.
        dict: Dictionary containing simulation parameters.
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    np.random.seed(seed)
    data = {
        'x': np.random.rand(n_niches),
        'y': np.random.rand(n_niches)}

    n_niches_cell_type = int(n_niches / 5)
    scale_dict = {'A': n_niches_cell_type, 'B': n_niches_cell_type, 'C': n_niches_cell_type,
                  'D': n_niches_cell_type, 'E': n_niches_cell_type}

    df = pd.DataFrame(data)

    grid_size_x = 1 / num_grids_x
    grid_size_y = 1 / num_grids_y

    if show_grid == True:
    # # Plot vertical grid lines
        for i in range(1, num_grids_x):
            plt.axvline(i * grid_size_x, color='gray', linestyle='--', linewidth=0.5)

        # Plot horizontal grid lines
        for i in range(1, num_grids_y):
            plt.axhline(i * grid_size_y, color='gray', linestyle='--', linewidth=0.5)

    df['group'] = np.nan
    df['group'] = df['group'].astype('object')

    df['DA_group'] = 'random'
    df['DA_group'] = df['DA_group'].astype('object')

    df['DA_group_center'] = 'random'
    df['DA_group_center'] = df['DA_group_center'].astype('object')

    linestyles = ['-', '--']
    selected_grid_list = []
    available_grids = set([(x, y) for x in range(1, num_grids_x + 1) for y in range(1, num_grids_y + 1)])
    total_grid_list = []
    tracker = []

    for n_regions in n_region_list:
        grids = []
        while len(grids) < n_regions:
            # Randomly select a grid cell
            grid_ = random.sample(available_grids, 1)[0]

            # Check if the selected grid is not adjacent to any of the previously selected grids
            if all((abs(grid_[0] - x) > 1 or abs(grid_[1] - y) > 1) for (x, y) in tracker):
                grids.append(grid_)
                tracker.append(grid_)
                available_grids.remove(grid_)
        total_grid_list.append(grids)
    # for set_idx, (n_regions, da_vec) in enumerate(zip(n_region_list, da_vec_list)):
    #     grids = random.sample(list(available_grids), n_regions)

    for set_idx, (grids, da_vec) in enumerate(zip(total_grid_list, da_vec_list)):
        for grid in grids:
            selected_grid_x = grid[0]
            selected_grid_y = grid[1]
            selected_grid_list.append((grid[0], grid[1]))
            # available_grids.remove(grid)

            selected_locations = df[
                (df['x'] >= (selected_grid_x - 1) * grid_size_x) & (df['x'] < selected_grid_x * grid_size_x) &
                (df['y'] >= (selected_grid_y - 1) * grid_size_y) & (df['y'] < selected_grid_y * grid_size_y)
            ].index

            # Reassign labels for the selected locations
            df.loc[selected_locations, 'group'] = np.random.choice(da_vec, size=len(selected_locations))
            df.loc[selected_locations, 'DA_group'] = '_'.join(da_vec)
        
                # Find the centroid of the selected grid
            centroid = (
                (selected_grid_x - 1 + selected_grid_x) * 0.5 * grid_size_x,
                (selected_grid_y - 1 + selected_grid_y) * 0.5 * grid_size_y
            )

            # Calculate distances from each cell to the centroid
            distances = cdist(df.loc[selected_locations, ['x', 'y']], [centroid])

            # Find the indices of the 5 closest cells
            closest_indices = np.argsort(distances.flatten())[0]

            # Assign DA_group to the 5 closest cells
            df.loc[selected_locations[closest_indices], 'DA_group_center'] = '_'.join(da_vec)
            selected_grid_rect = Rectangle(
                ((selected_grid_x - 1) * grid_size_x, (selected_grid_y - 1) * grid_size_y),
                grid_size_x, grid_size_y,
                edgecolor=hex, linewidth=1.5, fill=False)
            plt.gca().add_patch(selected_grid_rect)

    labels = df.loc[:, 'group'].value_counts()
    scale_dict = {'A': n_niches_cell_type, 'B': n_niches_cell_type, 'C': n_niches_cell_type,
                  'D': n_niches_cell_type, 'E': n_niches_cell_type}
    
    for i, v in labels.items():
        scale_dict[i] = scale_dict[i] - v
    relabel_arr = np.concatenate([[v] * k for i, (v, k) in enumerate(scale_dict.items())])

    df.loc[df['group'].isnull(), 'group'] = np.random.choice(relabel_arr, size=len(relabel_arr), replace=False)
    df['foldchange'] = pd.Series(df['group']).map(fc_dict)
    df.index = [f'Loc{i}' for i in range(1, df.shape[0] + 1)]
    sns.scatterplot(x='x', y='y', hue='group', data=df, alpha=0.5, palette=colors_dict,
                    hue_order=['A', 'B', 'C', 'D', 'E'])

    axes.tick_params(labelsize=10)
    plt.xlabel('Y', fontsize=12)
    plt.ylabel('X', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    if save_directory is not False:
        plt.savefig(os.path.join(save_directory, filename_save+'.pdf'), bbox_inches = 'tight')

    param_dict = {'seed': seed,
                  'num_grids_x': num_grids_x,
                  'num_grids_y': num_grids_y,
                  'n_regions': n_region_list,
                  'da_vec': da_vec_list,
                  'n_niches': n_niches,
                  'selected_grids': selected_grid_list}

    return df, param_dict

def sim_design1(n_patients_cond = 10,
                num_grids_x = 5,
                num_grids_y = 5,
                n_niches = 1000,
                n_regionsA = 6,
                n_regionsB = [3,3],
                da_vec_A = ['A', 'C', 'E'],
                da_vec_B = [['B', 'D'], ['E']],
                random_state_list_A = [58, 322, 1426, 65, 651, 417, 2788, 576, 213, 1828],
                random_state_list_B = [51, 1939, 2700, 1831, 804, 2633, 2777, 2053, 948, 420], 
                save_directory = None):
    """
    Simulate spatial data for two conditions (A and B) with different parameters.

    Parameters:
    - n_patients_cond (int): Number of patients for each condition.
    - num_grids_x (int): Number of grid cells along the x-axis.
    - num_grids_y (int): Number of grid cells along the y-axis.
    - n_niches (int): Number of niches.
    - n_regionsA (int): Number of regions for condition A.
    - n_regionsB (list of ints): List containing the number of regions for each patient in condition B.
    - da_vec_A (list of str): List of cell types for condition A.
    - da_vec_B (list of lists of str): List of lists containing cell types for each patient in condition B.
    - random_state_list_A (list of ints): List of random state values for condition A.
    - random_state_list_B (list of ints): List of random state values for condition B.
    - save_directory (str or None): Directory to save the simulated data. If None, data won't be saved.

    Returns:
    - adata_simulated (AnnData): An AnnData object containing the simulated spatial data.

    The function generates spatial data for conditions A and B, creates simulated patients, saves the data, and aggregates
    counts with metadata.
    """

    df_condA = pd.DataFrame()
    param_condA = {'seed':[], 'num_grids_x': [], 'num_grids_y': [], 'n_regions': [], 'da_vec': [], 'n_niches': [], 'selected_grids': []}
    for i in range(0, n_patients_cond):
        #n_regions = np.random.randint(0, 10)
        df, param_dict = simulate_spatial(num_grids_x = num_grids_x, num_grids_y = num_grids_y, n_regions = n_regionsA, n_niches = n_niches, da_vec = da_vec_A, seed = random_state_list_A[i], save_directory = save_directory, filename_save=f'A_{i}')
        df.to_csv(os.path.join('data', 'simulated', f'spatial_condA{i}.csv'))
        df['Patient_ID'] = i
        df_condA = pd.concat([df_condA, df], axis = 0)
        param_condA['seed'].append(param_dict['seed'])
        param_condA['num_grids_x'].append(param_dict['num_grids_x'])
        param_condA['num_grids_y'].append(param_dict['num_grids_y'])
        param_condA['n_regions'].append(param_dict['n_regions'])
        param_condA['da_vec'].append(param_dict['da_vec'])
        param_condA['n_niches'].append(param_dict['n_niches'])
        param_condA['selected_grids'].append(param_dict['selected_grids'])
    df_condA.to_csv(os.path.join('data', 'simulated', 'spatial_condA.csv'))
    np.save(os.path.join('data', 'simulated', 'param_condA'), param_condA)
        
    df_condB = pd.DataFrame()
    param_condB = {'seed':[], 'num_grids_x': [], 'num_grids_y': [], 'n_regions': [], 'da_vec': [], 'n_niches': [], 'selected_grids': []}
    for i in range(0, n_patients_cond):
        #np.random.randint(0, 5)
        #np.random.randint(0, 5)
        df, param_dict = simulate_spatial_multi(num_grids_x = num_grids_x, num_grids_y = num_grids_y, n_region_list = [n_regionsB[0], n_regionsB[1]], n_niches = n_niches, da_vec_list = da_vec_B, seed = random_state_list_B[i], save_directory = save_directory, filename_save=f'B_{i}')
        df.to_csv(os.path.join('data', 'simulated', f'spatial_condB{i}.csv'))
        df['Patient_ID'] = i
        df_condB = pd.concat([df_condB, df], axis = 0)
        param_condB['seed'].append(param_dict['seed'])
        param_condB['num_grids_x'].append(param_dict['num_grids_x'])
        param_condB['num_grids_y'].append(param_dict['num_grids_y'])
        param_condB['n_regions'].append(param_dict['n_regions'])
        param_condB['da_vec'].append(param_dict['da_vec'])
        param_condB['n_niches'].append(param_dict['n_niches'])
        param_condB['selected_grids'].append(param_dict['selected_grids'])
    df_condB.to_csv(os.path.join('data', 'simulated', 'spatial_condB.csv'))
    np.save(os.path.join('data', 'simulated', 'param_condB'), param_condB)

    ##aggregates counts with metadata 
    adata_simulated = []
    for i in range(0, n_patients_cond): #can change this if we want class imbalance
        for cond in ['A', 'B']:
            location = pd.read_csv(os.path.join('data', 'simulated', f'spatial_cond{cond}{i}.csv'), index_col=0)
            adata_run = sc.read_h5ad(os.path.join('data', 'simulated', f'adata_condA0.h5ad'))
            expression_df = pd.DataFrame(adata_run.X, index = adata_run.obs_names, columns = adata_run.var_names)
            A = expression_df.loc[adata_run.obs['group'][adata_run.obs['group'] == 'Path1'].sample(200).index, :]
            A.index = location[location['group'] == 'A'].index
            B = expression_df.loc[adata_run.obs['group'][adata_run.obs['group'] == 'Path2'].sample(200).index, :]
            B.index = location[location['group'] == 'B'].index
            C = expression_df.loc[adata_run.obs['group'][adata_run.obs['group'] == 'Path3'].sample(200).index, :]
            C.index = location[location['group'] == 'C'].index
            D = expression_df.loc[adata_run.obs['group'][adata_run.obs['group'] == 'Path4'].sample(200).index, :]
            D.index = location[location['group'] == 'D'].index
            E = expression_df.loc[adata_run.obs['group'][adata_run.obs['group'] == 'Path5'].sample(200).index, :]
            E.index = location[location['group'] == 'E'].index

            expression_df = pd.concat([A, B, C, D, E], axis = 0)
            expression_df = expression_df.loc[location.index]
            adata = anndata.AnnData(expression_df)
            adata.obsm['X_spatial'] = location.loc[:, ['x', 'y']].values
            adata.obs['cell_cluster'] = location.loc[:, 'group']
            adata.obs['DA_group'] = location.loc[:, 'DA_group']
            adata.obs['DA_group_center'] = location.loc[:, 'DA_group_center']
            adata.obs['condition'] = cond
            adata.obs['Patient_ID'] = f'{cond}{i}'
            adata.obs_names = [f'{cond}{i}_{j}' for j in adata.obs_names] #make unique obs names
            adata.obs['ground_labels'] = 0
            adata.obs['ground_labels'][np.where((np.isin(adata.obs['DA_group_center'], ['_'.join(da_vec_A)])) | (np.isin(adata.obs['DA_group_center'], ['_'.join(da_vec_B[0])]) ) | (np.isin(adata.obs['DA_group_center'], ['_'.join(da_vec_B[1])]) ))[0]] = 1
            adata_simulated.append(adata)
    adata_simulated = anndata.concat(adata_simulated)
    return adata_simulated

def simulate_spatial_design2(num_grids_x = 5, num_grids_y = 5, n_regions = 6, n_niches = 1000, da_vec = ['A', 'C', 'E'], seed = 0):
    """
    Simulate spatial data for a single condition with specified parameters.

    Parameters:
    - num_grids_x (int): Number of grid cells along the x-axis.
    - num_grids_y (int): Number of grid cells along the y-axis.
    - n_regions (int): Number of regions.
    - n_niches (int): Number of niches.
    - da_vec (list of str): List of cell types.
    - seed (int): Random seed for reproducibility.

    Returns:
    - df (DataFrame): DataFrame containing the simulated spatial data.
    - pop (list): List of population identifiers.
    - selected_grid_list (list of tuples): List of selected grid cells.

    This function generates spatial data for a single condition with specified parameters, assigns cell types to regions,
    and calculates centroids for selected grid cells.
    """
    # Create a sample DataFrame with x, y, and fold_change columns
    np.random.seed(seed)
    data = {
        'x': np.random.rand(n_niches),
        'y': np.random.rand(n_niches)}

    n_niches_cell_type = int(n_niches/5)
    scale_dict = {'A': n_niches_cell_type, 'B': n_niches_cell_type, 'C': n_niches_cell_type, 'D': n_niches_cell_type, 'E': n_niches_cell_type}

    df = pd.DataFrame(data)

    # Calculate grid size
    grid_size_x = 1 / num_grids_x
    grid_size_y = 1 / num_grids_y

    # Initialize the 'label' column
    df['group'] = np.nan
    df['group'] = df['group'].astype('object')

    df['DA_group'] = 'random'
    df['DA_group'] = df['DA_group'].astype('object')

    df['DA_group_center'] = 'random'
    df['DA_group_center'] = df['DA_group_center'].astype('object')

    df['grid_id'] = 0

    available_grids = list(set([(x, y) for x in range(1, num_grids_x + 1) for y in range(1, num_grids_y + 1)]))
    loc_dict = {}
    #int(''.join([str(grid[0]), str(grid[1])]))
    g = 0
    for grid in available_grids:
        selected_locations = df[
            (df['x'] >= (grid[0] - 1) * grid_size_x) & (df['x'] < grid[0] * grid_size_x) &
            (df['y'] >= (grid[1] - 1) * grid_size_y) & (df['y'] < grid[1] * grid_size_y)].index
        
        df['grid_id'][selected_locations] = g
        loc_dict[grid[0], grid[1]] = g
        g+=1
        
    grids = random.sample(list(available_grids), n_regions)
    selected_grid_list = []
    for grid in grids:
        # Randomly select a grid cell
        selected_grid_x = grid[0]
        selected_grid_y = grid[1]
        selected_grid_list.append((grid[0], grid[1]))

        # Select all cells in the chosen grid cell
        selected_locations = df[
            (df['x'] >= (selected_grid_x - 1) * grid_size_x) & (df['x'] < selected_grid_x * grid_size_x) &
            (df['y'] >= (selected_grid_y - 1) * grid_size_y) & (df['y'] < selected_grid_y * grid_size_y)
        ].index

        # Reassign labels for the selected locations
        df.loc[selected_locations, 'group'] = np.random.choice(da_vec, size=len(selected_locations))
        df.loc[selected_locations, 'DA_group'] = '_'.join(da_vec)

        # Find the centroid of the selected grid
        centroid = (
            (selected_grid_x - 1 + selected_grid_x) * 0.5 * grid_size_x,
            (selected_grid_y - 1 + selected_grid_y) * 0.5 * grid_size_y
        )    

        # Calculate distances from each cell to the centroid
        distances = cdist(df.loc[selected_locations, ['x', 'y']], [centroid])

        # Find the indices of the 5 closest cells
        closest_indices = np.argsort(distances.flatten())[0]

        # Assign DA_group to the 5 closest cells
        df.loc[selected_locations[closest_indices], 'DA_group_center'] = '_'.join(da_vec)

    labels = df.loc[:, 'group'].value_counts()
    scale_dict = {'A': n_niches_cell_type, 'B': n_niches_cell_type, 'C': n_niches_cell_type, 'D': n_niches_cell_type, 'E': n_niches_cell_type}
    for i, v in labels.items():
        scale_dict[i] = scale_dict[i] - v
    relabel_arr = np.concatenate([[v]*k for i, (v, k) in enumerate(scale_dict.items())])

    df.loc[df['group'].isnull(), 'group'] = np.random.choice(relabel_arr, size = len(relabel_arr), replace=False)
    pop = [loc_dict[i] for i in selected_grid_list]
    return df, pop, selected_grid_list
        
def sim_design2(num_grids_x = 5,
                num_grids_y = 5,
                n_niches = 1000,
                n_regions = 6,
                da_vec = ['A', 'C', 'E'],
                random_state = 0,
                p = 95,
                enr_fac = 10,
                save_directory = None):

    df, pop, selected_grid_list = simulate_spatial_design2(num_grids_x = num_grids_x, num_grids_y = num_grids_y, n_regions = n_regions, n_niches = n_niches, da_vec = da_vec, seed = random_state)
    cond_probability, synth_labels, synth_samples = compute_conditional_prob(df, pop = pop, region_label = 'grid_id', random_state = 0, enr_fac = enr_fac)
    
    location = pd.concat([df,
                        pd.DataFrame(synth_labels, columns = ['condition']),
                        pd.DataFrame(synth_samples, columns = ['sample'])], axis = 1)
    
    idx_ground = np.where(cond_probability['A'] > np.percentile(cond_probability['B'], p))[0]
    location.index = [f'Loc{i}' for i in location.index]
    location['ground_labels'] = 0
    location['ground_labels'].iloc[idx_ground] = 1

    quiche.pl.plot_grid_enrichment(location[location['condition'] == 'B'], ['A', 'B', 'C', 'D', 'E'], {'A': '#B46CDA','B': '#78CE8B', 'C': '#FF8595', 'D': '#1885F2', 'E': '#D78F09'}, selected_grid_list, num_grids_x, num_grids_y, save_directory, 'sim2_B')
    quiche.pl.plot_grid_enrichment(location[location['condition'] == 'A'], ['A', 'B', 'C', 'D', 'E'], {'A': '#B46CDA','B': '#78CE8B', 'C': '#FF8595', 'D': '#1885F2', 'E': '#D78F09'}, selected_grid_list, num_grids_x, num_grids_y, save_directory, 'sim2_A')

    adata_run = sc.read_h5ad(os.path.join('data', 'simulated', f'adata_condA0.h5ad'))
    expression_df = pd.DataFrame(adata_run.X, index = adata_run.obs_names, columns = adata_run.var_names)
    A = expression_df.loc[adata_run.obs['group'][adata_run.obs['group'] == 'Path1'].sample(200).index, :]
    A.index = location[location['group'] == 'A'].index
    B = expression_df.loc[adata_run.obs['group'][adata_run.obs['group'] == 'Path2'].sample(200).index, :]
    B.index = location[location['group'] == 'B'].index
    C = expression_df.loc[adata_run.obs['group'][adata_run.obs['group'] == 'Path3'].sample(200).index, :]
    C.index = location[location['group'] == 'C'].index
    D = expression_df.loc[adata_run.obs['group'][adata_run.obs['group'] == 'Path4'].sample(200).index, :]
    D.index = location[location['group'] == 'D'].index
    E = expression_df.loc[adata_run.obs['group'][adata_run.obs['group'] == 'Path5'].sample(200).index, :]
    E.index = location[location['group'] == 'E'].index

    expression_df = pd.concat([A, B, C, D, E], axis = 0)
    expression_df = expression_df.loc[location.index]
    adata = anndata.AnnData(expression_df)
    adata.obsm['X_spatial'] = location.loc[:, ['x', 'y']].values
    adata.obs['cell_cluster'] = location.loc[:, 'group']
    adata.obs['DA_group'] = location.loc[:, 'DA_group']
    adata.obs['DA_group_center'] = location.loc[:, 'DA_group_center']
    adata.obs['condition'] = location.loc[:, 'condition']
    adata.obs['ground_labels'] = location.loc[:, 'ground_labels']
    adata.obs['Patient_ID'] = location.loc[:, 'sample']
    adata.obs['conditional_prob_A'] = np.array(cond_probability['A'].values, dtype = 'float')
    adata.obs['conditional_prob_B'] = np.array(cond_probability['B'].values, dtype = 'float')
    adata.obs_names = (location['condition'] + '_' + location.index).values #make unique obs names
    return adata

def find_centroid(X_emb, cluster_membership):
    """
    Calculate the centroid of each cluster based on the embedding coordinates.

    Args:
    - X_emb (numpy.ndarray): An array containing the embedding coordinates of each data point.
    - cluster_membership (numpy.ndarray): An array containing the cluster assignments of each data point.

    Returns:
    - centroid_emb (numpy.ndarray): An array containing the coordinates of the centroids of each cluster.
    """
    cl_ixs = [np.where(cluster_membership == i)[0] for i in range(max(cluster_membership) + 1)]
    centroid_emb = np.array([np.mean(X_emb[ix, :], axis=0) for ix in cl_ixs])
    return centroid_emb

def scale_to_range(x, minimum = 1, maximum = 10):
    """
    Scale the input array to a specified range.

    Args:
    - x (numpy.ndarray): The input array to be scaled.
    - minimum (int or float): The minimum value of the desired range.
    - maximum (int or float): The maximum value of the desired range.

    Returns:
    - scaled_x (numpy.ndarray): The scaled array.
    """
    scaled_x = ((x - np.min(x)) / (np.max(x) - np.min(x))) * (maximum - minimum) + minimum
    return scaled_x

def scale_to_logit(w, a_logit=0.5):
    """
    Scale the input DataFrame to the logit space.

    Args:
    - w (pandas.DataFrame): The input DataFrame to be scaled.
    - a_logit (float): Scaling factor for the logit transformation.

    Returns:
    - logit_w (pandas.DataFrame): The scaled DataFrame in the logit space.
    """

    scaled_columns = []
    
    for col in w.columns:
        column_data = w[col].values.reshape(-1, 1)
        scaled_data = StandardScaler().fit_transform(column_data)
        logit_data = 1 / (1 + np.exp(-a_logit * scaled_data))
        scaled_columns.append(logit_data)
    
    logit_w = pd.DataFrame(np.concatenate(scaled_columns, axis=1), columns=w.columns)
    return logit_w

def calculate_weights(centroid_dist, df, m=2, a_logit = 0.5, cluster_key = 'group', region_key = 'grid_id', pop = None, enr_fac = 10):
    """
    Calculate weights for the synthetic data generation based on the distances from cluster centroids.

    Args:
    - centroid_dist (numpy.ndarray): Array containing the distances of each data point to cluster centroids.
    - df (pandas.DataFrame): DataFrame containing the data points and their associated cluster and region labels.
    - m (int): Fuzziness parameter for the calculation of weights.
    - a_logit (float): Scaling factor for the logit transformation.
    - cluster_key (str): Column name in df representing the cluster labels.
    - region_key (str): Column name in df representing the region labels.
    - pop (list or None): List of regions of interest.
    - enr_fac (int): Enrichment factor for regions of interest.

    Returns:
    - cond_probability (pandas.DataFrame): DataFrame containing the conditional probabilities of each data point belonging to different conditions.
    - synth_labels (pandas.Series): Series containing the synthetic labels assigned to each data point.
    - synth_samples (list): List of synthetic sample names.
    """
    w = np.zeros_like(centroid_dist)
    for j in range(centroid_dist.shape[1]):
        for i in range(centroid_dist.shape[0]):
            w[i, j] = 1 / np.sum((centroid_dist[i, j] / centroid_dist[i, :]) ** (2 / (m - 1)))

    w = pd.DataFrame(w, columns=np.arange(0, df[region_key].max()+1), index=df.index)
    w = scale_to_logit(w, a_logit)

    enr_scores = pd.Series(0.5, index = w.columns)
    enr_scores[pop] = enr_fac #set pop of interest to higher DA

    enr_prob = pd.DataFrame(index=w.index, columns=w.columns)
    for i in range(w.shape[1]):
        enr_prob.iloc[:, i] = scale_to_range(w.iloc[:, i], minimum=0.5 * 1, maximum=enr_scores[i]) #normalize the weights 

    prob_matrix = enr_prob[pop]
    
    if isinstance(prob_matrix, pd.DataFrame):
        cond_probability = prob_matrix.max(axis=1)
        for x in range(len(pop)):
            cond_probability[df[region_key] == pop[x]] = prob_matrix[df[region_key] == pop[x]][pop[x]]
    else:
        cond_probability = prob_matrix

    cond_probability = np.column_stack([cond_probability, 1 - cond_probability])
    cond_probability = pd.DataFrame(cond_probability, columns=['A', 'B'])

    # Initialize an empty list to store the sampled condition labels
    synth_labels = pd.DataFrame()
    for cell_type in df[cluster_key].unique():
        # Subset the DataFrame for the current cell type
        subset_cond = cond_probability[df[cluster_key] == cell_type]
        # cond1 = subset_cond.max(1).sort_values(ascending = False)[:100].index
        # cond2 = subset_cond.max(1).sort_values(ascending = False)[100:].index
        scaled_prob = MinMaxScaler().fit_transform(subset_cond['A'].values.reshape(-1,1))
        scaled_prob = pd.DataFrame(scaled_prob, index = subset_cond['A'].index)
        cond1 = np.random.choice(subset_cond.index, size = 100, p = subset_cond['A'] / subset_cond['A'].sum(), replace = False)
        # cond1 = np.random.choice(subset_cond.index, size = 100, p = (scaled_prob / scaled_prob.sum()).values.flatten(), replace = False)
        cond2 = list(set(subset_cond.index).difference(set(cond1)))
        subset_cond['label'] = 'N'
        subset_cond['label'].loc[cond1] = 'A'
        subset_cond['label'].loc[cond2] = 'B'
        synth_labels = pd.concat([synth_labels,subset_cond['label']], axis = 0)

    synth_labels = synth_labels.loc[df.index, :]
    synth_labels = pd.Series(synth_labels[0].values)

    # synth_labels = pd.Series([np.random.choice(cond_probability.columns, p=cond_probability.iloc[i, :]) for i in range(len(cond_probability))])

    # Generate replicates
    replicates = [f"R{i}" for i in range(1, 3 + 1)]

    # Generate synthetic samples
    synth_samples = [f"{label}_{replicate}" for label, replicate in zip(synth_labels, replicates * len(synth_labels))]

    return cond_probability, synth_labels, synth_samples

def compute_conditional_prob(df, pop = None, region_label = 'grid_id', random_state = 0, enr_fac = 10):
    """
    Compute the conditional probabilities of each data point belonging to different conditions.

    Args:
    - df (pandas.DataFrame): DataFrame containing the data points and their associated region labels.
    - pop (list or None): List of regions of interest.
    - region_label (str): Column name in df representing the region labels.
    - random_state (int): Random seed for reproducibility.
    - enr_fac (int): Enrichment factor for regions of interest.

    Returns:
    - cond_probability (pandas.DataFrame): DataFrame containing the conditional probabilities of each data point belonging to different conditions.
    - synth_labels (pandas.Series): Series containing the synthetic labels assigned to each data point.
    - synth_samples (list): List of synthetic sample names.
    """
    np.random.seed(random_state)
    centroid_emb = find_centroid(df.loc[:, ['x', 'y']].values,  df[region_label].values)
    centroid_dist = np.sqrt(np.sum((df.loc[:, ['x', 'y']].values[:, np.newaxis, :] - centroid_emb) ** 2, axis=2))
    cond_probability, synth_labels, synth_samples = calculate_weights(centroid_dist, df, m=2, a_logit=0.5, pop = pop, enr_fac = enr_fac)
    return cond_probability, synth_labels, synth_samples

def sim_design1_uneven(n_patients_cond = 10,
                        num_grids_x = 5,
                        num_grids_y = 5,
                        n_niches = 1000,
                        n_regionsA = 1,
                        n_regionsB = [1,1],
                        A_ratio = 0.2,
                        B_ratio = 0.2,
                        hex_A = '#e41a1c',
                        hex_B = '#377eb8',
                        da_vec_A = ['A', 'C', 'E'],
                        da_vec_B = [['B', 'D'], ['E']],
                        fig_id = 'ACE_BD_E_region1_ratio0.2',
                        random_state_list_A = [58, 322, 1426, 65, 651, 417, 2788, 576, 213, 1828],
                        random_state_list_B = [51, 1939, 2700, 1831, 804, 2633, 2777, 2053, 948, 420], 
                        save_directory = None):
    """
    Generate synthetic spatial data with uneven region distributions for different conditions.

    Args:
    - n_patients_cond (int): Number of patients for each condition.
    - num_grids_x (int): Number of grids along the x-axis.
    - num_grids_y (int): Number of grids along the y-axis.
    - n_niches (int): Number of niches.
    - n_regionsA (int): Number of regions for condition A.
    - n_regionsB (list): List containing the number of regions for each sub-condition in condition B.
    - A_ratio (float): Ratio of condition A samples to total samples.
    - B_ratio (float): Ratio of condition B samples to total samples.
    - hex_A (str): Hex color code for condition A.
    - hex_B (str): Hex color code for condition B.
    - da_vec_A (list): List of data attributes for condition A.
    - da_vec_B (list): List of data attributes for condition B.
    - fig_id (str): Identifier for the figure.
    - random_state_list_A (list): List of random states for condition A.
    - random_state_list_B (list): List of random states for condition B.
    - save_directory (str or None): Directory to save the simulated data.

    Returns:
    - adata_simulated (anndata.AnnData): Anndata object containing the simulated spatial data.
    """
    
    df_condA = pd.DataFrame()
    param_condA = {'seed':[], 'num_grids_x': [], 'num_grids_y': [], 'n_regions': [], 'da_vec': [], 'n_niches': [], 'selected_grids': []}
    run = 0
    for i in range(0, n_patients_cond):
        #n_regions = np.random.randint(0, 10)
        if run < int(n_patients_cond*A_ratio):
            df, param_dict = qu.tl.simulate_spatial(num_grids_x = num_grids_x, num_grids_y = num_grids_y, n_regions = n_regionsA, hex = hex_A,n_niches = n_niches, da_vec = da_vec_A, seed = random_state_list_A[i], save_directory = save_directory, filename_save=f'A_{i}_{fig_id}')
        else:
            df, param_dict = qu.tl.simulate_spatial(num_grids_x = num_grids_x, num_grids_y = num_grids_y, n_regions = 0, n_niches = n_niches, hex = hex_A, da_vec = da_vec_A, seed = random_state_list_A[i], save_directory = save_directory, filename_save=f'A_{i}_{fig_id}')
        df.to_csv(os.path.join('data', 'simulated', f'spatial_condA{i}_{fig_id}.csv'))
        df['Patient_ID'] = i
        df_condA = pd.concat([df_condA, df], axis = 0)
        param_condA['seed'].append(param_dict['seed'])
        param_condA['num_grids_x'].append(param_dict['num_grids_x'])
        param_condA['num_grids_y'].append(param_dict['num_grids_y'])
        param_condA['n_regions'].append(param_dict['n_regions'])
        param_condA['da_vec'].append(param_dict['da_vec'])
        param_condA['n_niches'].append(param_dict['n_niches'])
        param_condA['selected_grids'].append(param_dict['selected_grids'])
        run +=1

    df_condA.to_csv(os.path.join('data', 'simulated', f'spatial_condA_{fig_id}.csv'))
    np.save(os.path.join('data', 'simulated', f'param_condA_{fig_id}'), param_condA)
        
    df_condB = pd.DataFrame()
    param_condB = {'seed':[], 'num_grids_x': [], 'num_grids_y': [], 'n_regions': [], 'da_vec': [], 'n_niches': [], 'selected_grids': []}
    run = 0
    for i in range(0, n_patients_cond):
        #np.random.randint(0, 5)
        #np.random.randint(0, 5)
        if run < int(n_patients_cond*B_ratio):
            df, param_dict = qu.tl.simulate_spatial_multi(num_grids_x = num_grids_x, num_grids_y = num_grids_y, n_region_list = [n_regionsB[0], n_regionsB[1]], hex = hex_B, n_niches = n_niches, da_vec_list = da_vec_B, seed = random_state_list_B[i], save_directory = save_directory, filename_save=f'B_{i}_{B_ratio}')
        else:
            df, param_dict = qu.tl.simulate_spatial_multi(num_grids_x = num_grids_x, num_grids_y = num_grids_y, n_region_list = [0,0], n_niches = n_niches, hex = hex_B, da_vec_list = da_vec_B, seed = random_state_list_B[i], save_directory = save_directory, filename_save=f'B_{i}_{B_ratio}')
        df.to_csv(os.path.join('data', 'simulated', f'spatial_condB{i}_{fig_id}.csv'))
        df['Patient_ID'] = i
        df_condB = pd.concat([df_condB, df], axis = 0)
        param_condB['seed'].append(param_dict['seed'])
        param_condB['num_grids_x'].append(param_dict['num_grids_x'])
        param_condB['num_grids_y'].append(param_dict['num_grids_y'])
        param_condB['n_regions'].append(param_dict['n_regions'])
        param_condB['da_vec'].append(param_dict['da_vec'])
        param_condB['n_niches'].append(param_dict['n_niches'])
        param_condB['selected_grids'].append(param_dict['selected_grids'])
        run+=1
    df_condB.to_csv(os.path.join('data', 'simulated', f'spatial_condB_{fig_id}.csv'))
    np.save(os.path.join('data', 'simulated', f'param_condB_{fig_id}'), param_condB)

    ##aggregates counts with metadata 
    adata_simulated = []
    adata_run = sc.read_h5ad(os.path.join('data', 'simulated', f'adata_simulated_expression.h5ad'))

    for i in range(0, n_patients_cond): #can change this if we want class imbalance
        for cond in ['A', 'B']:
            print(i, cond)
            expression_df = pd.DataFrame(adata_run.X, index = adata_run.obs_names, columns = adata_run.var_names)
            location = pd.read_csv(os.path.join('data', 'simulated', f'spatial_cond{cond}{i}_{fig_id}.csv'), index_col=0)
            sample = list(adata_run.obs['group'][adata_run.obs['group'] == 'Path1'].sample(200).index)
            A = expression_df.loc[sample, :]
            adata_run = adata_run[~np.isin(adata_run.obs_names, sample)]
            A.index = location[location['group'] == 'A'].index
            sample = list(adata_run.obs['group'][adata_run.obs['group'] == 'Path2'].sample(200).index)
            B = expression_df.loc[sample, :]
            adata_run = adata_run[~np.isin(adata_run.obs_names, sample)]
            B.index = location[location['group'] == 'B'].index
            sample = list(adata_run.obs['group'][adata_run.obs['group'] == 'Path3'].sample(200).index)
            C = expression_df.loc[sample, :]
            adata_run = adata_run[~np.isin(adata_run.obs_names, sample)]
            C.index = location[location['group'] == 'C'].index
            sample = list(adata_run.obs['group'][adata_run.obs['group'] == 'Path4'].sample(200).index)
            D = expression_df.loc[sample, :]
            adata_run = adata_run[~np.isin(adata_run.obs_names, sample)]
            D.index = location[location['group'] == 'D'].index
            sample = list(adata_run.obs['group'][adata_run.obs['group'] == 'Path5'].sample(200).index)
            E = expression_df.loc[sample, :]
            adata_run = adata_run[~np.isin(adata_run.obs_names, sample)]
            E.index = location[location['group'] == 'E'].index

            expression_df = pd.concat([A, B, C, D, E], axis = 0)
            expression_df = expression_df.loc[location.index]
            adata = anndata.AnnData(expression_df)
            adata.obsm['X_spatial'] = location.loc[:, ['x', 'y']].values
            adata.obs['cell_cluster'] = location.loc[:, 'group']
            adata.obs['DA_group'] = location.loc[:, 'DA_group']
            adata.obs['DA_group_center'] = location.loc[:, 'DA_group_center']
            adata.obs['condition'] = cond
            adata.obs['Patient_ID'] = f'{cond}{i}'
            adata.obs_names = [f'{cond}{i}_{j}' for j in adata.obs_names] #make unique obs names
            adata.obs['ground_labels'] = 0
            adata.obs['ground_labels'][np.where((np.isin(adata.obs['DA_group_center'], ['_'.join(da_vec_A)])) | (np.isin(adata.obs['DA_group_center'], ['_'.join(da_vec_B[0])]) ) | (np.isin(adata.obs['DA_group_center'], ['_'.join(da_vec_B[1])]) ))[0]] = 1
            adata_simulated.append(adata)
    adata_simulated = anndata.concat(adata_simulated)
    return adata_simulated

def sim_design1_uneven2(n_patients_cond = 10,
                        num_grids_x = 5,
                        num_grids_y = 5,
                        n_niches = 1000,
                        n_regionsA = 1,
                        n_regionsB = [1],
                        A_ratio = 0.2,
                        B_ratio = 0.2,
                        da_vec_A = ['A', 'C', 'E'],
                        da_vec_B = [['B', 'D']],
                        hex_A = '#e41a1c',
                        hex_B = '#377eb8',
                        random_state_list_A = [58, 322, 1426, 65, 651, 417, 2788, 576, 213, 1828],
                        random_state_list_B = [51, 1939, 2700, 1831, 804, 2633, 2777, 2053, 948, 420], 
                        fig_id = 'ACE_BD_region1_ratio0.2',
                        save_directory = None):
    """
    Generate synthetic spatial data with uneven region distributions for two conditions.

    Args:
    - n_patients_cond (int): Number of patients for each condition.
    - num_grids_x (int): Number of grids along the x-axis.
    - num_grids_y (int): Number of grids along the y-axis.
    - n_niches (int): Number of niches.
    - n_regionsA (int): Number of regions for condition A.
    - n_regionsB (list): List containing the number of regions for condition B.
    - A_ratio (float): Ratio of condition A samples to total samples.
    - B_ratio (float): Ratio of condition B samples to total samples.
    - da_vec_A (list): List of data attributes for condition A.
    - da_vec_B (list): List of data attributes for condition B.
    - hex_A (str): Hex color code for condition A.
    - hex_B (str): Hex color code for condition B.
    - random_state_list_A (list): List of random states for condition A.
    - random_state_list_B (list): List of random states for condition B.
    - fig_id (str): Identifier for the figure.
    - save_directory (str or None): Directory to save the simulated data.

    Returns:
    - adata_simulated (anndata.AnnData): Anndata object containing the simulated spatial data.
    """

    
    df_condA = pd.DataFrame()
    param_condA = {'seed':[], 'num_grids_x': [], 'num_grids_y': [], 'n_regions': [], 'da_vec': [], 'n_niches': [], 'selected_grids': []}
    run = 0
    for i in range(0, n_patients_cond):
        #n_regions = np.random.randint(0, 10)
        if run < int(n_patients_cond*A_ratio):
            df, param_dict = qu.tl.simulate_spatial(num_grids_x = num_grids_x, hex = hex_A, num_grids_y = num_grids_y, n_regions = n_regionsA, n_niches = n_niches, da_vec = da_vec_A, seed = random_state_list_A[i], save_directory = save_directory, filename_save=f'A_{i}_{fig_id}')
        else:
            df, param_dict = qu.tl.simulate_spatial(num_grids_x = num_grids_x, hex = hex_A, num_grids_y = num_grids_y, n_regions = 0, n_niches = n_niches, da_vec = da_vec_A, seed = random_state_list_A[i], save_directory = save_directory, filename_save=f'A_{i}_{fig_id}')
        df.to_csv(os.path.join('data', 'simulated', f'spatial_condA{i}_{fig_id}.csv'))
        df['Patient_ID'] = i
        df_condA = pd.concat([df_condA, df], axis = 0)
        param_condA['seed'].append(param_dict['seed'])
        param_condA['num_grids_x'].append(param_dict['num_grids_x'])
        param_condA['num_grids_y'].append(param_dict['num_grids_y'])
        param_condA['n_regions'].append(param_dict['n_regions'])
        param_condA['da_vec'].append(param_dict['da_vec'])
        param_condA['n_niches'].append(param_dict['n_niches'])
        param_condA['selected_grids'].append(param_dict['selected_grids'])
        run +=1

    df_condA.to_csv(os.path.join('data', 'simulated', f'spatial_condA_{fig_id}.csv'))
    np.save(os.path.join('data', 'simulated', f'param_condA_{fig_id}'), param_condA)
        
    df_condB = pd.DataFrame()
    param_condB = {'seed':[], 'num_grids_x': [], 'num_grids_y': [], 'n_regions': [], 'da_vec': [], 'n_niches': [], 'selected_grids': []}
    run = 0
    for i in range(0, n_patients_cond):
        #np.random.randint(0, 5)
        #np.random.randint(0, 5)
        if run < int(n_patients_cond*B_ratio):
            df, param_dict = qu.tl.simulate_spatial(num_grids_x = num_grids_x, hex = hex_B, num_grids_y = num_grids_y, n_regions = n_regionsB, n_niches = n_niches, da_vec = da_vec_B, seed = random_state_list_B[i], save_directory = save_directory, filename_save=f'B_{i}_{fig_id}')
        else:
            df, param_dict = qu.tl.simulate_spatial(num_grids_x = num_grids_x, hex = hex_B, num_grids_y = num_grids_y, n_regions = 0, n_niches = n_niches, da_vec = da_vec_B, seed = random_state_list_B[i], save_directory = save_directory, filename_save=f'B_{i}_{fig_id}')
        df.to_csv(os.path.join('data', 'simulated', f'spatial_condB{i}_{fig_id}.csv'))
        df['Patient_ID'] = i
        df_condB = pd.concat([df_condB, df], axis = 0)
        param_condB['seed'].append(param_dict['seed'])
        param_condB['num_grids_x'].append(param_dict['num_grids_x'])
        param_condB['num_grids_y'].append(param_dict['num_grids_y'])
        param_condB['n_regions'].append(param_dict['n_regions'])
        param_condB['da_vec'].append(param_dict['da_vec'])
        param_condB['n_niches'].append(param_dict['n_niches'])
        param_condB['selected_grids'].append(param_dict['selected_grids'])
        run+=1
    df_condB.to_csv(os.path.join('data', 'simulated', f'spatial_condB_{fig_id}.csv'))
    np.save(os.path.join('data', 'simulated', f'param_condB_{fig_id}'), param_condB)

    ##aggregates counts with metadata 
    adata_simulated = []
    adata_run = sc.read_h5ad(os.path.join('data', 'simulated', f'adata_simulated_expression.h5ad'))

    for i in range(0, n_patients_cond): #can change this if we want class imbalance
        for cond in ['A', 'B']:
            print(i, cond)
            expression_df = pd.DataFrame(adata_run.X, index = adata_run.obs_names, columns = adata_run.var_names)
            location = pd.read_csv(os.path.join('data', 'simulated', f'spatial_cond{cond}{i}_{fig_id}.csv'), index_col=0)
            sample = list(adata_run.obs['group'][adata_run.obs['group'] == 'Path1'].sample(200).index)
            A = expression_df.loc[sample, :]
            adata_run = adata_run[~np.isin(adata_run.obs_names, sample)]
            A.index = location[location['group'] == 'A'].index
            sample = list(adata_run.obs['group'][adata_run.obs['group'] == 'Path2'].sample(200).index)
            B = expression_df.loc[sample, :]
            adata_run = adata_run[~np.isin(adata_run.obs_names, sample)]
            B.index = location[location['group'] == 'B'].index
            sample = list(adata_run.obs['group'][adata_run.obs['group'] == 'Path3'].sample(200).index)
            C = expression_df.loc[sample, :]
            adata_run = adata_run[~np.isin(adata_run.obs_names, sample)]
            C.index = location[location['group'] == 'C'].index
            sample = list(adata_run.obs['group'][adata_run.obs['group'] == 'Path4'].sample(200).index)
            D = expression_df.loc[sample, :]
            adata_run = adata_run[~np.isin(adata_run.obs_names, sample)]
            D.index = location[location['group'] == 'D'].index
            sample = list(adata_run.obs['group'][adata_run.obs['group'] == 'Path5'].sample(200).index)
            E = expression_df.loc[sample, :]
            adata_run = adata_run[~np.isin(adata_run.obs_names, sample)]
            E.index = location[location['group'] == 'E'].index

            expression_df = pd.concat([A, B, C, D, E], axis = 0)
            expression_df = expression_df.loc[location.index]
            adata = anndata.AnnData(expression_df)
            adata.obsm['X_spatial'] = location.loc[:, ['x', 'y']].values
            adata.obs['cell_cluster'] = location.loc[:, 'group']
            adata.obs['DA_group'] = location.loc[:, 'DA_group']
            adata.obs['DA_group_center'] = location.loc[:, 'DA_group_center']
            adata.obs['condition'] = cond
            adata.obs['Patient_ID'] = f'{cond}{i}'
            adata.obs_names = [f'{cond}{i}_{j}' for j in adata.obs_names] #make unique obs names
            adata.obs['ground_labels'] = 0
            adata.obs['ground_labels'][np.where((np.isin(adata.obs['DA_group_center'], ['_'.join(da_vec_A)])) | (np.isin(adata.obs['DA_group_center'], ['_'.join(da_vec_B)]) ))[0]] = 1
            adata_simulated.append(adata)
    adata_simulated = anndata.concat(adata_simulated)
    return adata_simulated

# import numpy as np
# import scanpy as sc 
# import anndata
# import delve_benchmark
# import gc
# import logging
# import rpy2.robjects as robjects
# import gc 
# from sklearn.model_selection import train_test_split
# import scprep
# import pandas as pd

# def linear_mask_(metadata):
#     # makes step variable monotonically increeasing for the linear trajectory
#     metadata_ = metadata.copy()
#     mask_root = metadata_['group'] == 'Path1'
#     metadata_.loc[mask_root, 'step'] = 100 - metadata_.loc[mask_root, 'step']
#     for i in [2,3,4,5]:
#         mask = metadata_['group'] == 'Path'+str(i)
#         metadata_.loc[mask, 'step'] = 100*(i-1) + metadata_.loc[mask, 'step']
#     return metadata_

# def _sum_to_one(x):
#     x = x / np.sum(x)
#     x = x.round(3)
#     if np.sum(x) != 1:
#         x[0] += 1 - np.sum(x)
#     x = x.round(3)
#     return x

# def splatter_sim(cells_per_path = 200,
#                 n_paths = 5,
#                 n_genes = 500,
#                 bcv_common = 0.1,
#                 lib_loc = 12,
#                 path_from = None,
#                 path_skew = None,
#                 path_type = None,
#                 group_prob = None,
#                 random_state = 0):
#     """Simulates a single-cell RNA sequencing trajectory using Splatter: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1305-0. 
#     ~~~ Uses the scprep wrapper function: https://scprep.readthedocs.io/en/stable/_modules/scprep/run/splatter.html ~~~  
#     Parameters
#     For more details on the parameters, see: https://scprep.readthedocs.io/en/stable/_modules/scprep/run/splatter.html#SplatSimulate
#     ----------
#     Returns
#     adata: anndata.AnnData
#         annotated data object containing simulated single-cell RNA sequecing data (dimensions = cells x features)
#     ----------
#     """ 
#     #set simulation parameters from real single-cell RNA sequencing dataset: https://pubmed.ncbi.nlm.nih.gov/27419872/
#     params = {}
#     params['group_prob'] = group_prob
#     params['bcv_common'] = bcv_common
#     params['path_from'] = path_from
#     params['path_skew'] = path_skew
#     params['mean_rate'] = 0.0173
#     params['mean_shape'] = 0.54
#     if lib_loc is None:
#         params['lib_loc'] = 12.6
#     else: 
#         params['lib_loc'] = lib_loc
#     params['lib_scale'] = 0.423
#     params['out_prob'] = 0.000342
#     params['out_fac_loc'] = 0.1
#     params['out_fac_scale'] = 0.4
#     params['bcv_df'] = 90.2
#     results = scprep.run.SplatSimulate(method = 'paths', 
#                                         batch_cells = [cells_per_path * n_paths], 
#                                         group_prob = params['group_prob'], 
#                                         n_genes = n_genes,
#                                         de_prob = 0.1,
#                                         de_down_prob = 0.5,
#                                         de_fac_loc = 0.1,
#                                         de_fac_scale = 0.4, 
#                                         bcv_common = params['bcv_common'],
#                                         dropout_type = 'none',
#                                         path_from = params['path_from'],
#                                         path_skew = params['path_skew'],
#                                         mean_rate = params['mean_rate'],
#                                         mean_shape = params['mean_shape'],
#                                         lib_loc = params['lib_loc'], 
#                                         lib_scale = params['lib_scale'], 
#                                         out_prob = params['out_prob'], 
#                                         out_fac_loc = params['out_fac_loc'], 
#                                         out_fac_scale = params['out_fac_scale'], 
#                                         bcv_df = params['bcv_df'],
#                                         seed = random_state)
#     data = pd.DataFrame(results['counts'])
#     group = results['group'].copy()
#     metadata = pd.DataFrame({'group':group.astype('str'), 'step':results['step'].astype(int)})
#     if path_type == 'linear':
#         metadata = linear_mask_(metadata)
#     elif path_type == 'branch':
#         metadata = branch_mask_(metadata)
#     de_genes = pd.concat([pd.DataFrame(results['de_fac_1'], columns = ['path1']),
#                             pd.DataFrame(results['de_fac_2'], columns = ['path2']),
#                             pd.DataFrame(results['de_fac_3'], columns = ['path3']),
#                             pd.DataFrame(results['de_fac_4'], columns = ['path4']),
#                             pd.DataFrame(results['de_fac_5'], columns = ['path5'])], axis = 1)            
#     gene_index = []
#     for i in range(0, len(de_genes.index)):
#         if de_genes.loc[i].sum() != n_paths:
#             id = 'DE_group_' + '_'.join(map(str, (np.where(de_genes.loc[i] !=1)[0]))) + '.{}'.format(i)
#             gene_index.append(id)
#         else:
#             gene_index.append(str(i))
#     cell_index = pd.Index(['cell_{}'.format(i) for i in range(metadata.shape[0])])
#     data.index = cell_index
#     data.columns = gene_index
#     metadata.index = cell_index
#     adata = anndata.AnnData(data)
#     adata.obs = metadata
#     adata.layers['raw'] = adata.X.copy()
#     sc.pp.normalize_total(adata)
#     sc.pp.log1p(adata)
#     return adata

# n_paths = 5
# group_prob = np.random.dirichlet(np.ones(n_paths) * 1.).round(3)
# group_prob = _sum_to_one(group_prob)
# path_skew = np.random.beta(10., 10., n_paths)
# adata = splatter_sim(cells_per_path = 3000, n_paths = n_paths, n_genes = 500, bcv_common = 0.1, lib_loc = 12,
#                     path_from = [0, 1, 2, 3, 4], path_skew = path_skew,
#                     group_prob = [1,1,1,1,1], path_type = 'linear', random_state = 0)
# adata.write_h5ad(f'adata_simulated_expression.h5ad')
