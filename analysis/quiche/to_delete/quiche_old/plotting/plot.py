import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc 
import quiche
from matplotlib.colors import to_rgba
from matplotlib.cm import ScalarMappable

def generate_colors(cmap="viridis", n_colors=3, alpha=.4):
    """
    Generate colors from matplotlib colormap.

    Args:
        cmap (str or list): Colormap name or list of colors.
        n_colors (int): Number of colors.
        alpha (float): Alpha value for transparency.

    Returns:
        colors (list): List of RGBA colors.
    """
    if not isinstance(n_colors, int) or (n_colors < 2) or (n_colors > 6):
        raise ValueError("n_colors must be an integer between 2 and 6")
    if isinstance(cmap, list):
        colors = [to_rgba(color, alpha=alpha) for color in cmap]
    else:
        scalar_mappable = ScalarMappable(cmap=cmap)
        colors = scalar_mappable.to_rgba(range(n_colors), alpha=alpha).tolist()
    return colors[:n_colors]

def plot_neighbors_histogram_per_fov(adata,
                                        radius=200,
                                        p=2,
                                        figcomp = [6,5],
                                        xlim = [0,450],
                                        ylim = [0, 2000],
                                        figsize = (15, 16),
                                        fov_subset = None,
                                        spatial_key='X_spatial',
                                        fov_key='fov',
                                        save_directory = 'figures',
                                        filename_save = None):
    """
    Plot histogram of neighbors per field of view (FOV).

    Args:
        adata (AnnData): Annotated data object.
        radius (int): Radius for neighbor search.
        p (int): Power parameter for Minkowski metric.
        figcomp (list): Number of rows and columns for subplots.
        xlim (list): Limits for x-axis.
        ylim (list): Limits for y-axis.
        figsize (tuple): Figure size.
        fov_subset (list): Subset of FOVs to plot.
        spatial_key (str): Key for spatial coordinates.
        fov_key (str): Key for field of view.
        save_directory (str): Directory to save figures.
        filename_save (str): Filename to save the figure.

    Returns:
        nn_counts_per_fov (list): List of neighbor counts per FOV.
    """
    fig, axes = plt.subplots(nrows=figcomp[0], ncols=figcomp[1], figsize=figsize, gridspec_kw={'hspace': 0.5, 'wspace': 0.3, 'bottom':0.15})
    sns.set_style('ticks')
    # Flatten the 2D array of subplots into a 1D array
    axes_flat = axes.flatten()
    nn_counts_per_fov = []
    # Loop through unique fields of view (FOV)
    for i, fov in enumerate(fov_subset):
        adata_fov = adata[np.where(adata.obs[fov_key] == fov)[0], :].copy()
        spatial_kdTree = cKDTree(adata_fov.obsm[spatial_key])

        # Query the KD-tree to find neighbors within the specified radius
        nn = spatial_kdTree.query_ball_point(adata_fov.obsm[spatial_key], r=radius, p=p)

        # Count the number of neighbors for each cell
        nn_counts = [len(neighbors) for neighbors in nn]
        nn_counts_per_fov.append(nn_counts)
        # Plot the histogram for the current FOV
        axes_flat[i].hist(nn_counts)
        axes_flat[i].set_title(f'{fov}')
        axes_flat[i].set_xlabel('Number of Neighbors', fontsize = 8)
        axes_flat[i].set_ylabel('Frequency', fontsize = 8)
        axes_flat[i].tick_params(labelsize=10)
        axes_flat[i].set_xlim(xlim[0], xlim[1])
        axes_flat[i].set_ylim(ylim[0], ylim[1])
        mean_counts = np.mean(nn_counts)
        axes_flat[i].axvline(mean_counts, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_counts:.2f}')
        axes_flat[i].legend()

    # Adjust layout
    plt.tight_layout()
    if filename_save is not None:
        plt.savefig(os.path.join(save_directory, filename_save + '.pdf'), bbox_inches = 'tight')
    return nn_counts_per_fov

def plot_nhood_graph(
        mdata,
        alpha = 0.1,
        min_logFC = 0,
        min_size = 10,
        plot_edges = False,
        save = None,
        ax = None,
        vmin = -10,
        vmax = 10,
        save_directory = None,
        filename_save = None,
        xlim = None,
        ylim = None,
        **kwargs,
    ) -> None:
        """Visualize DA results on abstracted graph (wrapper around sc.pl.embedding)

        Args:
            mdata: MuData object
            alpha: Significance threshold. (default: 0.1)
            min_logFC: Minimum absolute log-Fold Change to show results. If is 0, show all significant neighbourhoods. (default: 0)
            min_size: Minimum size of nodes in visualization. (default: 10)
            plot_edges: If edges for neighbourhood overlaps whould be plotted. Defaults to False.
            title: Plot title. Defaults to "DA log-Fold Change".
            show: Show the plot, do not return axis.
            save: If `True` or a `str`, save the figure. A string is appended to the default filename.
                  Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
            **kwargs: Additional arguments to `scanpy.pl.embedding`.
        """
        nhood_adata = mdata["milo"].T.copy()

        if "Nhood_size" not in nhood_adata.obs.columns:
            raise KeyError(
                'Cannot find "Nhood_size" column in adata.uns["nhood_adata"].obs -- \
                    please run milopy.utils.build_nhood_graph(adata)'
            )

        nhood_adata.obs["graph_color"] = nhood_adata.obs["logFC"]
        nhood_adata.obs.loc[nhood_adata.obs["SpatialFDR"] > alpha, "graph_color"] = np.nan
        nhood_adata.obs["abs_logFC"] = abs(nhood_adata.obs["logFC"])
        nhood_adata.obs.loc[nhood_adata.obs["abs_logFC"] < min_logFC, "graph_color"] = np.nan

        # Plotting order - extreme logFC on top
        nhood_adata.obs.loc[nhood_adata.obs["graph_color"].isna(), "abs_logFC"] = np.nan
        ordered = nhood_adata.obs.sort_values("abs_logFC", na_position="first").index
        nhood_adata = nhood_adata[ordered]

        vmax = np.max([nhood_adata.obs["graph_color"].max(), abs(nhood_adata.obs["graph_color"].min())])
        vmin = -vmax

        size = np.array(nhood_adata.obs["Nhood_size"] * min_size)
        size[np.isnan(nhood_adata.obs["Nhood_size"] * min_size)] = 0.5

        g = sc.pl.embedding(
            nhood_adata,
            "X_milo_graph",
            color="graph_color",
            cmap="RdBu_r",
            edges=plot_edges,
            neighbors_key="nhood",
            sort_order=False,
            frameon=False,
            size = size,
            vmax=vmax,
            vmin=vmin,
            save=save,
            ax=ax,
            show = False,
            **kwargs)
        
        if xlim is not None: 
            g.set_xlim(-7, 22.5)
        if ylim is not None:
            g.set_ylim(-10, 22.5)
        plt.savefig(os.path.join(save_directory, filename_save+'.pdf'), bbox_inches = 'tight')

# def da_beeswarm(mdata,
        #         feature_key = "rna",
        #         anno_col: str = "nhood_annotation",
        #         alpha: float = 0.1,
        #         subset_nhoods = None,
        #         figsize = (6, 12),
        #         labels_key = None,
        #         labels_list = None,
        #         patient_key = 'sample',
        #         xlim = None,
        #         percentile = 70,
        #         save_directory = 'figures',
        #         filename_save = None):
        # """Plot beeswarm plot of logFC against nhood labels

        # Args:
        #     mdata: MuData object
        #     anno_col: Column in adata.uns['nhood_adata'].obs to use as annotation. (default: 'nhood_annotation'.)
        #     alpha: Significance threshold. (default: 0.1)
        #     subset_nhoods: List of nhoods to plot. If None, plot all nhoods. (default: None)
        #     palette: Name of Seaborn color palette for violinplots.
        #              Defaults to pre-defined category colors for violinplots.

        # Examples:
        #     >>> import pertpy as pt
        #     >>> import scanpy as sc
        #     >>> adata = pt.dt.bhattacherjee()
        #     >>> milo = pt.tl.Milo()
        #     >>> mdata = milo.load(adata)
        #     >>> sc.pp.neighbors(mdata["rna"])
        #     >>> milo.make_nhoods(mdata["rna"])
        #     >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
        #     >>> milo.da_nhoods(mdata, design="~label")
        #     >>> milo.annotate_nhoods(mdata, anno_col='cell_type')
        #     >>> pt.pl.milo.da_beeswarm(mdata)
        # """
        # try:
        #     nhood_adata = mdata["milo"].T.copy()
        #     nhood_adata.obs[[patient_key, labels_key]] = mdata[feature_key][mdata[feature_key].obs['nhood_ixs_refined'] == 1].obs[[patient_key, labels_key]].values
        # except KeyError:
        #     raise RuntimeError(
        #         "mdata should be a MuData object with two slots: feature_key and 'milo'. Run 'milopy.count_nhoods(adata)' first."
        #     ) from None

        # if subset_nhoods is not None:
        #     nhood_adata = nhood_adata[subset_nhoods]

        # try:
        #     nhood_adata.obs[anno_col]
        # except KeyError:
        #     raise RuntimeError(
        #         f"Unable to find {anno_col} in mdata.uns['nhood_adata']. Run 'milopy.utils.annotate_nhoods(adata, anno_col)' first"
        #     ) from None

        # try:
        #     nhood_adata.obs["logFC"]
        # except KeyError:
        #     raise RuntimeError(
        #         "Unable to find 'logFC' in mdata.uns['nhood_adata'].obs. Run 'core.da_nhoods(adata)' first."
        #     ) from None

        # sorted_annos = (
        #     nhood_adata.obs[[anno_col, "logFC"]].groupby(anno_col).mean().sort_values("logFC", ascending=True).index
        # )

        # anno_df = nhood_adata.obs[[anno_col, "logFC", "SpatialFDR", patient_key, labels_key]].copy()
        # anno_df["is_signif"] = anno_df["SpatialFDR"] < alpha
        # anno_df = anno_df[anno_df[anno_col] != "nan"]

        # prop_df = anno_df.groupby([patient_key, labels_key])[anno_col].value_counts().unstack().fillna(0).astype('bool').astype('int').groupby(labels_key).mean(1)
        # prop_df = prop_df.melt(ignore_index=False)
        # prop_df = prop_df.reset_index()
        # prop_df = prop_df[np.isin(prop_df.loc[:, labels_key], labels_list)]

        # adata_runner = mdata['rna'][mdata['rna'].obs['nhood_ixs_refined'] == 1].copy()
        # adata_runner.obs[anno_col] = mdata['milo'].var.loc[:, anno_col].values
        # adata_runner = adata_runner[np.isin(adata_runner.obs[anno_col], sorted_annos)]
        # adata_runner.obs[anno_col] = pd.Categorical(adata_runner.obs[anno_col], categories=sorted_annos)
        # perc  = quiche.pp.compute_percentile(np.abs(mdata['milo'].var.groupby(anno_col)['logFC'].mean()[sorted_annos]), p = percentile)
        # cmap_df = pd.DataFrame(mdata['milo'].var.groupby(anno_col)['logFC'].mean(), columns = ['logFC'])
        # cmap = np.full(np.shape(mdata['milo'].var.groupby(anno_col)['logFC'].mean())[0], 'lightgrey', dtype = 'object')
        # cmap[mdata['milo'].var.groupby(anno_col)['logFC'].mean() < -1*perc] = '#377eb8'
        # cmap[mdata['milo'].var.groupby(anno_col)['logFC'].mean() > perc] = '#e41a1c'
        # cmap_df['cmap'] = cmap
        # sorted_annos = sorted_annos[np.where((cmap_df.loc[sorted_annos]['logFC'] < -1*perc) | (cmap_df.loc[sorted_annos]['logFC'] > perc))[0]]
        

        # _, ax = plt.subplots(1, 1, figsize = figsize, gridspec_kw={'hspace': 0.45, 'wspace': 0.4, 'bottom':0.15})
        # g = sns.violinplot(
        #         data=anno_df,
        #         y=anno_col,
        #         x="logFC",
        #         order=sorted_annos,
        #         inner=None,
        #         orient="h",
        #         palette= cmap_df.loc[sorted_annos]['cmap'].values,
        #         linewidth=0,
        #         scale="width",
        #         alpha = 0.8,
        #         ax=ax
        #     )

        # g = sns.stripplot(
        #     data=anno_df,
        #     y=anno_col,
        #     x="logFC",
        #     order=sorted_annos,
        #     hue="is_signif",
        #     palette=["grey", "black"],
        #     size=2,
        #     orient="h",
        #     alpha=0.5,
        #     ax = ax
        # )
        
        # g.tick_params(labelsize=12)
        # g.set_xlabel('log2(fold change)', fontsize = 12)
        # g.set_ylabel('annotated niche neighborhoods', fontsize = 12)
        # if xlim is not None:
        #     ax.set_xlim(xlim[0], xlim[1])
        # ax.axvline(x=0, ymin=0, ymax=1, color="black", linestyle="--")
        # ax.legend(loc="upper right", title=f"< {int(alpha * 100)}% spatial FDR", bbox_to_anchor=(1, 1), frameon=False, prop={'size':10}, markerscale=1)
        # plt.savefig(os.path.join(save_directory, filename_save+'.pdf'), bbox_inches = 'tight')
        # return sorted_annos
def da_beeswarm(mdata,
                feature_key = "rna",
                alpha: float = 0.1,
                subset_nhoods = None,
                figsize = (6, 12),
                niche_key = None,
                design_key = 'condition',
                patient_key = 'sample',
                xlim = None,
                percentile = 70,
                save_directory = 'figures',
                filename_save = None):
        """Plot beeswarm plot of logFC against nhood labels

        Args:
            mdata: MuData object
            anno_col: Column in adata.uns['nhood_adata'].obs to use as annotation. (default: 'nhood_annotation'.)
            alpha: Significance threshold. (default: 0.1)
            subset_nhoods: List of nhoods to plot. If None, plot all nhoods. (default: None)
            palette: Name of Seaborn color palette for violinplots.
                     Defaults to pre-defined category colors for violinplots.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata, design="~label")
            >>> milo.annotate_nhoods(mdata, anno_col='cell_type')
            >>> pt.pl.milo.da_beeswarm(mdata)
        """
        try:
            nhood_adata = mdata["milo"].T.copy()
            nhood_adata.obs[[patient_key, design_key]] = mdata[feature_key][mdata[feature_key].obs['nhood_ixs_refined'] == 1].obs[[patient_key, design_key]].values
        except KeyError:
            raise RuntimeError(
                "mdata should be a MuData object with two slots: feature_key and 'milo'. Run 'milopy.count_nhoods(adata)' first."
            ) from None

        if subset_nhoods is not None:
            nhood_adata = nhood_adata[subset_nhoods]

        try:
            nhood_adata.obs[niche_key]
        except KeyError:
            raise RuntimeError(
                f"Unable to find {niche_key} in mdata.uns['nhood_adata']. Run 'milopy.utils.annotate_nhoods(adata, anno_col)' first"
            ) from None

        try:
            nhood_adata.obs["logFC"]
        except KeyError:
            raise RuntimeError(
                "Unable to find 'logFC' in mdata.uns['nhood_adata'].obs. Run 'core.da_nhoods(adata)' first."
            ) from None

        sorted_annos = (
            nhood_adata.obs[[niche_key, "logFC"]].groupby(niche_key).mean().sort_values("logFC", ascending=True).index
        )

        anno_df = nhood_adata.obs[[niche_key, "logFC", "SpatialFDR", patient_key, design_key]].copy()
        anno_df["is_signif"] = anno_df["SpatialFDR"] < alpha
        anno_df = anno_df[anno_df[niche_key] != "nan"]

        mdata['milo'].var[design_key] = mdata['rna'].obs[design_key].values


        adata_runner = mdata['rna'][mdata['rna'].obs['nhood_ixs_refined'] == 1].copy()
        adata_runner.obs[niche_key] = mdata['milo'].var.loc[:, niche_key].values
        adata_runner = adata_runner[np.isin(adata_runner.obs[niche_key], sorted_annos)]
        adata_runner.obs[niche_key] = pd.Categorical(adata_runner.obs[niche_key], categories=sorted_annos)
        perc  = quiche.pp.compute_percentile(np.abs(mdata['milo'].var.groupby(niche_key)['logFC'].mean()[sorted_annos]), p = percentile)
        cmap_df = pd.DataFrame(mdata['milo'].var.groupby(niche_key)['logFC'].mean(), columns = ['logFC'])
        cmap = np.full(np.shape(mdata['milo'].var.groupby(niche_key)['logFC'].mean())[0], 'lightgrey', dtype = 'object')
        cmap[mdata['milo'].var.groupby(niche_key)['logFC'].mean() < -1*perc] = '#377eb8'
        cmap[mdata['milo'].var.groupby(niche_key)['logFC'].mean() >= perc] = '#e41a1c'
        cmap_df['cmap'] = cmap
        if percentile != 0:
            sorted_annos = sorted_annos[np.where((cmap_df.loc[sorted_annos]['logFC'] < -1*perc) | (cmap_df.loc[sorted_annos]['logFC'] >= perc))[0]]
        

        _, ax = plt.subplots(1, 1, figsize = figsize, gridspec_kw={'hspace': 0.45, 'wspace': 0.4, 'bottom':0.15})
        g = sns.violinplot(
                data=anno_df,
                y=niche_key,
                x="logFC",
                order=sorted_annos,
                inner=None,
                orient="h",
                palette= cmap_df.loc[sorted_annos]['cmap'].values,
                linewidth=0,
                scale="width",
                alpha = 0.8,
                ax=ax
            )

        g = sns.stripplot(
            data=anno_df,
            y=niche_key,
            x="logFC",
            order=sorted_annos,
            hue="is_signif",
            palette=["grey", "black"],
            size=2,
            orient="h",
            alpha=0.5,
            ax = ax
        )
        
        g.tick_params(labelsize=12)
        g.set_xlabel('log2(fold change)', fontsize = 12)
        g.set_ylabel('annotated niche neighborhoods', fontsize = 12)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        ax.axvline(x=0, ymin=0, ymax=1, color="black", linestyle="--")
        ax.legend(loc="upper right", title=f"< {int(alpha * 100)}% spatial FDR", bbox_to_anchor=(1, 1), frameon=False, prop={'size':10}, markerscale=1)
        min_name = mdata['milo'].var.groupby(design_key)['logFC'].mean().idxmin()
        max_name = mdata['milo'].var.groupby(design_key)['logFC'].mean().idxmax()
        plt.title(f'{min_name} vs {max_name}', fontsize = 12)
        plt.savefig(os.path.join(save_directory, filename_save+'.pdf'), bbox_inches = 'tight')
        return sorted_annos


def plot_neighborhood_dist(mdata, save_directory, filename_save):
    """
    Plot histogram of neighborhood sizes.

    Args:
        mdata (MuData): MuData object.
        save_directory (str): Directory to save the figure.
        filename_save (str): Filename to save the figure.
    """
    _, axes = plt.subplots(1, 1, figsize = (4, 3.5), gridspec_kw={'hspace': 0.45, 'wspace': 0.4, 'bottom':0.15})
    sns.set_style('ticks')
    nhood_size = np.array(mdata['rna'].obsm["nhoods"].sum(0)).ravel()
    axes.hist(nhood_size, bins=50)
    axes.tick_params(labelsize=12)
    plt.xlabel('number of cells in neighborhood', fontsize = 12)
    plt.ylabel('number of neighborhoods', fontsize = 12)
    plt.savefig(os.path.join(save_directory, filename_save+'.pdf'), bbox_inches = 'tight')

def plot_grid_enrichment(df, hue_order, colors_dict, selected_grid_list, num_grids_x, num_grids_y, save_directory, filename_save):
    """
    Plot grid enrichment.

    Args:
        df (DataFrame): DataFrame containing grid data.
        hue_order (list): Order of hue levels.
        colors_dict (dict): Dictionary mapping hue levels to colors.
        selected_grid_list (list): List of selected grid coordinates.
        num_grids_x (int): Number of grid cells in the x direction.
        num_grids_y (int): Number of grid cells in the y direction.
        save_directory (str): Directory to save the figure.
        filename_save (str): Filename to save the figure.
    """
    _, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    # Plot the points with colors representing the labels
    sns.scatterplot(x = 'x', y = 'y', hue = 'group', data =df, alpha=0.5, palette=colors_dict, hue_order = hue_order, ax = axes)
    axes.tick_params(labelsize=10)
    # Set axis labels and legend
    plt.xlabel('Y', fontsize = 12)
    plt.ylabel('X', fontsize = 12)
    plt.xlim(0, 1)
    plt.ylim(0,1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
        # Outline the selected grid box in red
    for selected_grid in selected_grid_list:
        selected_grid_rect = Rectangle(
            ((selected_grid[0] - 1) * (1 / num_grids_x), (selected_grid[1] - 1) * (1 / num_grids_y)),
            (1 / num_grids_x), (1 / num_grids_y),
            edgecolor='k', linewidth=1.5, fill=False
        )
        plt.gca().add_patch(selected_grid_rect)
    plt.savefig(os.path.join(save_directory, filename_save+'.pdf'), bbox_inches = 'tight')
