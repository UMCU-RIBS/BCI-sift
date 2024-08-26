# -------------------------------------------------------------
# HandDecoding
# Copyright (c) 2023
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

import os
from copy import copy
from typing import List, Tuple, Optional

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from src.optimizer.backend._backend import channel_id_to_int

matplotlib.use('Agg')  # Use the 'Agg' backend for PNG rendering


def elimination_plot(
        grid: np.ndarray, result_grid: pd.DataFrame, output_path: str,
        identifier: str = '', ied_ed: Optional[Tuple[float, float]] = None, seed: Optional[int] = None,
) -> None:
    """
    Generates and saves a plot visualizing the maximum and all scores across
    different subgrid sizes.

    The function scales the scores by 100 for better readability, identifies the
    maximal scores within each subgrid size, and visualizes both the maximum scores
    (as a line plot) and all scores (as scatter and density plots) to illustrate
    the score distribution across different subgrid sizes.

    Parameters:
    -----------
    :param grid: numpy.ndarray
        The grid structure specifying how channels are arranged.
    :param result_grid: pd.DataFrame
        The result grid containing the optimization diagnostics from the optimization algorithm.
    :param output_path: str
        The output path where the plot will be saved.
    :param identifier: str
        An identifier to customize the save file (e.g. subject or condition).
    :param ied_ed: Optional[Tuple[float, float]], default = None
        Inter electrode Distance (IED) and Electrode Diameter (ED)
        to compute the surface area in millimeter. Default uses the
        number of electrodes for plotting
    :param seed: Optional[int], default = None
        Random seed for reproducibility of the evolutionary process.

    Returns:
    --------
    :return: None
    """
    # Handle Dir
    os.makedirs(output_path, exist_ok=True)

    # Copy and scale the result grid for readability
    result_grid = copy(result_grid)
    result_grid['Mean (Score)'] *= 100  # Scale score for readability

    metric = result_grid['Metric'][0].capitalize().replace('_', ' ')

    target = 'Size'
    x_label = 'Grid Size (channels)'
    if ied_ed is not None:
        if all([method in ['SpatialExhaustiveSearch', 'SpatialStochasticHillClimbing']
                for method in set(result_grid['Method'].values)]):
            target = 'Area'
            x_label = 'Grid Size (in $mm^{2}$)'

            ied, ed = ied_ed
            width = (result_grid['Width'] - 1) * ied + 0.5 * ed
            height = (result_grid['Height'] - 1) * ied + 0.5 * ed
            result_grid['Area'] = width * height
        else:
            UserWarning(f'Computation of the Area (using ied_ed = {ied_ed}) is only possible for SES and SSHC')

    # Determine the x-axis bounds
    xlim_max, xlim_min = result_grid[target].max(), result_grid[target].min()

    # Separate maximum scores for clarity in visualization
    # max_scores = result_grid.loc[result_grid.groupby(target)['Mean (Score)'].idxmax()].sort_values(by=target)
    # rest_scores = result_grid.drop(max_scores.index).sort_values(by=target)

    # Create figure and gridspec layout
    width, height = 4 * (grid.size / 32), 8
    fig = plt.figure(figsize=(width, height))
    grid_plot = plt.GridSpec(2, 2, hspace=0.2 / height, wspace=0.2 / width,
                             height_ratios=(1, 8), width_ratios=(4 * (grid.size / 32), 1))

    # Main plot
    ax_main = fig.add_subplot(grid_plot[1:, :-1])
    # Marginal plots
    ax_top_marginal = fig.add_subplot(grid_plot[0, :-1], sharex=ax_main)
    ax_right_marginal = fig.add_subplot(grid_plot[1:, -1], sharey=ax_main)

    # Main plots
    sns.scatterplot(data=result_grid, x=target, y='Mean (Score)', hue='Iteration', alpha=0.5,  # label='All Grids',
                    ax=ax_main)
    sns.lineplot(data=result_grid, x=target, y='Mean (Score)', marker='o', color='forestgreen', alpha=0.75,
                 label='Best Grids', estimator='max', ci=95, err_style=None, linewidth=4.0, ax=ax_main,
                 seed=seed)
    sns.lineplot(data=result_grid, x=target, y='Mean (Score)', marker='o', color='navy', alpha=0.75,
                 label='Median Grids', estimator='median', ci=95, err_style=None, linewidth=4.0, ax=ax_main,
                 seed=seed)
    sns.kdeplot(data=result_grid, x=target, y='Mean (Score)', ax=ax_main, fill=True, color='black', alpha=0.5)

    # Marginal plots
    sns.kdeplot(data=result_grid, x=target, ax=ax_top_marginal, fill=False, color='black', alpha=0.8)
    sns.kdeplot(data=result_grid, y='Mean (Score)', ax=ax_right_marginal, fill=False, color='black', alpha=0.8)

    # Axes and plot formatting
    ax_main.set_xlim(0, (xlim_max + 1 if target == 'Size' else xlim_max + xlim_min))
    ax_main.set_ylim(0, 101.5)
    xtick = int(xlim_max / (8 * (grid.size / 32)))
    ax_main.set_xticks(np.arange(0, xlim_max + (xtick if target == 'Size' else 0), xtick))
    ax_main.set_yticks(np.arange(0, 110, 10))
    ax_main.tick_params(axis='x', rotation=45)

    ax_main.spines[['right', 'top']].set_visible(False)
    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(f'{metric} Score (in %)')
    ax_main.legend().remove()
    ax_main.grid(which='major', linewidth=0.3, linestyle='--', color='grey', alpha=0.25)

    # Extract the legend handles and labels
    handles, labels = ax_main.get_legend_handles_labels()
    ax_main.legend(handles=handles, labels=labels, bbox_to_anchor=(.66, -.15),
                   borderaxespad=0., shadow=True)

    # Remove labels and ticks for marginal plots
    ax_top_marginal.spines[['left', 'top', 'right']].set_visible(False)
    ax_right_marginal.spines[['bottom', 'right', 'top']].set_visible(False)
    ax_top_marginal.set_xlabel('')
    ax_top_marginal.set_ylabel('')
    ax_right_marginal.set_xlabel('')
    ax_right_marginal.set_ylabel('')
    ax_top_marginal.xaxis.set_tick_params(which='both', labelbottom=False)
    ax_top_marginal.yaxis.set_tick_params(which='both', labelleft=False)
    ax_right_marginal.xaxis.set_tick_params(which='both', labelbottom=False)
    ax_right_marginal.yaxis.set_tick_params(which='both', labelleft=False)
    ax_top_marginal.yaxis.set_ticks([])
    ax_right_marginal.xaxis.set_ticks([])

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{output_path}/{identifier}_{target}_channel_elimination_plot.png',
                bbox_inches='tight', dpi=100 + np.log(grid.size) * 200)
    plt.close()


# f'{output_path}/_{condition}_{subject}_{target}_channel_elimination_plot.png'

def importance_plot(
        grid: np.ndarray, result_grid: pd.DataFrame, output_path: str, identifier: str = '',
        viz='distribution', bads: Optional[List[str]] = None, top_k: int = 5
) -> None:
    """
    Generates and saves a heatmap visualizing the importance of each channel
    within the grid.

    The function scales the scores by 100 for readability, computes the average
    scores for each channel ID, and creates a heatmap to illustrate the score distribution
     across the channel grid.

    Parameters:
    -----------
    :param grid: numpy.ndarray
        The grid structure specifying how channels are arranged.
    :param result_grid: pd.DataFrame
        The result grid containing the optimization diagnostics from the optimization algorithm.
    :param output_path: str
        The output path where the plot will be saved.
    :param identifier: str
        An identifier to customize the save file (e.g. subject or condition).
    :param viz: str, default = 'distribution'
        Type of visualization used. Chose from 'heat_map', 'dist', 'grid_overlay'.
    :param bads: List[str], default = ''
        A list of bad channels. Bad channels will be marked with a red square.
    :param top_k: int, default = 5
        Number of top subgrids to plot.
    Returns:
    --------
    :return: None
    """

    # Handle Dir
    os.makedirs(output_path, exist_ok=True)

    metric = result_grid['Metric'][0].capitalize().replace('_', ' ')

    result_grid = copy(result_grid)
    new_column = result_grid['Mask'].apply(lambda x: grid[x.reshape(grid.shape)])
    result_grid.insert(2, 'Channel IDs', new_column)
    # result_grid['Channel IDs'] = result_grid['Mask'].apply(lambda x: grid[x.reshape(grid.shape)])
    result_grid['Mean (Score)'] *= 100  # Scale score for readability
    bad_channels = channel_id_to_int(bads)  # Bad channel IDs

    # Sort by score and select top k grids
    top_grids = result_grid.nlargest(top_k, 'Mean (Score)')

    top_channels_list = [np.array(grid) for grid in top_grids['Channel IDs']]

    # Weighted average scores for each channel ID
    weighted_scores = {}
    for row in grid:
        for channel_id in row:
            # Extract rows containing the current channel_id and calculate weighted score
            channel_rows = result_grid[
                result_grid['Channel IDs'].apply(lambda ids: channel_id in [int(i) for i in ids])]

            weighted_score = np.average(channel_rows['Mean (Score)'])  # , weights=1 / channel_rows['Grid Size'])
            weighted_scores[channel_id] = weighted_score

    channel_scores = pd.Series(weighted_scores)

    # Prepare the data for heatmap
    numerical_data = np.array([[channel_scores.get(channel_id, np.nan) if channel_id not in bad_channels else np.nan
                                for channel_id in row] for row in grid])

    # Extract performance scores for each channel
    channel_scores_dist = {}
    for row in grid:
        for channel_id in row:
            # Extract rows containing the current channel_id
            channel_rows = result_grid[
                result_grid['Channel IDs'].apply(lambda ids: channel_id in [int(i) for i in ids])]
            scores = channel_rows['Mean (Score)'].values
            channel_scores_dist[channel_id] = scores

    bins = int(grid.size / 2)

    # Determine the maximum frequency for y-axis limits
    max_freq = 0
    xlim_min, xlim_max = 100, 0
    for scores in channel_scores_dist.values():
        if len(scores) > 0:
            hist, edge = np.histogram(scores, bins=bins, density=True)
            max_freq = max(max_freq, max(hist))
            xlim_min = min(xlim_min, min(edge))
            xlim_max = max(xlim_max, max(edge))

    # Adjust figure size based on grid dimensions
    fig, main_ax = plt.subplots(figsize=(grid.shape[1], grid.shape[0]))
    font_size = int(10 + grid.size * 0.1)

    # Custom colormap (to avoid conflicts with malfunctioning channels)
    greens = plt.get_cmap('Greens')
    colors = greens(np.linspace(0, 1, 256))
    custom_greens = LinearSegmentedColormap.from_list('custom_greens', colors[64:])

    # Create a divider for existing axes instance
    cax = fig.add_axes([1.0, .3, .03, .4])

    # Heatmap visualization and customization
    sns.heatmap(data=numerical_data, fmt="", cmap=custom_greens, cbar=True, cbar_ax=cax, ax=main_ax)
    cbar = main_ax.collections[0].colorbar
    cbar.set_label(f'Average {metric} Score', size=font_size)
    cbar.ax.tick_params(labelsize=font_size - 5)

    main_ax.set_title(f'Channel Importance for {identifier}', fontsize=font_size)
    main_ax.set_xticks([])
    main_ax.set_yticks([])

    # Check whether a spatial algorithm was used?
    spatial = all([method in ['SpatialExhaustiveSearch', 'SpatialStochasticHillClimbing']
                   for method in set(result_grid['Method'].values)])
    for row_idx, row in enumerate(grid):
        for col_idx, channel_id in enumerate(row):
            ax = plt.subplot2grid((grid.shape[0], grid.shape[1]), (row_idx, col_idx), fig=fig)
            if channel_id in bad_channels:
                ax.add_patch(
                    Rectangle((0, 0), 1, 1, fill=True, facecolor='white', linewidth=0)
                )
            elif viz == 'grid_overlay' and not spatial:
                if channel_id in grid[result_grid['Mask'] == max(result_grid['Mean (Score)'])]:
                    ax.add_patch(
                        Rectangle((0, 0), 1, 1, fill=False, edgecolor='navy',
                                  hatch='/', linewidth=1, alpha=0.7)
                    )
                ax.text(0.95, 0.95, f"{channel_id}", horizontalalignment='right', verticalalignment='top',
                        transform=ax.transAxes, fontsize=9 - grid.size * 0.02, fontweight='bold',
                        color='white')
            elif viz == 'distribution':
                ax.text(0.95, 0.95, f"{channel_id}", horizontalalignment='right', verticalalignment='top',
                        transform=ax.transAxes, fontsize=9 - grid.size * 0.02, fontweight='bold',
                        color='white')

                scores = channel_scores_dist.get(channel_id, [])
                if len(scores) > 0:
                    sns.histplot(scores, bins=bins, kde=False, color='black', edgecolor='black',
                                 ax=ax, stat="density", fill=False)
                    sns.kdeplot(scores, color='black', ax=ax, linewidth=1.5)
                ax.set_ylim(0, max_freq * 1.1)
                ax.set_xlim(-5, 105)
            elif viz == 'heat_map' or (viz == 'grid_overlay' and spatial):
                ax.text(0.95, 0.95, f"{channel_id}", horizontalalignment='right', verticalalignment='top',
                        transform=ax.transAxes, fontsize=9 - grid.size * 0.02, fontweight='bold',
                        color='white')
                score = channel_scores.get(channel_id, np.nan)
                ax.text(0.5, 0.5, f"{score:.1f}", horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=7 - grid.size * 0.02, fontweight='bold',
                        color='white')
            else:
                raise ValueError(f'Visualization method is not supported. Got {viz}')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel('')  # Remove density label
            ax.patch.set_alpha(0)  # Make background transparent
            # Remove axis spines
            for spine in ax.spines.values():
                spine.set_visible(False)

    if viz == 'grid_overlay' and spatial:
        # Plot top k sub-grids with colors from Blues colormap
        blues = plt.get_cmap('Blues_r')
        top_k_colors = blues(np.linspace(0, 0.5, top_k))

        for channels, color, k in zip(top_channels_list, top_k_colors, range(top_k)):
            x_min, x_max, y_min, y_max = float('inf'), -float('inf'), float('inf'), -float('inf')
            for row_idx, row in enumerate(grid):
                for col_idx, channel_id in enumerate(row):
                    if channel_id in channels:
                        x_min, x_max = min(x_min, col_idx), max(x_max, col_idx)
                        y_min, y_max = min(y_min, row_idx), max(y_max, row_idx)
                if x_min <= x_max and y_min <= y_max:
                    rect = Rectangle((x_min, y_min), x_max - x_min + 1, y_max - y_min + 1, fill=False,
                                     edgecolor=color, linewidth=top_k / 1.5 + 2 / 3 - k / 1.5, alpha=0.7)
                    main_ax.add_patch(rect)

    for _, spine in main_ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)  # Make cells adjacent to each other
    plt.savefig(f'{output_path}/{identifier}_channel_importance_{viz}_plot.png',
                bbox_inches='tight', dpi=100 + np.log(grid.size) * 200)
    plt.close()
