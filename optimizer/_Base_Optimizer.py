# -------------------------------------------------------------
# Channel Elimination
# Copyright (c) 2024
#       Dirk Keller,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

import os
from copy import copy
from typing import Tuple, List, Union, Dict, Any, Optional, Type

import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy import stats
from sklearn.base import TransformerMixin, MetaEstimatorMixin
from sklearn.metrics import get_scorer
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
# from sklearn.utils._metadata_requests import _RoutingNotSupportedMixin
from sklearn.utils.validation import check_is_fitted as sklearn_is_fitted

from .utils import to_dict_keys, channel_id_to_int

matplotlib.use('Agg')  # Use the 'Agg' backend for PNG rendering


class BaseOptimizer(MetaEstimatorMixin, TransformerMixin):  # _RoutingNotSupportedMixin
    """
    Base class for all channel optimizers that provides framework
    functionalities such as estimator serialization, cross-validation
    strategy setup, parameter and data validation.

    Optimizes channel combinations within a grid for a given performance
    metric using a specified machine learning model or pipeline.

    Parameters:
    -----------
    :param grid: numpy.ndarray
        The grid structure specifying how channels (IDs) are arranged.
    :param estimator: Union[Any, Pipeline]
        The machine learning model or pipeline to evaluate feature sets.
    :param metric: str, default = 'f1_weighted'
        The metric to optimize. Must be scikit-learn compatible.
    :param cv: Union[BaseCrossValidator, int], default = 10
        The cross-validation strategy or number of folds.
    :param groups: numpy.ndarray, default = None
        Groups for LeaveOneGroupOut-crossvalidator
    :param n_jobs: int, default = 1
        The number of parallel jobs to run during cross-validation.
    :param seed: Optional[int], default = None
        Setting a seed to fix randomness (for reproduceability).
        Default does not use a seed.
    :param verbose: Union[bool, int], default = False
         If set to True, enables the output of progress status
         during the optimization process.

    Methods:
    --------
    - fit:
        Abstract method that must be implemented by subclasses, defining
        the algorithm design and fit to the data.
    - transform:
        Abstract method that must be implemented by subclasses, executing
        the transformation for the data with the optimizer result.
    - run:
        Abstract method that must be implemented by subclasses, defining
        the specific steps of the optimization process.
    - evaluate_candidates:
        Evaluates the selected features using cross-validation or train-test split.
    - objective_function:
        Calculates the score to maximize or minimize based on the provided mask.
    - elimination_plot:
        Generates and saves a plot visualizing the maximum and all scores across different subgrid sizes.
    - importance_plot:
        Generates and saves a heatmap visualizing the importance of each channel within the grid.

    Notes:
    ------
    This implementation is semi-compatible with the scikit-learn framework,
    which builds around two-dimensional feature matrices. To use this
    transformation within a scikit-learn Pipeline, the four dimensional data
    must be flattened after the first dimension [samples, features]. For example,
    scikit-learn's FunctionTransformer can achieve this.

    Returns:
    --------
    :return: None
    """

    def __init__(
            self,

            # General and Decoder
            grid: numpy.ndarray,
            estimator: Union[Any, Pipeline],
            metric: str = 'f1_weighted',
            cv: Union[BaseCrossValidator, int] = 10,
            groups: numpy.ndarray = None,
            # Misc
            n_jobs: int = 1,
            seed: Optional[int] = None,
            verbose: bool = False
    ) -> None:

        self.grid = grid
        self.estimator = estimator
        self.metric = metric
        self.cv = cv
        self.groups = groups
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

    def fit(
            self, X: numpy.ndarray, y: numpy.ndarray = None
    ) -> NotImplementedError:
        """
        Fit method to fit the optimizer to the data.

        The subclass should overwrite this method. Note it must implement:
            self.X_
            self.y_
            self.iter_
            self.result_grid_

        Parameters:
        -----------
        :param X: numpy.ndarray
            Array-like of the shape (n_sampels, n_features)
        :param y: numpy.ndarray, default = None
            Array-like of shape (n_targets). Target is ignored.

        Raises:
        -------
        :raises NotImplementedError:
            If the method is not implemented by the subclass.
        """
        self.X_ = X
        self.y_ = y

        self.iter_ = int(0)
        cv = self.cv if self.cv is int else self.cv.get_n_splits()
        self.result_grid_ = []

        self.solution_, self.mask_, self.score_ = self.run()

        # Conclude the result grid
        self.result_grid_ = pd.concat(self.result_grid_, axis=0, ignore_index=True)
        raise NotImplementedError('The fit method must be implemented by subclasses')

    def transform(
            self, X: numpy.ndarray, y: numpy.ndarray = None
    ) -> NotImplementedError:
        """
        Transforms the input with the mask obtained from
        the solution of Optimization process.

        Parameters:
        -----------
        :param X: numpy.ndarray
            Array-like with dimensions [samples, channel_height, channel_width, time]
        :param y: numpy.ndarray, default = None
            Array-like with dimensions [targets].

        Return:
        -------
        :return: numpy.ndarray
            Returns a filtered array-like with dimensions
            [samples, channel_height, channel_width, time]
        """
        raise NotImplementedError('The transform method must be implemented by subclasses')

    def run(
            self
    ) -> NotImplementedError:
        """
        Abstract method defining the main optimization procedure.

        This method must be implemented by subclasses.

        Raises:
        -------
        :raises NotImplementedError:
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError('The run method must be implemented by subclasses')

    def evaluate_candidates(
            self, selected_features: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Evaluate the given features using cross-validation or train-test split.

        Parameters:
        -----------
        :param selected_features: numpy.ndarray
            The selected features to evaluate.

        Returns:
        --------
        :return: numpy.ndarray
            The evaluation scores for the selected features.
        """
        if self.cv == 1:
            # Use train-test split instead of cross-validation
            X_train, X_test, y_train, y_test = train_test_split(selected_features, self.y_, test_size=0.2,
                                                                random_state=self.seed)
            self.estimator.fit(X_train, y_train)
            scorer = get_scorer(self.metric)
            scores = scorer(self.estimator, X_test, y_test)
        else:
            scores = cross_val_score(self.estimator, selected_features, self.y_, scoring=get_scorer(self.metric),
                                     cv=self.cv, groups=self.groups, n_jobs=self.n_jobs)
        return scores

    def objective_function(
            self, mask: numpy.ndarray
    ) -> float:
        """
        Objective function that calculates the score to maximize/minimize.

        Parameters:
        -----------
        :param mask: numpy.ndarray
            The boolean mask indicating selected features.

        Returns:
        --------
        :return: float
            The evaluation score for the selected features,
            or -inf if no features are selected.
        """
        self.iter_ += 1

        if np.array(mask).dtype != bool:
            mask = np.array(mask) > 0.5  # Convert to boolean mask
        if len(mask.shape) > 1:
            mask = mask.reshape(-1)
        if not np.any(mask):
            return float('-inf')  # No features selected
        selected_features = self.X_.reshape((self.X_.shape[0], self.X_.shape[1] * self.X_.shape[2], self.X_.shape[3]))[
                            :, mask, :]
        selected_features = selected_features.reshape(self.X_.shape[0], -1)  # Flatten the selected features
        scores = self.evaluate_candidates(selected_features)
        self.save_statistics(mask.reshape(self.grid.shape), scores)

        return scores.mean()

    def save_statistics(
            self, mask: numpy.ndarray, scores: numpy.ndarray
    ) -> None:
        """
        Saves the diagnostics at a given iteration. The
        diagnostics include: Method, Iteration, Mask,
        Channel IDs, Size, Mean (Score), Median (Score),
        SD (Score), CI Lower, CI Upper and the score from
        the crossvaldiation folds (if selected)

        Parameters:
        -----------
        :param mask: numpy.ndarray
            The boolean mask indicating selected features.
        :param scores: numpy.ndarray
            The array of cross-validation scores.

        Returns:
        --------
        :return: None
        """
        channel_ids = self.grid[mask].flatten().tolist()

        new_row = {
            'Method': self.__class__.__name__,
            'Iteration': self.iter_,
            'Mask': [mask],
            'Channel IDs': to_dict_keys(channel_ids),
            'Size': len(channel_ids)
        }

        cv = self.cv if isinstance(self.cv, int) else self.cv.get_n_splits(groups=self.groups)
        if cv == 1:
            new_row['TrainTest (Score)'] = scores  # To maintain consistency with the CV case
        else:
            ci = stats.t.interval(0.95, len(scores) - 1, loc=np.mean(scores), scale=stats.sem(scores))
            cv_stats = {
                'Mean (Score)': np.round(np.mean(scores), 5),
                'Median (Score)': np.round(np.median(scores), 5),
                'Std (Score)': np.round(np.std(scores), 5),
                'CI Lower': np.round(ci[0], 5),
                'CI Upper': np.round(ci[1], 5),
            }
            for i in range(len(scores)):
                cv_stats[f'Fold {i + 1}'] = scores[i]
            new_row = {**new_row, **cv_stats}

        # Add the new row to the history
        self.result_grid_.append(pd.DataFrame(new_row))

    def elimination_plot(
            self, condition: str, output_path: str, subject: str, ied_ed: Union[Tuple[float, float], None] = None
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
        :param condition : str
            Description or condition under which the plot is generated
            (e.g., specific subject or condition).
        :param output_path : str
            The output path where the plot will be saved.
        :param subject : str
            Subject name to customize the save file.
        :param ied_ed : Union[Tuple[float, float], None], default = None
            Inter electrode Distance (IED) and Electrode Diameter (ED)
            to compute the surface area in millimeter. Default uses the
            number of electrodes for plotting

        Returns:
        --------
        :return: None
        """
        sklearn_is_fitted(self)

        # Handle Dir
        os.makedirs(output_path, exist_ok=True)

        # Copy and scale the result grid for readability
        result_grid = copy(self.result_grid_)
        result_grid['Mean (Score)'] *= 100  # Scale score for readability

        metric = self.metric.split('_')[0].capitalize()

        target = 'Size'
        x_label = 'Grid Size (channels)'
        if ied_ed is not None:
            if self.__class__.__name__ in ['SpatialExhaustiveSearch', 'SpatialStochasticHillClimbing']:
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
        width, height = 4 * (self.grid.size / 32), 8
        fig = plt.figure(figsize=(width, height))
        grid = plt.GridSpec(2, 2, hspace=0.2 / height, wspace=0.2 / width,
                            height_ratios=(1, 8), width_ratios=(4 * (self.grid.size / 32), 1))

        # Main plot
        ax_main = fig.add_subplot(grid[1:, :-1])
        # Marginal plots
        ax_top_marginal = fig.add_subplot(grid[0, :-1], sharex=ax_main)
        ax_right_marginal = fig.add_subplot(grid[1:, -1], sharey=ax_main)

        # Main plots
        sns.scatterplot(data=result_grid, x=target, y='Mean (Score)', hue='Iteration', alpha=0.5,  # label='All Grids',
                        ax=ax_main)
        sns.lineplot(data=result_grid, x=target, y='Mean (Score)', marker='o', color='forestgreen', alpha=0.75,
                     label='Best Grids', estimator='max', ci=95, err_style=None, linewidth=4.0, ax=ax_main,
                     seed=self.seed)
        sns.lineplot(data=result_grid, x=target, y='Mean (Score)', marker='o', color='navy', alpha=0.75,
                     label='Median Grids', estimator='median', ci=95, err_style=None, linewidth=4.0, ax=ax_main,
                     seed=self.seed)
        sns.kdeplot(data=result_grid, x=target, y='Mean (Score)', ax=ax_main, fill=True, color='black', alpha=0.5)

        # Marginal plots
        sns.kdeplot(data=result_grid, x=target, ax=ax_top_marginal, fill=False, color='black', alpha=0.8)
        sns.kdeplot(data=result_grid, y='Mean (Score)', ax=ax_right_marginal, fill=False, color='black', alpha=0.8)

        # Axes and plot formatting
        ax_main.set_xlim(0, (xlim_max + 1 if target == 'Size' else xlim_max + xlim_min))
        ax_main.set_ylim(0, 101.5)
        xtick = int(xlim_max / (8 * (self.grid.size / 32)))
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
        plt.savefig(f'{output_path}/{condition}_{subject}_{target}_channel_elimination_plot.png',
                    bbox_inches='tight', dpi=100 + np.log(self.grid.size) * 200)
        plt.close()

    def importance_plot(
            self, condition: str, output_path: str, subject: str = '', viz='distribution',
            bads: List[str] = [], export: bool = False, top_k: int = 5
    ) -> None:
        """
        Generates and saves a heatmap visualizing the importance of each channel
        within the grid.

        The function scales the scores by 100 for readability, computes the average
        scores for each channel ID, and creates a heatmap to illustrate the score distribution
         across the channel grid.

        Parameters:
        -----------
        :param condition : str
            Description or condition under which the plot is generated
            (e.g., specific subject or condition).
        :param output_path : str
            The output path where the heatmap will be saved.
        :param subject : str, default = ''
            Subject name to customize the save file.
        :param viz : str, default = 'distribution'
            Type of visualization used. Chose from 'heat_map', 'dist', 'grid_overlay'.
        :param bads : List[str], default = ''
            A list of bad channels. Bad channels will be marked with a red square.
        :param export : bool, default = True
            Export the importance values to a .mat file.
        :param top_k : int, default = 5
            Number of top subgrids to plot.
        Returns:
        --------
        :return: None
        """

        sklearn_is_fitted(self)

        # Handle Dir
        os.makedirs(output_path, exist_ok=True)

        result_grid = self.result_grid_.copy()
        result_grid['Mean (Score)'] *= 100  # Scale score for readability
        bad_channels = channel_id_to_int(bads)  # Bad channel IDs

        # Compute grid sizes and add as a column in result_grid
        result_grid['Grid Size'] = result_grid['Channel IDs'].apply(lambda ids: len(ids.split('-')))

        # Sort by score and select top k grids
        top_grids = result_grid.nlargest(top_k, 'Mean (Score)')

        top_channels_list = [set(int(i) for i in grid['Channel IDs'].split('-')) for _, grid in top_grids.iterrows()]

        # Weighted average scores for each channel ID
        weighted_scores = {}
        for row in self.grid:
            for channel_id in row:
                # Extract rows containing the current channel_id and calculate weighted score
                channel_rows = result_grid[
                    result_grid['Channel IDs'].apply(lambda ids: channel_id in [int(i) for i in ids.split('-')])]
                weighted_score = np.average(channel_rows['Mean (Score)'])  # , weights=1 / channel_rows['Grid Size'])
                weighted_scores[channel_id] = weighted_score

        channel_scores = pd.Series(weighted_scores)

        # Prepare the data for heatmap
        numerical_data = np.array([[channel_scores.get(channel_id, np.nan) if channel_id not in bad_channels else np.nan
                                    for channel_id in row] for row in self.grid])

        # Extract performance scores for each channel
        channel_scores_dist = {}
        for row in self.grid:
            for channel_id in row:
                # Extract rows containing the current channel_id
                channel_rows = result_grid[
                    result_grid['Channel IDs'].apply(lambda ids: channel_id in [int(i) for i in ids.split('-')])]
                scores = channel_rows['Mean (Score)'].values
                channel_scores_dist[channel_id] = scores

        bins = int(self.grid.size / 2)

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
        fig, main_ax = plt.subplots(figsize=(self.grid.shape[1], self.grid.shape[0]))
        font_size = int(10 + self.grid.size * 0.1)

        # Custom colormap (to avoid conflicts with malfunctioning channels)
        greens = plt.get_cmap('Greens')
        colors = greens(np.linspace(0, 1, 256))
        custom_greens = LinearSegmentedColormap.from_list('custom_greens', colors[64:])

        # Create a divider for existing axes instance
        cax = fig.add_axes([1.0, .3, .03, .4])

        # Heatmap visualization and customization
        sns.heatmap(data=numerical_data, fmt="", cmap=custom_greens, cbar=True, cbar_ax=cax, ax=main_ax)
        cbar = main_ax.collections[0].colorbar
        cbar.set_label(f'Average {self.metric.split("_")[0].capitalize()} Score', size=font_size)
        cbar.ax.tick_params(labelsize=font_size - 5)

        main_ax.set_title(f'Channel Importance for {condition}', fontsize=font_size)
        main_ax.set_xticks([])
        main_ax.set_yticks([])

        # Check whether a spatial algorithm was used?
        spatial = all([method in ['SpatialExhaustiveSearch', 'SpatialStochasticHillClimbing']
                       for method in set(result_grid['Method'].values)])
        for row_idx, row in enumerate(self.grid):
            for col_idx, channel_id in enumerate(row):
                ax = plt.subplot2grid((self.grid.shape[0], self.grid.shape[1]), (row_idx, col_idx), fig=fig)
                if channel_id in bad_channels:
                    ax.add_patch(
                        Rectangle((0, 0), 1, 1, fill=True, facecolor='white', linewidth=0)
                    )
                elif viz == 'grid_overlay' and not spatial:
                    if channel_id in self.grid[self.mask_]:
                        ax.add_patch(
                            Rectangle((0, 0), 1, 1, fill=False, edgecolor='navy',
                                      hatch='/', linewidth=1, alpha=0.7)
                        )
                    ax.text(0.95, 0.95, f"{channel_id}", horizontalalignment='right', verticalalignment='top',
                            transform=ax.transAxes, fontsize=9 - self.grid.size * 0.02, fontweight='bold',
                            color='white')
                elif viz == 'distribution':
                    ax.text(0.95, 0.95, f"{channel_id}", horizontalalignment='right', verticalalignment='top',
                            transform=ax.transAxes, fontsize=9 - self.grid.size * 0.02, fontweight='bold',
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
                            transform=ax.transAxes, fontsize=9 - self.grid.size * 0.02, fontweight='bold',
                            color='white')
                    score = channel_scores.get(channel_id, np.nan)
                    ax.text(0.5, 0.5, f"{score:.1f}", horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=7 - self.grid.size * 0.02, fontweight='bold',
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
                for row_idx, row in enumerate(self.grid):
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
        plt.savefig(f'{output_path}/{condition}_{subject}_channel_importance_{viz}_plot.png',
                    bbox_inches='tight', dpi=100 + np.log(self.grid.size) * 200)
        plt.close()
