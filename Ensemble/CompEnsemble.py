import numpy as np
import pandas as pd
from data_func import *
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from kneed import KneeLocator
from scipy.signal import savgol_filter
from pathlib import Path


class CompEnsemble:
    def __init__(self, scorelist_path, output_dir, setting_id='default', n_neighbors=6):
        """
        Initialize the Ensemble.

        Args:
            scorelist_path (str): Path to the .npy file containing score lists.
            output_dir (str): Directory to save results.
            setting_id (str): Identifier for the current setting.
            n_neighbors (int): Number of neighbors for KNN-related calculations.
        """
        self.scorelist_path = scorelist_path
        self.output_dir = Path(output_dir)
        self.setting_id = setting_id
        self.n_neighbors = n_neighbors

        # Create the output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load scores and initialize other variables
        self.scorelists = np.load(self.scorelist_path, allow_pickle=True)
        self.truelabels = np.array([0] * 9500 + [1] * 500)  # Adjust as needed
        self.metrics_df = pd.DataFrame()
        self.tau_df = pd.DataFrame()

    def calculate_initial_metrics(self):
        """
        Calculate initial metrics (Mean and Max) for the score lists.
        """
        inverse_rank_lists = np.stack([1 / stats.rankdata(-score_list) for score_list in self.scorelists], axis=-1)

        # Mean
        avg_inverse_ranks = np.mean(inverse_rank_lists, axis=-1)
        mean_roc_auc = roc_auc_score(self.truelabels, avg_inverse_ranks)
        mean_aupr = average_precision_score(self.truelabels, avg_inverse_ranks)
        print(f"MEAN: ROC AUC: {mean_roc_auc}, AUPR: {mean_aupr}")

        # Max
        max_inverse_ranks = np.max(inverse_rank_lists, axis=-1)
        max_roc_auc = roc_auc_score(self.truelabels, max_inverse_ranks)
        max_aupr = average_precision_score(self.truelabels, max_inverse_ranks)
        print(f"MAX: ROC AUC: {max_roc_auc}, AUPR: {max_aupr}")

    def eliminate_lists(self):
        """
        Perform elimination of lists based on a similarity measure and compute metrics.
        """
        excluded_keys = []
        original_score_lists = self.scorelists.copy()
        list_identifiers = {i: self.scorelists[i] for i in range(len(self.scorelists))}

        for iteration in tqdm(range(len(original_score_lists) + 1)):
            # Exclude keys
            indices_to_exclude = [int(key) for key in excluded_keys]
            current_lists = {idx: lst for idx, lst in list_identifiers.items() if idx not in indices_to_exclude}

            if len(current_lists) == 1:
                break

            # Compute MAX(S)
            all_lists = list(current_lists.values())
            max_S_rank = np.max(np.stack([1 / stats.rankdata(-lst) for lst in all_lists], axis=-1), axis=-1)

            # Decide which list to remove
            weighted_kendall_taus = {}
            for idx, lst in current_lists.items():
                temp_lists = [l for k, l in current_lists.items() if k != idx]
                max_S_rank_D = np.max(
                    np.stack([1 / stats.rankdata(-temp_lst) for temp_lst in temp_lists], axis=-1), axis=-1
                )
                weighted_kendall_tau = jaccard_similarity(max_S_rank, max_S_rank_D)
                weighted_kendall_taus[idx] = weighted_kendall_tau

            # Store tau values
            current_taus = pd.Series(weighted_kendall_taus, name=iteration)
            self.tau_df = pd.concat([self.tau_df, current_taus.to_frame().T], ignore_index=True)

            # Identify and exclude the list with the lowest similarity
            min_index = min(weighted_kendall_taus, key=weighted_kendall_taus.get)
            excluded_keys.append(min_index)

            # Compute metrics after removal
            subset = [lst for idx, lst in enumerate(original_score_lists) if idx not in excluded_keys]
            self.compute_metrics(subset, max_S_rank, iteration, excluded_keys)

    def compute_metrics(self, subset, max_S_rank, iteration, excluded_keys):
        """
        Compute metrics after each elimination step.

        Args:
            subset (list): Current subset of score lists.
            max_S_rank (array): MAX(S) rank for the current subset.
            iteration (int): Current iteration number.
            excluded_keys (list): Indices of excluded lists.
        """
        max_sd_rank = np.max(np.stack([1 / stats.rankdata(-lst) for lst in subset], axis=-1), axis=-1)
        roc_auc = roc_auc_score(self.truelabels, max_sd_rank)
        aupr = average_precision_score(self.truelabels, max_sd_rank)
        topkprec = topKprec(max_sd_rank, self.truelabels, 500)
        jac_max = jaccard_similarity(max_S_rank, max_sd_rank)

        # Record metrics
        self.metrics_df.loc[iteration, 'Removed'] = str(excluded_keys)
        self.metrics_df.loc[iteration, 'ROC_AUC'] = roc_auc
        self.metrics_df.loc[iteration, 'AUPR'] = aupr
        self.metrics_df.loc[iteration, 'TopK_Precision'] = topkprec
        self.metrics_df.loc[iteration, 'JAC(MAX(S),MAX(S\d))'] = jac_max

    def save_results(self):
        """
        Save the results to CSV files and plot the Jaccard similarity curve.
        """
        # Save metrics and tau values
        metrics_path = self.output_dir / f"AOMELIM_metrics_{self.setting_id}.csv"
        tau_path = self.output_dir / f"AOMELIM_tau_{self.setting_id}.csv"
        self.metrics_df.to_csv(metrics_path, index=False)
        self.tau_df.to_csv(tau_path, index=False)

        # Plot the Jaccard similarity curve
        xdata = list(range(1, len(self.metrics_df) + 1))
        ydata = self.metrics_df['JAC(MAX(S),MAX(S\d))']
        y_smooth = savgol_filter(ydata, window_length=3, polyorder=1)
        kneedle = KneeLocator(xdata, y_smooth, S=1, curve="concave", direction="decreasing")
        stoppoint = round(kneedle.knee, 3)

        print(f"Stopping Point: {stoppoint}")
        print(self.metrics_df.iloc[stoppoint - 1])

    def process(self):
        """
        Main process to calculate metrics, eliminate lists, and save results.
        """
        self.calculate_initial_metrics()
        self.eliminate_lists()
        self.save_results()


# Example Usage
if __name__ == "__main__":
    scorelist_path = "/path/to/score_lists.npy"  # Path to the .npy file
    output_dir = "/path/to/output"  # Directory to save results
    setting_id = "example_setting"

    processor = CompEnsemble(scorelist_path, output_dir, setting_id=setting_id)
    processor.process()
