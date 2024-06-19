import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

from methods.distance_function import *
from methods.odac_cluster import OdacCluster


@dataclass
class FOMO:
    """
    Implementation of the FOMO algorithm.
    """
    # Required attributes
    names: List[str]  # Names of the streams
    w: int  # Sliding window size

    # Optional attributes
    freq: str = 'W'  # Frequency of the data
    metric: str = 'euclidean'  # Distance metric
    tau: float = 1  # Threshold for the cluster tree

    # Inferred attributes
    n: int = 0  # Number of streams
    distfunc: DistanceFunction = None

    # Distance attributes
    D: np.ndarray = None  # Distance matrix
    W: pd.DataFrame = None  # Sliding window

    # Clustering attributes
    root: OdacCluster = None

    # Model attributes
    maintain_forecasts: bool = False  # Flat to update predictions when clusters are split or merged

    # Algorithm strategy attributes
    selection_strategy: str = 'odac'  # The strategy for selecting the different models
    prio_strategy: str = 'rmse'  # The prioritization method for updating the forecasts of different models

    # TODO IMPLEMENT THAT THIS STORES THE SQUARED ERRORS AT EACH TIME STEP INSTEAD OF THE ROLLING RMSE
    performance_history: pd.DataFrame = None  # Prediction performance on each stream over time

    def __post_init__(self):
        self.n = len(self.names)
        self.distfunc = get_distfunc(self.metric)

        # Check certain parameter values
        assert self.selection_strategy in ['odac',
                                           'singleton'], f"Invalid selection strategy: {self.selection_strategy}"
        assert self.prio_strategy in ['rmse', 'random'], f"Invalid prioritization strategy: {self.prio_strategy}"

        # Initialize distance matrix and sliding window
        self.D = np.zeros((self.n, self.n))
        self.W = pd.DataFrame(
            columns=self.names,
            index=pd.to_datetime([]),
        )

        # Initialize the clusters and models
        self.root = OdacCluster(ids=np.arange(self.n), D=self.D, W=self.W, tau=self.tau)
        if self.selection_strategy == 'singleton':
            self.root.split_to_singletons(self.maintain_forecasts)

        #     Initialize the performance history
        self.performance_history = pd.DataFrame(columns=self.names)

    def __str__(self):
        return f"FOMO algorithm with {self.n} streams and a window size of {self.w}"

    def __repr__(self):
        return self.__str__()

    # ------------------------- Model selection -------------------------
    def predict_all(self):
        """
        Fit the models of all leaf clusters and initialize the predictions
        """
        for c in self.root.get_leaves():
            c.model.fit_forecast(periods=c.prediction_window)

    def update_window(self, new_values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slide the window by one timestep on the global sliding window
        """
        # Check if the new values are valid
        assert new_values.shape[0] == self.n, f"Invalid number of values: {new_values.shape[0]}"

        oldarr = np.zeros(self.n)

        # Drop the first row if the window is full
        if self.W.shape[0] == self.w:
            firstrow = self.W.iloc[0]
            oldarr = firstrow.values
            self.W.drop(firstrow.name, inplace=True)

        # Add the new data
        self.W.loc[new_values.name] = new_values

        return oldarr, new_values.values

    def update_clusters(self, old_values: np.ndarray, new_values: np.ndarray):
        """
        Update the FOMO algorithm with a new arrival
        """
        # Dont change the cluster tree if using singleton strategy
        if self.selection_strategy == 'singleton': return

        # Update the distance matrix
        self.update_distances(old_values, new_values)

        # Update the clusters
        self.update_cluster_tree(new_values)

    def update_distances(self, old: np.ndarray, new: np.ndarray):
        """
        Update the distance matrix with a new arrival
        """
        # Update the distance matrix
        self.distfunc.update_distances(self.D, old, new)

    def update_cluster_tree(self, new_values: np.ndarray):
        """
        Update the cluster tree
        """
        for c in self.root.get_leaves():
            # Update the statistics of the cluster
            c.update_stats(new_values)

            # TODO: MAKE SURE TO ONLY TO THE CHECKS IF WE STILL HAVE BUDGET TO CREATE NEW MODELS

            # Check if the cluster needs to split or merge
            action = None
            if c.check_merge():
                action = "merge"
            elif c.check_split():
                action = "split"

            # Perform the action
            if action == "merge":
                c.merge(predict=self.maintain_forecasts)
            elif action == "split":
                c.split(predict=self.maintain_forecasts)

            if action:
                logging.info(f"New tree after {action} of cluster {c.identifier}:")
                self.root.print_tree()

    # ------------------------- Forecast maintenance -------------------------
    def update_forecasts(self, budget: float) -> None:
        """
        Evaluate all models in the cluster tree, and prioritize the updating of forecasts.

        Parameters
        ----------
        budget: float
            The time budget in ms to update the forecasts
        """
        if not self.maintain_forecasts: return

        start = time.time()

        #     Get the clusters to be updated, ordered by past performance starting with the worst
        sorted_clusters = self.prioritize_updates()

        # Update the forecasts until the budget is exceeded
        for c in sorted_clusters:
            if time.time() - start > budget:
                logging.info(f"Budget of {budget}ms exceeded; stopping forecast updates")
                break

            logging.info(f"Updating forecast for cluster {c.identifier} with RMSE {c.model.curr_rmse}")
            c.model.fit_forecast(periods=c.prediction_window)

    def prioritize_updates(self) -> List[OdacCluster]:
        """
        Prioritize the updating of forecasts by evaluating the models
        """

        # TODO Implement different prioritizations here

        # Sort the clusters by their latest RMSE (desc) and filter out models that were never evaluated or have RMSE 0
        filt_clusters = [c for c in self.root.get_leaves() if c.model.curr_rmse > 0]
        sorted_clusters = sorted(filt_clusters, key=lambda c: c.model.curr_rmse, reverse=True)

        return sorted_clusters

    def evaluate_all(self, evaluation_window=5):
        """
        Evaluate all models in the cluster tree and store the RMSEs
        """
        ytrue = self.W.iloc[-evaluation_window:]

        # Get the RMSE of all models
        rmse_series = []
        for c in self.root.get_leaves():
            rmse_sr = c.model.evaluate(ytrue)
            rmse_series.append(rmse_sr)

        # Append the RMSEs to the performance history
        rmse_sr = pd.concat(rmse_series, axis=0).to_frame().T
        self.performance_history = pd.concat([self.performance_history, rmse_sr], axis=0)

    # ------------------------- Misc ---------------------------------------
    def print_tree(self):
        """
        Print the tree
        """
        self.root.print_tree()
