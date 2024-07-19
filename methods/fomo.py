import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

from methods.distance_function import *
from methods.model import Model
from methods.odac_cluster import OdacCluster
from parameters import Parameters as p


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
    prio_strategy: str = 'smape'  # The prioritization method for updating the forecasts of different models

    performance_history: pd.DataFrame = None  # Prediction performance on each stream over time

    def __post_init__(self):
        self.n = len(self.names)
        self.distfunc = get_distfunc(self.metric)

        # Check certain parameter values
        assert self.selection_strategy in ['odac',
                                           'singleton'], f"Invalid selection strategy: {self.selection_strategy}"
        assert self.prio_strategy in ['rmse', 'smape', 'random'], f"Invalid prioritization strategy: {self.prio_strategy}"

        # Initialize distance matrix and sliding window
        self.D = np.zeros((self.n, self.n))
        self.W = pd.DataFrame(
            columns=self.names,
            index=pd.to_datetime([]),
        )

        # Initialize the clusters and models
        self.root = OdacCluster(ids=np.arange(self.n), D=self.D, W=self.W, tau=self.tau)
        if self.selection_strategy == 'singleton':
            self.root.split_to_singletons(predict=False)

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

            # Check if the cluster needs to split or merge
            action = None
            # if c.check_merge():
            #     action = "merge"
            if c.check_split():
                action = "split"

            # Perform the action
            if action == "merge":
                c.merge(predict=self.maintain_forecasts)
            elif action == "split":
                c.split(predict=self.maintain_forecasts)

            if action:
                logging.debug(f"New tree after {action} of cluster {c.idx}:")
                logging.debug(self.root.print_tree())

    def split_full_tree(self):
        """
        Split the full tree into singletons
        """
        queue = [self.root]
        while queue:
            c = queue.pop(0)
            if c.is_singleton():
                continue
            
            c.split(predict=self.maintain_forecasts)
            queue.extend(c.children)

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
            if (time.time() - start) * 1000 > budget:
                logging.debug(f"Budget of {budget}ms exceeded; stopping forecast updates")
                break

            logging.info(f"Updating forecast for cluster {c.idx} with {self.prio_strategy}={c.model.curr_kpi}")
            c.model.fit_forecast(periods=c.prediction_window)

    def prioritize_updates(self) -> List[OdacCluster]:
        """
        Prioritize the updating of forecasts using the specified strategy
        """

        if self.prio_strategy == 'rmse' or self.prio_strategy == 'smape':
            return self.prioritize_updates_kpi()
        elif self.prio_strategy == 'random':
            return self.prioritize_updates_random()
        else:
            raise ValueError(f"Invalid prioritization strategy: {self.prio_strategy}")

    def prioritize_updates_kpi(self) -> List[OdacCluster]:
        # Sort the clusters by their latest KPI (desc) and filter out models that were never evaluated or have KPI 0
        filt_clusters = [c for c in self.root.get_leaves() if c.model.curr_kpi is not None]
        sorted_clusters = sorted(filt_clusters, key=lambda c: c.model.curr_kpi, reverse=True)

        return sorted_clusters

    def prioritize_updates_random(self) -> List[OdacCluster]:
        """
        Prioritize the updating of forecasts randomly
        """
        return self.root.get_leaves()

    # ------------------------- Forecast evaluation -------------------------

    def update_forecast_history(self, new_values: pd.Series) -> None:
        """
        Get the forecasts for the current date and update the forecast_history in Parameters
        """
        ts = new_values.name

        #     Get the forecasts for each active model
        rows = []
        for c in self.root.get_leaves():
            ypred = c.model.get_forecast(ts)
            if ypred is None:
                logging.warning(f"No forecast available for cluster {c.idx} on timestamp {ts}")
                continue

            # Append the forecast to the forecast history for each stream in the cluster
            for sid in c.ids:
                name = self.names[sid]
                ytrue = new_values[name]
                denum = np.abs(ytrue) + np.abs(ypred)
                ape = 0 if denum == 0 else 2 * np.abs(ytrue - ypred) / denum * 100
                row = {
                    'ds': ts,
                    'stream_id': sid,
                    'stream_name': name,
                    'ypred': ypred,
                    'ytrue': ytrue,
                    'squared_error': (ypred - ytrue) ** 2,
                    'ape': ape,
                    'cluster_id': c.idx,
                    'model_id': c.model.idx
                }
                rows.append(row)

        # Append the forecast to the forecast history
        p.forecast_history = pd.concat([p.forecast_history, pd.DataFrame(rows)], axis=0)

    def evaluate_all_models(self, curr_date: pd.Timestamp, evaluation_window=5):
        """
        Evaluate all active models in the cluster tree by analyzing the forecast history in Parameters
        """
        # Create the evaluation window in timestamps
        eval_ts = Model.date_range(end=curr_date, periods=evaluation_window, freq=self.freq)

        # Get the view of the forecast history
        view = p.forecast_history[p.forecast_history.ds.isin(eval_ts)]

        # Get the kpi relevant to the prioritization strategy for each of the models
        if self.prio_strategy == 'rmse':
            kpis = p.get_rmses(view, gb='model_id')
        elif self.prio_strategy in ['smape']:
            kpis = p.get_smapes(view, gb='model_id')
        else:
            raise ValueError(f"Invalid prioritization strategy: {self.prio_strategy} for evaluating models.")

        #     Get the active models and their kpis
        for c in self.root.get_leaves():
            m = c.model
            if m.idx in kpis.index:
                m.curr_kpi = kpis[m.idx]
            else:
                m.curr_kpi = np.nan

    # ------------------------- Misc ---------------------------------------
    def print_tree(self):
        """
        Print the tree
        """
        return self.root.print_tree()
