from dataclasses import dataclass
from itertools import count
import numpy as np
import logging
from typing import List
from methods.odac_cluster import OdacCluster
from methods.distance_function import *
from prophet import Prophet
import pandas as pd


@dataclass
class FOMO:
    """
    Implementation of the FOMO algorithm.
    """
    # Required attributes
    names: List[str] # Names of the streams
    w: int # Sliding window size

    # Optional attributes
    freq: str = 'W' # Frequency of the data
    metric: str = 'euclidean' # Distance metric
    tau: float = 1 # Threshold for the cluster tree
    prediction_window: int = 40 # Number of periods to forecast

    # Inferred attributes
    n: int = 0 # Number of streams
    distfunc: DistanceFunction = None

    # Distance attributes
    D: np.ndarray = None # Distance matrix
    W: pd.DataFrame = None # Sliding window

    # Clustering attributes
    root: OdacCluster = None

    def __post_init__(self):
        self.n = len(self.names)
        self.distfunc = get_distfunc(self.metric)

        # Initialize distance matrix and sliding window
        self.D = np.zeros((self.n, self.n))
        self.W = pd.DataFrame(
            columns=self.names,
            index=pd.to_datetime([]),
            )

        # Initialize root node of the tree (the initial cluster) and set
        self.root = OdacCluster(ids=np.arange(self.n), D=self.D, W=self.W, tau=self.tau)

    def __str__(self):
        return f"FOMO algorithm with {self.n} streams and a window size of {self.w}"
    
    def __repr__(self):
        return self.__str__()
    
    # ------------------------- Model management -------------------------
    def build_all_models(self):
        """
        Fit the models of all leaf clusters and initialize the predictions
        """
        for c in self.root.get_leaves():
            c.model.fit_forecast(periods=self.prediction_window)

    def update_window(self, new_values: pd.Series) -> None:
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
    
    def update(self, new_values: pd.Series):
        """
        Update the FOMO algorithm with a new arrival
        """
        # Update the sliding window
        old_arr, new_arr = self.update_window(new_values)

        # Update the distance matrix
        self.update_distances(old_arr, new_arr)

        # Update the clusters
        self.update_cluster_tree(new_arr)

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
            if c.is_singleton():
                continue

            # Update the statistics of the cluster
            updated = c.update_stats(new_values)
            if not updated:
                continue

            # TODO: MAKE SURE TO ONLY TO THE CHECKS IF WE STILL HAVE BUDGET TO CREATE NEW MODELS

            # Check if the cluster needs to split or merge
            action = None
            if c.check_merge():
                action = "merge"
            elif c.check_split():
                action = "split"

            # Perform the action
            if action == "merge":
                c.merge()
            elif action == "split":
                c.split()

            if action:
                logging.info(f"New tree after {action} of cluster {c.identifier}:")
                self.root.print_tree()
        
    def print_tree(self):
        """
        Print the tree
        """
        self.root.print_tree()