from dataclasses import dataclass
from itertools import count
import numpy as np
import logging
from typing import List
from .odac_cluster import OdacCluster
from prophet import Prophet
import pandas as pd

def update_l2(D: np.ndarray, oldvals: np.ndarray, newvals: np.ndarray):
    new_diffs = (newvals - newvals[:, None])**2
    old_diffs = (oldvals - oldvals[:, None])**2
    D += new_diffs - old_diffs

def update_l1(D: np.ndarray, oldvals: np.ndarray, newvals: np.ndarray):
    new_diffs = np.abs(newvals - newvals[:, None])
    old_diffs = np.abs(oldvals - oldvals[:, None])
    D += new_diffs - old_diffs

distfuncs = {
    'euclidean': update_l2,
    'manhattan': update_l1,
}

@dataclass
class FOMO:
    """
    Implementation of the FOMO algorithm.
    """
    # Required attributes
    names: List[str] # Names of the streams
    w: int # Sliding window size

    # Optional attributes
    metric: str = 'euclidean' # Distance metric
    tau: float = 1 # Threshold for the cluster tree

    # Inferred attributes
    n: int = 0 # Number of streams
    update_func: callable = None

    # Distance attributes
    D: np.ndarray = None # Distance matrix
    W: pd.DataFrame = None # Sliding window

    # Clustering attributes
    root: OdacCluster = None

    def __post_init__(self):
        self.n = len(self.names)

        assert self.metric in distfuncs, f"Invalid metric: {self.metric}"
        self.update_func = distfuncs[self.metric]

        # Initialize distance matrix and sliding window
        self.D = np.zeros((self.n, self.n))
        self.W = pd.DataFrame(
            columns=self.names,
            index=pd.to_datetime([]),
            )

        # Initialize root node of the tree (the initial cluster) and set
        self.root = OdacCluster(ids=np.arange(self.n), names=self.names, D=self.D, tau=self.tau)

    def __str__(self):
        return f"FOMO algorithm with {self.n} streams and a window size of {self.w}"
    
    def __repr__(self):
        return self.__str__()
    
    # ------------------------- Model management -------------------------

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
        self.update_func(self.D, old, new)

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