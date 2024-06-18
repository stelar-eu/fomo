from dataclasses import dataclass
from itertools import count
import numpy as np
import logging
from typing import List
from .odac_cluster import OdacCluster
from prophet import Prophet

def update_l2(D: np.ndarray, newvals: np.ndarray, oldvals: np.ndarray):
    new_diffs = (newvals - newvals[:, None])**2
    old_diffs = (oldvals - oldvals[:, None])**2
    D += new_diffs - old_diffs

def update_l1(D: np.ndarray, newvals: np.ndarray, oldvals: np.ndarray):
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
    D: np.ndarray = None
    A_history: np.ndarray = None

    # Clustering attributes
    root: OdacCluster = None

    def __post_init__(self):
        n = len(self.names)

        assert self.metric in distfuncs, f"Invalid metric: {self.metric}"
        self.update_func = distfuncs[self.metric]

        # Initialize distance matrix and arrival history
        self.D = np.zeros((n, n))
        self.A_history = np.zeros((self.w, n))

        # Initialize root node of the tree (the initial cluster) and set
        self.root = OdacCluster(ids=np.arange(n), names=self.names, D=self.D, tau=self.tau)

    def __str__(self):
        return f"FOMO algorithm with {self.n} streams and a window size of {self.w}"
    
    def __repr__(self):
        return self.__str__()
    
    def update(self, A: np.ndarray):
        """
        Update the FOMO algorithm with a new arrival
        """
        # Update the distance matrix
        self.update_distances(A)

        # Update the clusters
        self.update_cluster_tree(A)

    def update_distances(self, A: np.ndarray):
        """
        Update the distance matrix with a new arrival
        """
        # Get the values to retire
        oldvals = self.A_history[0]

        # Update the arrival history
        self.A_history = np.roll(self.A_history, -1, axis=0)
        self.A_history[-1] = A

        # Update the distance matrix
        self.update_func(self.D, A, oldvals)

    def update_cluster_tree(self, A: np.ndarray):
        """
        Update the cluster tree
        """
        for c in self.root.get_leaves():
            if c.is_singleton():
                continue

            # Update the statistics of the cluster
            updated = c.update_stats(A)
            if not updated:
                continue

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