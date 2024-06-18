from dataclasses import dataclass, field
from itertools import count
from collections import OrderedDict
from anytree import NodeMixin, RenderTree
import numpy as np
import logging
import pandas as pd
from typing import List
from .model import Model

@dataclass
class OdacCluster(NodeMixin):
    # Input attributes
    ids: np.ndarray # shape: (n, )
    D: np.ndarray # distance matrix    
    W: pd.DataFrame # pointer to the sliding window data

    # Inferred attributes
    names: np.ndarray = None # shape: (n, )

    # Model attributes
    model: Model = None

    # Cluster attributes
    identifier: int = field(default_factory=count().__next__)
    is_active: bool = True
    local_ids: dict = None # shape: (n, )

    # Distance statistics attributes
    d0: float = None
    d0_ids: tuple = None
    d1: float = None
    d1_ids: tuple = None
    d2: float = None
    d2_ids: tuple = None
    delta: float = None
    davg: float = None

    hoeffding_bound: float = None
    min_value: float = 0
    max_value: float = -np.inf
    Rsq: float = None

    tau: float = 1
    confidence_level: float = 0.95
    n_min: int = 5

    # Stream attributes
    n_updates: int = 0

    def __post_init__(self):
        assert len(self.ids) > 0
        self.local_ids = {idx: i for i, idx in enumerate(self.ids)}

        # Initialize the model
        self.names = self.W.columns[self.ids]
        self.model = Model(W=self.W, names=self.names)

    def __str__(self):
        string = f"Cluster {self.identifier}: "
        if self.is_active:
            string += str(self.names)
        else:
            string += "[INACTIVE]"
        return string
    
    def __eq__(self, other):
        return self.identifier == other.identifier
 
    def reset(self):
        """Reset the cluster"""
        self.__post_init__()
        self.n_updates = 0
        self.is_active = True
        self.children = []
    
    def print_tree(self):
        for pre, fill, node in RenderTree(self):
            logging.info(f"{pre}{node}")
    
    def is_leaf(self):
        """Check if the cluster is a leaf"""
        return not self.children
    
    def is_singleton(self):
        """Check if the cluster is a singleton"""
        return len(self.ids) == 1

    def get_leaves(self):
        """Get the leaves of the tree"""
        if self.is_leaf(): # I am leaf
            return [self]

        # Get the leaves of the children
        return self.children[0].get_leaves() + self.children[1].get_leaves()
    
    def deactivate(self):
        """Deactivate the cluster"""
        self.is_active = False

    def local_distances(self):
        """Get the local distances"""
        return self.D[np.ix_(self.ids, self.ids)]

    """Get the different diameter statistics of the cluster"""
    def update_diameter_coefficients(self):
        Dl = self.local_distances() # shape: (n, n)
        dshape = Dl.shape

        # Make sure the diagonal does not influence the statistics
        np.fill_diagonal(Dl, np.nan)

        # Get the minimum distance
        d0_idflat = np.nanargmin(Dl)
        self.d0_ids = np.unravel_index(d0_idflat, dshape)
        self.d0 = Dl[self.d0_ids]

        # Get the maximum distance
        d1_idflat = np.nanargmax(Dl)
        self.d1_ids = np.unravel_index(d1_idflat, dshape)
        self.d1 = Dl[self.d1_ids]

        # Get the 2nd maximum distance
        Dl[self.d1_ids] = -np.inf
        d2_idflat = np.nanargmax(Dl)
        self.d2_ids = np.unravel_index(d2_idflat, dshape)
        self.d2 = Dl[self.d2_ids]
        Dl[self.d1_ids] = self.d1

        self.delta = self.d1 - self.d2

        # Get the average distance
        self.davg = np.nanmean(Dl)

    def update_range(self,vals):
        """Update the range statistic"""
        self.max_value = np.max([self.max_value, np.max(vals)])
        self.Rsq = self.max_value * self.max_value

    def update_hoeffding(self):
        """Update the Hoeffding bound on the error of the diameter statistics"""
        self.hoeffding_bound = np.sqrt(self.Rsq * np.log(1 / self.confidence_level) / (2 * self.n_updates))

    def update_stats(self, arrivals:np.ndarray):
        """Update the cluster with multiple observations"""
        # Cut to ids and values that are in this cluster
        local_arrivals = arrivals[self.ids]

        if np.count_nonzero(local_arrivals) == 0:
            return False
        
        self.n_updates += 1

        # Update diameter statistics
        self.update_range(local_arrivals)
        self.update_hoeffding()
        self.update_diameter_coefficients()

        return True

    def check_split(self):
        """Test if the cluster should be split, if so, do it."""

        if self.n_updates <= self.n_min:
            return False
        
        e = self.hoeffding_bound

        # DEBUG
        if self.delta is None or e is None:
            logging.info(f"Cluster {self.identifier} has not been initialized yet")
            return False
        
        # Check if the Hoeffding bound is violated
        if ( self.delta > e ) or ( self.tau > e ):
            if ( (self.d1 - self.d0) * abs((self.d1 - self.davg) - (self.davg - self.d0)) ) > e:
                self.split()
                return True

        return False
    
    def split(self):
        """Split the cluster using d1_idx as pivots"""
        Dl = self.local_distances()

        # Get the pivot indices
        x1, y1 = self.d1_ids

        logging.info(f"Splitting cluster {self.identifier} with {self.n_updates} observations and pivot indices {self.ids[x1]} and {self.ids[y1]}")

        # Assign the observations to the new clusters based on the pivot indices
        c1_ids = self.ids[Dl[x1] < Dl[y1]]
        c2_ids = self.ids[Dl[x1] >= Dl[y1]]

        # Create new clusters
        c1 = OdacCluster(ids=c1_ids, D=self.D, W=self.W)
        c2 = OdacCluster(ids=c2_ids, D=self.D, W=self.W)

        # Set the new clusters as children
        self.children = [c1, c2]

        # Set the current cluster as parent
        c1.parent = self
        c2.parent = self

        # Set the current cluster as inactive
        self.deactivate()

    def check_merge(self):
        """
        Check if the cluster should be merged with its parent, if so, do it.
        """

        # Check if the cluster is a root
        if self.is_root:
            return False

        # Check if the cluster is active
        if not self.is_active:
            return False
        
        assert not self.parent.is_active
        
        # Check if enough observations are present
        if self.n_updates <= self.n_min:
            return False

        # Check if the Hoeffding bound is violated
        e = self.hoeffding_bound
        pe = self.parent.hoeffding_bound
        d1 = self.d1
        pd1 = self.parent.d1

        if d1 is None or pd1 is None or e is None or pe is None:
            logging.info(f"Cluster {self.identifier} has not been initialized yet")
            return False

        if (d1 - pd1) > max(e,pe):
            logging.info(f"Merging cluster {self.identifier} with parent {self.parent.identifier}, stats d1: {self.d1}, pd1: {self.parent.d1}, e: {self.hoeffding_bound}, pe: {self.parent.hoeffding_bound}")

            self.parent.merge()
            return True

        return False
    
    def merge(self):
        """
        Merge the children of this cluster
        """
        self.reset()







