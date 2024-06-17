from dataclasses import dataclass, field
from itertools import count
from collections import OrderedDict
from anytree import NodeMixin, RenderTree
import numpy as np
import logging
from typing import List

@dataclass
class OdacCluster(NodeMixin):
    # Input attributes
    ids: np.ndarray # shape: (n, )
    names: np.ndarray
    update_dist: callable

    # Cluster attributes
    identifier: int = field(default_factory=count().__next__)
    is_active: bool = True
    local_ids: dict = None # shape: (n, )

    # Diameter statistics attributes
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
    confidence_level: float = 0.9
    n_min: int = 5

    # Stream attributes
    n_updates: int = 0

    # Distance attributes
    Dsq: np.ndarray = None # distance matrix    

    def __post_init__(self):
        n = len(self.ids)

        assert n > 0

        self.Dsq = np.zeros((n, n))
        self.local_ids = {idx: i for i, idx in enumerate(self.ids)}

    def __str__(self):
        string = f"Cluster {self.identifier}: "
        if self.is_active:
            string += str(self.names[self.ids])
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
        self.Dsq = None

    def update_diameter_coefficients(self):
        """Initialize the diameter"""
        dshape = self.Dsq.shape

        # Make sure the diagonal does not influence the statistics
        np.fill_diagonal(self.Dsq, np.nan)

        # Get the minimum distance
        d0_idflat = np.nanargmin(self.Dsq)
        self.d0_ids = np.unravel_index(d0_idflat, dshape)
        self.d0 = self.Dsq[self.d0_ids]

        # Get the maximum distance
        d1_idflat = np.nanargmax(self.Dsq)
        self.d1_ids = np.unravel_index(d1_idflat, dshape)
        self.d1 = self.Dsq[self.d1_ids]

        # Get the 2nd maximum distance
        self.Dsq[self.d1_ids] = -np.inf
        d2_idflat = np.nanargmax(self.Dsq)
        self.d2_ids = np.unravel_index(d2_idflat, dshape)
        self.d2 = self.Dsq[self.d2_ids]
        self.Dsq[self.d1_ids] = self.d1

        self.delta = self.d1 - self.d2

        # Get the average distance
        self.davg = np.nanmean(self.Dsq)

    def update_range(self,vals):
        """Update the range statistic"""
        self.max_value = np.max([self.max_value, np.max(vals)])
        self.Rsq = self.max_value * self.max_value

    def update_hoeffding(self):
        """Update the Hoeffding bound on the error of the diameter statistics"""
        self.hoeffding_bound = np.sqrt(self.Rsq * np.log(1 / self.confidence_level) / (2 * self.n_updates))

    def update(self, ids:np.ndarray, vals:np.ndarray):
        """Update the cluster with multiple observations"""
        # Check input
        assert len(ids) == len(vals)
        assert len(ids) > 0

        # Cut to ids and values that are in this cluster
        tmp = np.isin(ids, self.ids)
        ids = ids[tmp]
        vals = vals[tmp]

        if len(ids) == 0:
            return
        
        self.n_updates += 1

        # Update distance matrix
        # Get the local indices of the updated observations through idx
        update_ids = [self.local_ids[idx] for idx in ids]

        # Update distance matrix
        update_vals = np.zeros(len(self.ids))
        update_vals[update_ids] = vals
        self.Dsq = self.update_dist(update_vals, self.Dsq)

        # Update diameter statistics
        self.update_range(vals)
        self.update_hoeffding()
        self.update_diameter_coefficients()

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

        # Get the pivot indices
        x1, y1 = self.d1_ids

        logging.info(f"Splitting cluster {self.identifier} with {self.n_updates} observations and pivot indices {self.ids[x1]} and {self.ids[y1]}")

        # Assign the observations to the new clusters based on the pivot indices
        c1_ids = self.ids[self.Dsq[x1] < self.Dsq[y1]]
        c2_ids = self.ids[self.Dsq[x1] >= self.Dsq[y1]]

        # Add pivots to the clusters to prevent empty clusters
        c1_ids = np.append(c1_ids, self.ids[x1])
        c2_ids = np.append(c2_ids, self.ids[y1])

        # Create new clusters
        c1 = OdacCluster(ids=c1_ids, names=self.names, update_dist=self.update_dist)
        c2 = OdacCluster(ids=c2_ids, names=self.names, update_dist=self.update_dist)

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

        # TODO DEBUG
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







