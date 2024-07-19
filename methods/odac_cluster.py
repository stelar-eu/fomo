import logging
from dataclasses import dataclass, field
from itertools import count

import numpy as np
import pandas as pd
from anytree import NodeMixin, RenderTree

from methods.model import Model
from parameters import Parameters as p


@dataclass
class OdacCluster(NodeMixin):
    # Input attributes
    ids: np.ndarray  # ids of the columns in this cluster
    D: np.ndarray  # distance matrix
    W: pd.DataFrame  # pointer to the sliding window data

    # Optional attributes
    freq: str = 'W'  # Frequency of the data

    # Inferred attributes
    names: np.ndarray = None  # names of the columns in this cluster
    prediction_window: int = None  # Number of periods to forecast

    # Model attributes
    model: Model = None

    # Cluster attributes
    idx: int = field(default_factory=count().__next__)
    is_active: bool = True
    local_ids: dict = None  # shape: (n, )
    n_updates: int = 0  # Number of times the statistics have been updated
    children: list = None  # List of children

    # Distance statistics attributes
    d0: float = 0
    d0_ids: tuple = None
    d1: float = 0
    d1_ids: tuple = None
    d2: float = 0
    d2_ids: tuple = None
    delta: float = 0
    davg: float = 0

    hoeffding_bound: float = np.inf
    min_value: float = 0
    max_value: float = -np.inf
    Rsq: float = None

    tau: float = 1
    confidence_level: float = 0.95
    n_min: int = 5

    def __post_init__(self):
        assert len(self.ids) > 0
        self.local_ids = {idx: i for i, idx in enumerate(self.ids)}
        self.children = []
        self.prediction_window = p.duration + p.warmup

        # Initialize diameter statistics
        self.update_diameter_coefficients()

        # Initialize the model
        self.names = self.W.columns[self.ids]
        self.model = Model(W=self.W, names=self.names, freq=self.freq)

    def __str__(self):
        string = f"Cluster {self.idx}: "
        if self.is_active:
            string += str(self.names.tolist())
        else:
            string += "[INACTIVE]"
        return string

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.idx == other.idx

    def __hash__(self):
        return hash(self.idx)

    def predict(self):
        """(Re-)Do the predictions of the next periods"""
        self.model.fit_forecast(periods=self.prediction_window)

    def reset(self, predict=True):
        """Reset the cluster"""
        self.__post_init__()
        self.n_updates = 0
        self.is_active = True

        #     Do predictions model if necessary
        if predict: self.predict()

    def print_tree(self) -> str:
        out = []
        for pre, fill, node in RenderTree(self):
            out.append(f"{pre}{node}")
        return "\n".join(out)

    def is_leaf(self):
        """Check if the cluster is a leaf"""
        return len(self.children) == 0

    def is_singleton(self):
        """Check if the cluster is a singleton"""
        return len(self.ids) == 1

    def get_leaves(self):
        """Get the leaves of the tree"""
        if self.is_leaf():  # I am leaf
            return [self]

        # Get the leaves of the children
        return [c for child in self.children for c in child.get_leaves()]

    def deactivate(self):
        """Deactivate the cluster"""
        self.is_active = False

    def local_distances(self):
        """Get the local distances"""
        return self.D[np.ix_(self.ids, self.ids)]

    def update_diameter_coefficients(self):
        """Update the different diameter statistics of the cluster"""
        if self.is_singleton(): return

        Dl = self.local_distances()  # shape: (n, n)
        dshape = Dl.shape

        # Make sure the diagonal does not influence the statistics
        np.fill_diagonal(Dl, np.nan)

        # Get the minimum distance
        d0_idflat = np.nanargmin(Dl)
        self.d0_ids = np.unravel_index(d0_idflat, dshape)
        self.d0 = Dl[self.d0_ids]

        # Get the maximum distance
        self.d1 = np.nanmax(Dl)
        max_ids_x, max_ids_y = np.where(Dl == self.d1)
        self.d1_ids = (max_ids_x[0], max_ids_y[0])

        # Remove the largest distance
        Dl[max_ids_x, max_ids_y] = -np.inf

        # Get the 2nd maximum distance
        d2_idflat = np.nanargmax(Dl)
        self.d2_ids = np.unravel_index(d2_idflat, dshape)
        self.d2 = Dl[self.d2_ids]

        # Reset the maximum distance
        Dl[max_ids_x, max_ids_y] = self.d1

        self.delta = self.d1 - self.d2

        # Get the average distance
        self.davg = np.nanmean(Dl)

    def update_range(self, vals):
        """Update the range statistic"""
        self.max_value = np.max([self.max_value, np.max(vals)])
        self.Rsq = self.max_value * self.max_value

    def update_hoeffding(self):
        """Update the Hoeffding bound on the error of the diameter statistics"""
        self.hoeffding_bound = np.sqrt(self.Rsq * np.log(1 / self.confidence_level) / (2 * self.n_updates))

    def update_stats(self, arrivals: np.ndarray):
        """Update the cluster with multiple observations"""
        # Cut to ids and values that are in this cluster
        local_arrivals = arrivals[self.ids]

        self.n_updates += 1

        # Update diameter statistics
        self.update_range(local_arrivals)
        self.update_hoeffding()
        self.update_diameter_coefficients()

    def check_split(self):
        """Test if the cluster should be split, if so, do it."""

        # Cannot split a singleton
        if self.is_singleton(): return False

        if self.n_updates <= self.n_min:
            return False

        e = self.hoeffding_bound

        if self.delta is None or e is None:
            logging.info(f"Cluster {self.idx} has not been initialized yet")
            return False

        # Check if the Hoeffding bound is violated
        if (self.delta > e) or (self.tau > e):
            if ((self.d1 - self.d0) * abs((self.d1 - self.davg) - (self.davg - self.d0))) > e:
                return True

        return False

    def split(self, predict=True):
        """Split the cluster using d1_idx as pivots"""
        Dl = self.local_distances()

        # Get the pivot indices
        x1, y1 = self.d1_ids

        logging.debug(
            f"Splitting cluster {self.idx} with {self.n_updates} observations and pivot indices {self.ids[x1]} and {self.ids[y1]}")

        # Assign the observations to the new clusters based on the pivot indices
        c1_ids = self.ids[Dl[x1] < Dl[y1]]
        c2_ids = self.ids[Dl[x1] >= Dl[y1]]

        # Edge case: all distances are the same so cluster should be split into singletons
        if len(c1_ids) == 0 or len(c2_ids) == 0:
            self.split_to_singletons(predict)
            return

        # Create new clusters
        c1 = OdacCluster(ids=c1_ids, D=self.D, W=self.W)
        c2 = OdacCluster(ids=c2_ids, D=self.D, W=self.W)

        # Predict if necessary
        if predict:
            c1.model.fit_forecast(periods=self.prediction_window)
            c2.model.fit_forecast(periods=self.prediction_window)

        # Set the new clusters as children
        self.children = [c1, c2]

        # Set the current cluster as parent
        c1.parent = self
        c2.parent = self

        # Initialize hoeffding bounds
        c1.hoeffding_bound = self.hoeffding_bound
        c2.hoeffding_bound = self.hoeffding_bound

        # Set the current cluster as inactive
        self.deactivate()

    def split_to_singletons(self, predict=True):
        """
        Split the cluster into singleton clusters
        """
        for i in self.ids:
            c = OdacCluster(ids=np.array([i]), D=self.D, W=self.W)
            self.children.append(c)  # Add the new cluster as a child
            c.parent = self  # Set the current cluster as parent

            # Initialize hoeffding bounds
            c.hoeffding_bound = self.hoeffding_bound
            if predict:
                c.model.fit_forecast(periods=self.prediction_window)

        # Deactivate this cluster
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

        # Check if enough observations are present
        if self.n_updates <= self.n_min:
            return False

        # Check if the Hoeffding bound is violated
        e = self.hoeffding_bound
        pe = self.parent.hoeffding_bound
        d1 = self.d1
        pd1 = self.parent.d1

        if d1 is None or pd1 is None or e is None or pe is None:
            logging.info(f"Cluster {self.idx} has not been initialized yet")
            return False

        if (d1 - pd1) > max(e, pe):
            logging.debug(
                f"Merging cluster {self.idx} with parent {self.parent.idx}, stats d1: {self.d1}, pd1: {self.parent.d1}, e: {self.hoeffding_bound}, pe: {self.parent.hoeffding_bound}")

            self.parent.merge()
            return True

        return False

    def merge(self, predict=True):
        """
        Merge the children of this cluster
        """
        self.reset(predict)
