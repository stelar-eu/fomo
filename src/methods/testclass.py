from dataclasses import dataclass, field
import numpy as np
from anytree import NodeMixin, RenderTree
import pandas as pd

@dataclass
class TestCluster(NodeMixin):
    W: pd.DataFrame
    D: np.ndarray
    update_func: callable

    # def __post_init__(self):
    #     self.Dl = self.D[self.ids]

    def update(self, id, val):
        self.update_func(self.D, id, val)

    # def get_max(self):
    #     view = self.D[self.ids]
    #     return np.max(view)

    def split(self):
        n = self.W.shape[1]
        leftW = self.W.iloc[:, :n//2]
        rightW = self.W.iloc[:, n//2:]

        left = TestCluster(leftW, self.D, update_func=self.update_func)
        right = TestCluster(rightW, self.D, update_func=self.update_func)
        return left, right
    
# Test something
if __name__ == "__main__":

    def update_func(D,idx,val):
        D[idx] = val

    n,m = 5, 10
    w = 3

    np.random.seed(0)

    # Initialize the sliding window
    df = pd.DataFrame(np.random.rand(m, n))
    W = df.iloc[:w, :]

    # # Initialize global distances
    D = np.zeros(n)

    print("Window before update:", W)
    print("Distances before update:", D)

    # # Initialize the cluster
    c1 = TestCluster(W, D, update_func)

    # # Update the cluster
    c1.update(1, 1)

    # # Update the sliding window
    W.drop(W.index[0], inplace=True)
    W.loc[w] = df.iloc[3]

    # # Print the distances
    print("Window after update:", c1.W)
    print("Distances after first update:", D, c1.D)
    # print("Max value in cluster:", c1.get_max())

    # # Split the cluster
    # c2, c3 = c1.split()

    # # Update the new cluster
    # c3.update(2, 2)

    # # Print the distances
    # print("Distances after second update:", D, c1.D, c2.D, c3.D)
    # print("Max value in cluster:", c1.get_max(), c2.get_max(), c3.get_max())


