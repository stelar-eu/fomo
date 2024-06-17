from dataclasses import dataclass, field
import numpy as np
from anytree import NodeMixin, RenderTree

@dataclass
class TestCluster(NodeMixin):
    ids: np.ndarray
    D: np.ndarray
    update_func: callable

    # def __post_init__(self):
    #     self.Dl = self.D[self.ids]

    def update(self, id, val):
        self.update_func(self.D, id, val)

    def get_max(self):
        view = self.D[self.ids]
        return np.max(view)

    def split(self):
        left_ids = self.ids[:len(self.ids)//2]
        right_ids = self.ids[len(self.ids)//2:]
        left = TestCluster(left_ids, self.D, update_func=self.update_func)
        right = TestCluster(right_ids, self.D, update_func=self.update_func)
        return left, right

# Test something
if __name__ == "__main__":

    def update_func(D,idx,val):
        D[idx] = val

    n = 10

    # Initialize global distances
    D = np.zeros(10)

    print("Distances before update:", D)

    # Initialize the cluster
    c1 = TestCluster(np.arange(n), D, update_func)

    # Update the cluster
    c1.update(1, 1)

    # Print the distances
    print("Distances after first update:", D, c1.D)
    print("Max value in cluster:", c1.get_max())

    # Split the cluster
    c2, c3 = c1.split()

    # Update the new cluster
    c3.update(2, 2)

    # Print the distances
    print("Distances after second update:", D, c1.D, c2.D, c3.D)
    print("Max value in cluster:", c1.get_max(), c2.get_max(), c3.get_max())


