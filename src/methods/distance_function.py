from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

def get_distfunc(name: str):
    if name == "euclidean":
        return EuclideanDistance()
    elif name == "manhattan":
        return ManhattanDistance()
    else:
        raise ValueError(f"Invalid distance function: {name}")

class DistanceFunction(ABC):
    @abstractmethod
    def initialize_distances(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update_distances(self, D: np.ndarray, oldvals: np.ndarray, newvals: np.ndarray) -> None:
        pass

class EuclideanDistance(DistanceFunction):
    def initialize_distances(self, data: np.ndarray):
        w,n = data.shape
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                D[i,j] = np.linalg.norm(data[:,i] - data[:,j])
                D[j,i] = D[i,j]
        return D

    def update_distances(self, D: np.ndarray, oldvals: np.ndarray, newvals: np.ndarray):
        new_diffs = (newvals - newvals[:, None])**2
        old_diffs = (oldvals - oldvals[:, None])**2
        D += new_diffs - old_diffs

class ManhattanDistance(DistanceFunction):
    def initialize_distances(self, data: np.ndarray):
        w,n = data.shape
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                D[i,j] = np.sum(np.abs(data[:,i] - data[:,j]))
                D[j,i] = D[i,j]
        return D

    def update_distances(self, D: np.ndarray, oldvals: np.ndarray, newvals: np.ndarray):
        new_diffs = np.abs(newvals - newvals[:, None])
        old_diffs = np.abs(oldvals - oldvals[:, None])
        D += new_diffs - old_diffs