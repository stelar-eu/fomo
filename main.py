import sys
import logging
import numpy as np
from methods.odac_cluster import OdacCluster
import argparse
import scipy.sparse as sp

# Setup logger
FORMAT = '%(asctime)s - [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

# Set up argument parser
argparse = argparse.ArgumentParser(description="Simulate a stream and continuously maintain a cluster tree")
argparse.add_argument("-i", "--input_path", type=str, help="Path to the csv file containing the stream data", required=True)
argparse.add_argument("-w", "--window", type=int, help="Window size for the stream", required=True)
argparse.add_argument("-n", "--n_streams",  type=int, help="Number of streams to consider", default=None)
argparse.add_argument("-d", "--duration", type=int, help="Duration of the stream; how many timesteps we want to simulate", default=None)
argparse.add_argument("-m", "--metric", type=str, help="Distance metric to use for clustering", default="euclidean")
argparse.add_argument("-t", "--tau", type=float, help="Threshold for the cluster tree", default=1)
argparse.add_argument("--index", type=bool, help="Flag to indicate if the input file has an index column", default=False)
argparse.add_argument("--header", type=bool, help="Flag to indicate if the input file has a header", default=False)

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

def get_data(path:str, **kwargs):
    """
    Get data from csv file
    """
    n_streams = kwargs.get("n_streams", None)
    duration = kwargs.get("duration", None)
    header = kwargs.get("header", False)
    index = kwargs.get("index", False)

    try:
        startrow = 1 if header else 0
        startcol = 1 if index else 0

        # Check if n_streams are valid
        with open(path) as f:
            header = f.readline().split(",")
            n_cols = len(header)
            if n_streams is not None and n_streams > n_cols:
                n_streams = n_cols - 1

            # Get ids from header if possible
            names = np.array(header[startcol:n_streams+startcol])

        if n_streams is None and duration is None:
            data = np.genfromtxt(path, delimiter=",", skip_header=startrow, usecols=range(startcol, None))
        elif n_streams is None:
            data = np.genfromtxt(path, delimiter=",", max_rows=duration, skip_header=startrow, usecols=range(startcol, None))
        elif duration is None:
            data = np.genfromtxt(path, delimiter=",", usecols=range(startcol, n_streams+startcol), skip_header=startrow)
        else:
            data = np.genfromtxt(path, delimiter=",", max_rows=duration, usecols=range(startcol, n_streams+startcol), skip_header=startrow)

        if not header:
            names = np.arange(n_streams)

        return data, names
    except Exception as e:
        logging.error(f"Error while reading data: {e}, data should be in csv format with columns [Date, Stream1, Stream2, ...]")
        sys.exit(1)

def main(input_path:str, metric:str, window:int, **kwargs):
    """
    Main function; simulate a stream and continuously maintain a cluster tree
    """

    n_streams = kwargs.get("n_streams", None)
    duration = kwargs.get("duration", None)
    tau = kwargs.get("tau", 1)

    logging.info(f"Starting main function with n_streams={n_streams} and duration={duration}")

    # Get data
    data, names = get_data(input_path, **kwargs)

    duration, n_streams = data.shape

    # Initialize distance matrix and arrival history
    D = np.zeros((n_streams, n_streams))
    A_history = np.zeros((window, n_streams))

    # Initialize root node of the tree (the initial cluster) and set
    root = OdacCluster(ids=np.arange(n_streams), names=names, D=D, tau=tau)

    T = 0

    # Simulate stream
    while T < duration:
        # Get all updates
        A = data[T]
        nupdates = np.count_nonzero(A)

        #  If no updates, continue
        if nupdates == 0:
            T += 1
            continue

        logging.info(f"T={T} - Number of updates: {nupdates}")

        # Get the values to retire
        oldvals = A_history[0]

        # Update the arrival history
        A_history = np.roll(A_history, -1, axis=0)
        A_history[-1] = A

        # Update the distance matrix
        distfuncs[metric](root.D, A, oldvals)

        # Update the clusters
        for c in root.get_leaves():
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

            if action:
                logging.info(f"T={T} - New tree after {action} of cluster {c.identifier}:")
                root.print_tree()

        # Increment time
        T += 1

    logging.info(f"Final tree:")
    root.print_tree()

if __name__ == "__main__":

    logging.info(f"Arguments: {sys.argv}")

    if len(sys.argv) == 1:
        input_path = "/home/jens/ownCloud/Documents/3.Werk/0.TUe_Research/0.STELAR/1.Agroknow/data/weekly.csv"
        metric="manhattan"
        window=100
        args = {
            "n_streams":  100,
            "duration":  600,
            "tau":  1,
            "index":  True,
            "header":  True,
        }
        g = main(input_path, metric, window, **args)
    else:
        args = argparse.parse_args()
        main(args.input_path, args._get_kwargs())