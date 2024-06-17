import sys
import logging
import numpy as np
from methods.odac_cluster import OdacCluster
import argparse

# Setup logger
FORMAT = '%(asctime)s - [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

# Set up argument parser
argparse = argparse.ArgumentParser(description="Simulate a stream and continuously maintain a cluster tree")
argparse.add_argument("-i", "--input_path", type=str, help="Path to the csv file containing the stream data", required=True) 
argparse.add_argument("-n", "--n_streams",  type=int, help="Number of streams to consider", default=None)
argparse.add_argument("-d", "--duration", type=int, help="Duration of the stream; how many timesteps we want to simulate", default=None)
argparse.add_argument("-m", "--metric", type=str, help="Distance metric to use for clustering", default="euclidean")
argparse.add_argument("-t", "--tau", type=float, help="Threshold for the cluster tree", default=1)
argparse.add_argument("--index", type=bool, help="Flag to indicate if the input file has an index column", default=False)
argparse.add_argument("--header", type=bool, help="Flag to indicate if the input file has a header", default=False)

def update_l2(newvals: np.ndarray, D: np.ndarray):
    new_diffs = newvals - newvals[:, None]
    D += new_diffs**2

def update_l1(newvals: np.ndarray, D: np.ndarray):
    new_diffs = newvals - newvals[:, None]
    D += np.abs(new_diffs)

def update_cheb(newvals: np.ndarray, D: np.ndarray):
    new_diffs = newvals - newvals[:, None]
    D = np.maximum(D, np.abs(new_diffs))

distfuncs = {
    'euclidean': update_l2,
    'manhattan': update_l1,
    'chebyshev': update_cheb
}

def get_data(path:str, n_streams:int=None, duration:int=None, index:bool=False, header:bool=False):
    """
    Get data from csv file
    """
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

def main(input_path:str, n_streams:int, duration:int, tau:float,
         metric:str,
         index:bool, header:bool):
    """
    Main function; simulate a stream and continuously maintain a cluster tree
    """

    logging.info(f"Starting main function with n_streams={n_streams} and duration={duration}")

    # Get data
    data, names = get_data(input_path, n_streams, duration, index, header)

    duration, n_streams = data.shape

    D = np.zeros((n_streams, n_streams))

    # Initialize root node of the tree (the initial cluster) and set
    root = OdacCluster(ids=np.arange(n_streams), names=names, tau=tau, D=D)

    T = 0

    # Simulate stream
    while T < duration:
        # Get all positive updates
        arrivals = data[T]
        update_ids = np.nonzero(arrivals)[0]

        if len(update_ids) == 0:
            T += 1
            continue

        logging.info(f"T={T} - Number of updates: {len(update_ids)}")

        # Update the distance matrix
        distfuncs[metric](arrivals, root.D)

        # Update the clusters
        for c in root.get_leaves():
            if c.is_singleton():
                continue

            # Update the statistics of the cluster
            updated = c.update_stats(arrivals)
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
        n_streams = 100
        duration = 500
        tau = 1
        metric = 'manhattan'
        index = True
        header = True
        g = main(
            input_path=input_path, 
            n_streams=n_streams, 
            duration=duration, 
            tau=tau, 
            metric=metric, 
            index=index, 
            header=header
            )
    else:
        args = argparse.parse_args()
        main(
            input_path = args.input_path, 
            n_streams = args.n_streams, 
            duration = args.duration, 
            tau = args.tau, 
            metric = args.metric,
            index = args.index, 
            header = args.header
            )