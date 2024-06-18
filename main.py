import sys
import logging
import numpy as np
from methods.fomo import FOMO
import argparse
from typing import List
import numpy as np
import pandas as pd
import math

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
argparse.add_argument("--warmup", type=int, help="Number of time steps after which we will start building models", default=None)
argparse.add_argument("--index", type=bool, help="Flag to indicate if the input file has an index column", default=False)
argparse.add_argument("--header", type=bool, help="Flag to indicate if the input file has a header", default=False)

def get_data(path:str, **kwargs) -> pd.DataFrame:
    """
    Get data from csv file
    """
    n_streams = kwargs.get("n_streams", None)
    duration = kwargs.get("duration", None)
    header = kwargs.get("header", None)
    index = kwargs.get("index", None)

    header = 0 if header else None
    index_col = 0 if index else None

    try:    
        # Check if n_streams are valid
        with open(path) as f:
            firstline = f.readline().split(",")
            n_cols = len(firstline)
            if index: n_cols -= 1
            if n_streams is not None and n_streams > n_cols:
                n_streams = n_cols

        # Read the data
        df = pd.read_csv(path, 
                        header=header, 
                        index_col=index_col,
                        usecols=range(n_streams+1),
                        nrows=duration)
    except Exception as e:
        logging.error(f"Error while reading data: {e}, data should be in csv format with columns [Date, Stream1, Stream2, ...]")
        sys.exit(1)

    # Parse the dates if index is given
    if index:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logging.error(f"Error while parsing index dates: {e}, make sure the index contains dates and that their are in the correct format")
            sys.exit(1)
    else:
        df.index = pd.date_range(start="2020-01-01", periods=df.shape[0], freq='W')

    return df

def simulate(df: pd.DataFrame, duration: int, window: int, **kwargs) -> None:
    """
    Simulate a stream and continuously maintain a cluster tree
    """
    tau = kwargs.get("tau", 1)
    warmup = kwargs.get("warmup", math.ceil(duration*.1))

    # Initialize the FOMO algorithm
    fomo = FOMO(names=df.columns, w=window, metric=metric, tau=tau)

    T = 0

    # Simulate stream
    while T < duration+warmup:
        # Get all updates
        new_values = df.iloc[T]
        nupdates = np.count_nonzero(new_values)

        #  If no updates, continue
        if nupdates == 0:
            T += 1
            continue

        # Build the models after warmup period
        if T == warmup:
            logging.info(f"T={T} - Warmup period over; building models")
            fomo.build_all_models()

        logging.info(f"T={T} - Number of updates: {nupdates}")

        # Push the update to the FOMO algorithm
        fomo.update(new_values)

        # Increment time
        T += 1

    logging.info(f"Final tree:")
    fomo.print_tree()

def main(input_path:str, metric:str, window:int, **kwargs):
    """
    Main function; simulate a stream and continuously maintain a cluster tree
    """

    # Load the data
    df = get_data(input_path, **kwargs)
    duration, n_streams = df.shape
    kwargs["n_streams"] = n_streams
    kwargs["duration"] = duration

    # Run the stream simulation
    simulate(df=df, window=window, metric=metric, **kwargs)

    
if __name__ == "__main__":

    logging.info(f"Arguments: {sys.argv}")

    if len(sys.argv) == 1:
        input_path = "/home/jens/ownCloud/Documents/3.Werk/0.TUe_Research/0.STELAR/1.Agroknow/data/weekly.csv"
        metric = "manhattan"
        window = 100
        args = {
            "n_streams":  100,
            "duration":  600,
            "warmup": 300,
            "tau":  1,
            "index":  True,
            "header":  True,
        }
        g = main(input_path, metric, window, **args)
    else:
        args = argparse.parse_args()
        main(args.input_path, args._get_kwargs())