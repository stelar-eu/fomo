import argparse
import logging
import math
import sys
import time

import numpy as np
import pandas as pd

from methods.fomo import FOMO

# Setup logger
FORMAT = '%(asctime)s - [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

# Set up argument parser
argparse = argparse.ArgumentParser(description="Simulate a stream and continuously maintain a cluster tree")
argparse.add_argument("-i", "--input_path", type=str, help="Path to the csv file containing the stream data",
                      required=True)
argparse.add_argument("-w", "--window", type=int, help="Window size for the stream", required=True)
argparse.add_argument("-b", "--budget", type=int,
                      help="Time budget in ms we have every timestep to manage & maintain forecasts", required=True)
argparse.add_argument("-n", "--n_streams", type=int, help="Number of streams to consider", default=None)
argparse.add_argument("-d", "--duration", type=int,
                      help="Duration of the stream; how many timesteps we want to simulate", default=None)
argparse.add_argument("-m", "--metric", type=str, help="Distance metric to use for clustering", default="euclidean")
argparse.add_argument("-t", "--tau", type=float, help="Threshold for the cluster tree", default=1)
argparse.add_argument("--warmup", type=int, help="Number of time steps after which we will start building models",
                      default=None)
argparse.add_argument("--index", type=bool, help="Flag to indicate if the input file has an index column",
                      default=False)
argparse.add_argument("--header", type=bool, help="Flag to indicate if the input file has a header", default=False)


def get_data(path: str, **kwargs) -> pd.DataFrame:
    """
    Get data from csv file
    """
    n_streams = kwargs.get("n_streams", None)
    duration = kwargs.get("duration", None)
    warmup = kwargs.get("warmup", 0)
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
        if duration:
            df = pd.read_csv(path,
                             header=header,
                             index_col=index_col,
                             usecols=range(n_streams + 1),
                             nrows=duration + warmup + 1)
        else:
            df = pd.read_csv(path,
                             header=header,
                             index_col=index_col,
                             usecols=range(n_streams + 1))
    except Exception as e:
        logging.error(
            f"Error while reading data: {e}, data should be in csv format with columns [Date, Stream1, Stream2, ...]")
        sys.exit(1)

    # Parse the dates if index is given
    if index:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logging.error(
                f"Error while parsing index dates: {e}, make sure the index contains dates and that their are in the correct format")
            sys.exit(1)
    else:
        df.index = pd.date_range(start="2020-01-01", periods=df.shape[0], freq='W')

    return df


def simulate(df: pd.DataFrame, window: int, budget: int, **kwargs) -> None:
    """
    Simulate a stream and continuously maintain a cluster tree
    """
    m = len(df)

    tau = kwargs.get("tau", 1)
    duration = kwargs.get("duration", math.ceil(m * .9))
    warmup = kwargs.get("warmup", math.ceil(m * .1))

    # Initialize the FOMO algorithm
    fomo = FOMO(names=df.columns, w=window, metric=metric, tau=tau)

    T = -1

    # Simulate stream
    while T < duration + warmup:
        T += 1

        # Get all updates
        new_values = df.iloc[T]
        nupdates = np.count_nonzero(new_values)

        logging.info(f"T={T} - Number of updates: {nupdates}")

        start = time.time()

        # Slide the window
        old_values, new_values = fomo.update_window(new_values)

        # Evaluate all models on the new values
        if fomo.maintain_forecasts:
            fomo.evaluate_all(evaluation_window=10)

        # --------------- PHASE 1: Cluster maintenance ---------------

        # Run cluster maintenance
        fomo.update_clusters(old_values, new_values)

        # --------------- PHASE 2: Forecast maintenance ---------------
        # Skip forecast maintenance during warmup period
        if T < warmup:
            continue

        # Create all predictions after warmup period
        if T == warmup:
            logging.info(f"T={T} - WARMUP PERIOD OVER; building models")
            fomo.predict_all()

            #     Set that predictions are redone as a new cluster is activated
            fomo.maintain_forecasts = True
            continue

        # Get remaining budget
        remaining_budget = budget - (time.time() - start) * 1000

        # TODO DIFFER BETWEEN DEBUG AND INFO LOGGING

        # Run forecast maintenance
        if remaining_budget > 0:
            logging.info(f"T={T} - Remaining budget for updating forecasts: {remaining_budget}")
            fomo.update_forecasts(budget=remaining_budget)
        else:
            logging.warning(f"T={T} - No budget left for updating forecasts")

        # TODO implement different strategies for maintaining the models.

    logging.info(f"Final tree:")
    fomo.print_tree()


def main(input_path: str, metric: str, window: int, budget: int, **kwargs):
    """
    Main function; simulate a stream and continuously maintain a cluster tree
    """

    # Load the data
    df = get_data(input_path, **kwargs)
    kwargs["n_streams"] = df.shape[1]

    # Run the stream simulation
    simulate(df=df, window=window, metric=metric, budget=budget, **kwargs)


if __name__ == "__main__":

    logging.info(f"Arguments: {sys.argv}")

    if len(sys.argv) == 1:
        input_path = "/home/jens/ownCloud/Documents/3.Werk/0.TUe_Research/0.STELAR/1.Agroknow/data/weekly.csv"
        metric = "manhattan"
        window = 100
        budget = 1000
        args = {
            "n_streams": 100,
            "duration": 30,
            "warmup": 10,
            "tau": 1,
            "index": True,
            "header": True,
        }
        g = main(input_path, metric, window, budget, **args)
    else:
        args = argparse.parse_args()
        main(args.input_path, args.metric, args.window, args.budget, args._get_kwargs())
