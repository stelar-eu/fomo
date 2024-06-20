import argparse
import logging
import math
import sys
import time

import numpy as np
import pandas as pd

from methods.fomo import FOMO
from parameters import Parameters as p

# Setup logger
FORMAT = '%(asctime)s.%(msecs)03d - [%(levelname)s] %(message)s'
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

argparse.add_argument("--selection", type=str,
                      help="The strategy for selecting the different models. Choose from [odac, singleton, random]",
                      default='odac')
argparse.add_argument("--prio", type=str,
                      help="The prioritization method for updating the forecasts of different models. Choose from [rmse, random]",
                      default='rmse')

argparse.add_argument("-t", "--tau", type=float, help="Threshold for the cluster tree", default=1)
argparse.add_argument("--warmup", type=int, help="Number of time steps after which we will start building models",
                      default=None)
argparse.add_argument("--index", type=bool, help="Flag to indicate if the input file has an index column",
                      default=False)
argparse.add_argument("--header", type=bool, help="Flag to indicate if the input file has a header", default=False)


def get_data() -> pd.DataFrame:
    """
    Get data from csv file
    """
    header = 0 if p.header else None
    index_col = 0 if p.index else None

    try:
        # Check if n_streams are valid
        with open(p.input_path) as f:
            firstline = f.readline().split(",")
            n_cols = len(firstline)
            if p.index: n_cols -= 1
            if p.n_streams is not None and p.n_streams > n_cols:
                n_streams = n_cols

        # Read the data
        if p.duration:
            df = pd.read_csv(p.input_path,
                             header=header,
                             index_col=index_col,
                             usecols=range(n_streams + 1),
                             nrows=p.duration + p.warmup + 1)
        else:
            df = pd.read_csv(p.input_path,
                             header=header,
                             index_col=index_col,
                             usecols=range(n_streams + 1))
    except Exception as e:
        logging.error(
            f"Error while reading data: {e}, data should be in csv format with columns [Date, Stream1, Stream2, ...]")
        sys.exit(1)

    # Parse the dates if index is given
    if p.index:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logging.error(
                f"Error while parsing index dates: {e}, make sure the index contains dates and that their are in the correct format")
            sys.exit(1)
    else:
        df.index = pd.date_range(start="2020-01-01", periods=df.shape[0], freq='W')

    return df


def simulate(df: pd.DataFrame) -> None:
    """
    Simulate a stream and continuously maintain a cluster tree
    """
    m = len(df)

    duration = p.duration if p.duration is not None else math.ceil(m * .9)
    warmup = p.warmup if p.warmup is not None else math.ceil(m * .1)

    # Initialize the FOMO algorithm
    fomo = FOMO(names=df.columns, w=p.window, metric=p.metric,
                tau=p.tau,
                selection_strategy=p.selection_strategy,
                prio_strategy=p.prio_strategy
                )

    T = -1

    # Simulate stream
    while T < duration + warmup:
        T += 1

        # Get all updates
        new_values_sr = df.iloc[T]
        nupdates = np.count_nonzero(new_values_sr)

        logging.info(f"T={T} - Number of updates: {nupdates}")

        start = time.time()

        # Slide the window
        old_values, new_values = fomo.update_window(new_values_sr)

        # Evaluate all models on the new values
        if fomo.maintain_forecasts:
            fomo.update_forecast_history(new_values_sr)

            # Compute the rmse per model if using that as a prioritization metric.
            if fomo.prio_strategy == 'rmse':
                fomo.evaluate_all(curr_date=new_values_sr.name, evaluation_window=10)

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
        remaining_budget = p.budget - (time.time() - start) * 1000

        # TODO DIFFER BETWEEN DEBUG AND INFO LOGGING

        # Run forecast maintenance
        if remaining_budget > 0:
            logging.info(f"T={T} - Remaining budget for updating forecasts: {remaining_budget}")
            fomo.update_forecasts(budget=remaining_budget)
        else:
            logging.warning(f"T={T} - No budget left for updating forecasts")

    logging.info(f"Final tree:")
    fomo.print_tree()


def main():
    """
    Main function; simulate a stream and continuously maintain a cluster tree
    """

    # Load the data
    df = get_data()
    p.n_streams = df.shape[1]

    # Run the stream simulation
    simulate(df=df)


if __name__ == "__main__":

    logging.info(f"Arguments: {sys.argv}")

    if len(sys.argv) == 1:
        p.input_path = "/home/jens/ownCloud/Documents/3.Werk/0.TUe_Research/0.STELAR/1.Agroknow/data/weekly.csv"
        p.metric = "manhattan"
        p.window = 100
        p.budget = 100
        p.n_streams = 100
        p.duration = 100
        p.warmup = 30
        p.selection = 'odac'
        p.prioritization = 'rmse'
        p.tau = 1
        p.index = True
        p.header = True
    else:
        args = argparse.parse_args()
        p.input_path = args.input_path
        p.metric = args.metric
        p.window = args.window
        p.budget = args.budget
        p.n_streams = args.n_streams
        p.duration = args.duration
        p.warmup = args.warmup
        p.selection = args.selection
        p.prioritization = args.prio
        p.tau = args.tau
        p.index = args.index
        p.header = args.header

    # Do parameters check
    p.check()

    # Run the main function
    main()
