import argparse
import logging
import math
import os
import sys
import time
import json

import numpy as np
import pandas as pd

from methods.fomo import FOMO
from parameters import Parameters as p, Stat
import utils.minio_client as mc

# # Set up argument parser
# argparse = argparse.ArgumentParser(description="Simulate a stream and continuously maintain a cluster tree")
# argparse.add_argument("-i", "--input_path", type=str, help="Path to the csv file containing the stream data",
#                       required=True)
# argparse.add_argument("-o", "--output_path", type=str, help="Path to the output directory", default=os.getcwd())
# argparse.add_argument("-b", "--budget", type=int,
#                       help="Time budget in ms we have every timestep to manage & maintain forecasts", default=20)
# argparse.add_argument("-w", "--window", type=int, help="Window size for the stream", default=100)
# argparse.add_argument("-n", "--n_streams", type=int, help="Number of streams to consider", default=None)
# argparse.add_argument("-d", "--duration", type=int,
#                       help="Duration of the stream; how many timesteps we want to simulate", default=100)
# argparse.add_argument("-m", "--metric", type=str, help="Distance metric to use for clustering", default="manhattan")

# argparse.add_argument("--selection", type=str,
#                       help="The strategy for selecting the different models. Choose from [odac, singleton, random]",
#                       default='odac')
# argparse.add_argument("--prio", type=str,
#                       help="The prioritization method for updating the forecasts of different models. Choose from [rmse, smape random]",
#                       default='smape')

# argparse.add_argument("-t", "--tau", type=float, help="Threshold for the cluster tree", default=1)
# argparse.add_argument("--warmup", type=int, help="Number of time steps after which we will start building models",
#                       default=30)
# argparse.add_argument("--index", type=bool, help="Flag to indicate if the input file has an index column",
#                       default=False)
# argparse.add_argument("--header", type=bool, help="Flag to indicate if the input file has a header", default=False)
# argparse.add_argument("--loglevel", type=str, help="Log level for the logger", default='INFO')
# argparse.add_argument("--savelogs", type=bool, help="Flag to indicate if the logs should be saved", default=False)


def get_data() -> pd.DataFrame:
    """
    Get data from csv file
    """

    tmp_path = "/tmp/data.csv"

    # First download the data from MinIO to a temporary location
    response = p.minio_client.get_object(p.input_path, tmp_path)

    if 'error' in response:
        logging.error(f"Error while downloading data: {response['message']}")
        sys.exit(1)

    header = 0 if p.header else None
    index_col = 0 if p.index else None

    try:
        # Check if n_streams are valid
        with open(tmp_path) as f:
            firstline = f.readline().split(",")
            n_cols = len(firstline)
            if p.index: n_cols -= 1
            if p.n_streams is None or p.n_streams > n_cols:
                p.n_streams = n_cols

        # Read the data
        if p.duration:
            df = pd.read_csv(tmp_path,
                             header=header,
                             index_col=index_col,
                             usecols=range(p.n_streams),
                             nrows=p.duration + p.warmup + 1)
        else:
            df = pd.read_csv(tmp_path,
                             header=header,
                             index_col=index_col,
                             usecols=range(p.n_streams))
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
                selection_strategy=p.selection_strategy,
                prio_strategy=p.prio_strategy,
                freq=p.resolution,
                )

    T = -1

    # Simulate stream
    while T < duration + warmup:
        T += 1

        # Get all updates
        new_values_sr = df.iloc[T]
        nupdates = np.count_nonzero(new_values_sr)

        logging.info(f"T={T} - Number of updates: {nupdates}")

        # Slide the window
        old_values, new_values = fomo.update_window(new_values_sr)

        # Evaluate all models on the new values
        if fomo.maintain_forecasts:
            fomo.update_forecast_history(new_values_sr)

            # Compute the rmse per model if using that as a prioritization metric.
            if fomo.prio_strategy != 'random':
                fomo.evaluate_all_models(curr_date=new_values_sr.name, evaluation_window=10)

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

        # Run forecast maintenance
        fomo.update_forecasts(budget=p.budget)

    logging.info(f"Final tree:")
    logging.info(fomo.print_tree())


def run():
    """
    Main function; simulate a stream and continuously maintain a cluster tree
    """

    # Print the parameters
    p.print_params()

    # Load the data
    logging.info(f"Loading data...")
    df = get_data()
    p.n_streams = df.shape[1]

    # Run the stream simulation
    simulate(df=df)

    #     Compute the aggregate statistics
    p.prepare_stats()

    #     Print the final run statistics
    p.print_stats()

    #     Save the statistics and parameters
    p.save()

    return {
        "message": "Stream simulation completed successfully!",
        "output": [{
            "path": p.output_path,
            "name": "Directory containing the output files"
        }],
        "metrics": p.get_attributes([str, int, bool, float, Stat]),
        "status": 200
    }


"""
    Example input json:

    {
        "docket_image": "alexdarancio/fomo:latest",
        "input": [{
            "path": "XXXXXXXXX-bucket/2824af95-1467-4b0b-b12a-21eba4c3ac0f.csv",
            "name": "List of time series data"
        }],
        "parameters": {
            "budget": 20,
            "window": 100,
        },
        "tags": ["fomo", "forecasting", "time-series"]
    }
"""

"""
    Example output json:
        {
            "message": "Stream simulation completed successfully!",
            "output": [
            {
                "path": "XXXXXXXXX-bucket/simulation.log",
                "name": "Log file for the simulation"
            },
            {
                "path": "XXXXXXXXX-bucket/predictions.csv",
                "name": "Predictions for the stream"
            }]
            "metrics": {
                "mean_rmse": 0.1,
                "mean_smape": 0.2
            },
            "status": 200
        }
"""

if __name__ == "__main__":

    print(f"Arguments: {sys.argv}")

    if len(sys.argv) == 1: # Test run
        sys.argv = [
            'main.py',
            'resources/input.json',
            'resources/output.json'
        ]
    elif len(sys.argv) != 3:
        raise ValueError("Please provide 2 files.")
    
    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    with open(input_json_path) as o:
        input_json = json.load(o)

    # Parse the input json
    try:
        p.input_path = input_json['input'][0]['path']
        p.output_path = input_json['parameters']['output_path']
        p.minio_id = input_json['minio']['id']
        p.minio_key = input_json['minio']['key']
        p.minio_skey = input_json['minio'].get('secret_key', None)
        p.minio_url = input_json['minio']['endpoint_url']
    except KeyError as e:
        raise ValueError(f"Missing key in input json: {e}")
    
    for key, value in input_json['parameters'].items():
        if key == 'output_path':
            continue
        setattr(p, key, value)

    # Check the parameters
    p.check()

    response = run()
    with open(sys.argv[2], 'w') as o:
        o.write(json.dumps(response, indent=4))