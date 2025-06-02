import logging
import os
import time
from typing import Any, List

from dataclasses import dataclass
import numpy as np
import pandas as pd

import utils.lib as lib
from utils.minio_client import MinioClient

# Create the different parameter types
class Stat:
    pass


class Hide:
    pass


class Parameters:
    # Required attributes (to be set)
    input_path: str = None
    output_path: str = None
    log_path: str = None
    minio_id: str = None
    minio_key: str = None
    minio_token: str = None
    minio_url: str = None

    # Optional attributes
    loglevel: int = logging.INFO
    window: int = 100
    budget: int = 50
    metric: str = "manhattan"
    n_streams: int = 300
    duration: int = 100
    warmup: int = 100
    selection_strategy: str = 'odac'
    prio_strategy: str = 'rmse'
    tau: float = 2
    resolution: str = "M" # Resolution of the data
    index: bool = True
    header: bool = True
    save_logs: bool = True  # If True the logging will be printed to the console, else it will be saved to a file in the output directory

    # Inferred attributes
    local_output_dir: str = None
    minio_client: MinioClient = None

    # Statistics
    # DataFrame storing the used predictions for each of the streams over time.
    forecast_history: Hide = pd.DataFrame(
        {
            'ds': pd.to_datetime([]),
            'stream_id': pd.Series(dtype='int'),
            'stream_name': pd.Series(dtype='str'),
            'ypred': pd.Series(dtype='int'),
            'ytrue': pd.Series(dtype='int'),
            'squared_error': pd.Series(dtype='float'),
            'ape': pd.Series(dtype='float'),
            'cluster_id': pd.Series(dtype='int'),
            'model_id': pd.Series(dtype='int')})

    mean_rmse: Stat = None
    mean_smape: Stat = None

    @staticmethod
    def check():
        """
        Check if the parameters are set correctly
        """
        assert Parameters.input_path is not None, "No input path provided"
        assert Parameters.window is not None, "No window size provided"
        assert Parameters.selection_strategy in ['odac',
                                                 'singleton'], f"Invalid selection strategy: {Parameters.selection_strategy}"
        assert Parameters.prio_strategy in ['rmse', 'smape', 'random'], f"Invalid prioritization strategy: {Parameters.prio_strategy}"

        #         Create an output directory
        output_dir = f"output/{int(time.time())}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving the run output to {output_dir}")
        Parameters.local_output_dir = output_dir

        # Setup logger
        FORMAT = '%(asctime)s.%(msecs)03d - [%(levelname)s] %(message)s'
        logging.basicConfig(format=FORMAT, datefmt='%d/%m/%Y %H:%M:%S')
        logging.getLogger().setLevel(Parameters.loglevel)

        if Parameters.log_path is not None:
            log_path = os.path.join(output_dir, "log.txt")
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter(FORMAT))
            logging.getLogger().addHandler(file_handler)

        # Initialize the MinIO client
        Parameters.minio_client = MinioClient(Parameters.minio_url, Parameters.minio_id, Parameters.minio_key, session_token=Parameters.minio_token)

    @staticmethod
    def get_rmses(df, gb: str = 'stream_name'):
        """
        Get the RMSE of all forecasts
        """
        return np.sqrt(df.groupby(gb).squared_error.median())

    @staticmethod
    def get_smapes(df, gb: str = 'stream_name'):
        """
        Get the sMAPE for all forecasts
        sMAPE = 2 * |ytrue - ypred| / (|ytrue| + |ypred|) * 100
        """
        return df.groupby(gb).ape.mean()

    @staticmethod
    def get_attributes(allowed_types: List[Any] = None):
        """
        Get the attributes of the class
        """
        annos = Parameters.__annotations__
        if allowed_types:
            return {k: lib.roundfloat(v) for k, v in Parameters.__dict__.items() if
                    k in annos.keys() and
                    annos[k] in allowed_types}
        else:
            return {k: lib.roundfloat(v) for k, v in Parameters.__dict__.items() if
                    k in Parameters.__annotations__.keys()}

    @staticmethod
    def print_params():
        """
        Print the parameters
        """
        paramdict = Parameters.get_attributes(allowed_types=[int, float, str, bool])
        paramdict = {k: v for k, v in paramdict.items() if not k.startswith("minio_")}
        lib.print_line()
        logging.info("Parameters:")
        lib.printdict(paramdict)
        lib.print_line()

    @staticmethod
    def prepare_stats():
        """
        Prepare the statistics
        """
        # Calculate the mean RMSE
        stream_rmses = Parameters.get_rmses(df=Parameters.forecast_history)
        Parameters.mean_rmse = stream_rmses.mean()

        # Calculate the mean sMAPE
        stream_smapes = Parameters.get_smapes(df=Parameters.forecast_history)
        Parameters.mean_smape = stream_smapes.mean()

    @staticmethod
    def print_stats():
        """
        Print the run statistics
        """
        statdict = Parameters.get_attributes(allowed_types=[Stat])
        lib.print_line()
        logging.info("Run statistics:")
        lib.printdict(statdict)
        lib.print_line()

    @staticmethod
    def save():
        #         Save the predictions
        Parameters.forecast_history.to_csv(f"{Parameters.local_output_dir}/predictions.csv", index=False)

        # Upload the predictions and logs to MinIO
        Parameters.minio_client.put_object(s3_path=Parameters.output_path, file_path=f"{Parameters.local_output_dir}/predictions.csv")

        if Parameters.save_logs:
            Parameters.minio_client.put_object(s3_path=Parameters.log_path, file_path=f"{Parameters.local_output_dir}/log.txt")       
