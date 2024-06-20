import logging
import os
import time
from typing import Any, List

import numpy as np
import pandas as pd


# Create the different parameter types
class Stat:
    pass


class Hide:
    pass


class Parameters:
    # Required attributes (to be set)
    input_path: str = None
    window: int = None

    # Optional attributes
    output_path: str = os.getcwd()
    budget: int = 100
    metric: str = "euclidean"
    n_streams: int = None
    duration: int = None
    warmup: int = 0
    selection_strategy: str = 'odac'
    prio_strategy: str = 'rmse'
    tau: float = 1
    index: bool = False
    header: bool = False

    # Statistics
    # TODO implement
    # DataFrame storing the used predictions for each of the streams over time.
    forecast_history: Hide = pd.DataFrame(
        {
            'ds': pd.to_datetime([]),
            'stream_id': pd.Series(dtype='int'),
            'stream_name': pd.Series(dtype='str'),
            'ypred': pd.Series(dtype='int'),
            'ytrue': pd.Series(dtype='int'),
            'squared_error': pd.Series(dtype='float'),
            'cluster_id': pd.Series(dtype='int'),
            'model_id': pd.Series(dtype='int')})

    median_rmse: Stat = None
    mean_rmse: Stat = None

    # TODO include different timings as well

    @staticmethod
    def check():
        """
        Check if the parameters are set correctly
        """
        assert Parameters.input_path is not None, "No input path provided"
        assert Parameters.window is not None, "No window size provided"
        assert Parameters.selection_strategy in ['odac',
                                                 'singleton'], f"Invalid selection strategy: {Parameters.selection_strategy}"
        assert Parameters.prio_strategy in ['rmse',
                                            'random'], f"Invalid prioritization strategy: {Parameters.prio_strategy}"

    @staticmethod
    def get_stream_rmses():
        """
        Get the per-stream RMSE of all forecasts
        """
        return np.sqrt(Parameters.forecast_history.groupby(['stream_name']).squared_error.mean())

    @staticmethod
    def print_line():
        """
        Print a line
        """
        logging.info("-" * 50)

    @staticmethod
    def roundfloat(x, decimals=3):
        """
        Round a float to certain decimals
        """
        return round(x, decimals) if type(x) == float else x

    @staticmethod
    def printdict(d):
        """
        Print a dictionary
        """
        for k, v in d.items():
            logging.info(f"{k}: {v}")

    @staticmethod
    def get_attributes(allowed_types: List[Any] = None):
        """
        Get the attributes of the class
        """
        annos = Parameters.__annotations__
        if allowed_types:
            return {k: Parameters.roundfloat(v) for k, v in Parameters.__dict__.items() if
                    k in annos.keys() and
                    annos[k] in allowed_types}
        else:
            return {k: Parameters.roundfloat(v) for k, v in Parameters.__dict__.items() if
                    k in Parameters.__annotations__.keys()}

    @staticmethod
    def print_params():
        """
        Print the parameters
        """
        paramdict = Parameters.get_attributes(allowed_types=[int, float, str, bool])

        Parameters.print_line()
        logging.info(f"Parameters:")
        Parameters.printdict(paramdict)
        Parameters.print_line()

    @staticmethod
    def prepare_stats():
        """
        Prepare the statistics
        """
        # Calculate the median and mean RMSE
        stream_rmses = Parameters.get_stream_rmses()
        Parameters.median_rmse = stream_rmses.median()
        Parameters.mean_rmse = stream_rmses.mean()

    @staticmethod
    def print_stats():
        """
        Print the run statistics
        """
        statdict = Parameters.get_attributes(allowed_types=[Stat])
        Parameters.print_line()
        logging.info(f"Run statistics:")
        Parameters.printdict(statdict)
        Parameters.print_line()

    @staticmethod
    def save():
        #         Create a run identifier by the current timestamp
        run_id = int(time.time())

        #         Create an output directory
        outdir = f"{Parameters.output_path}/{run_id}"
        os.makedirs(outdir, exist_ok=True)
        logging.info(f"Saving the run output to {outdir}")

        #         Save the predictions
        Parameters.forecast_history.to_csv(f"{outdir}/predictions.csv", index=False)

        #        Append the parameters and statistics to runs.csv file
        paramdf = pd.DataFrame(Parameters.get_attributes([str, int, bool, float, Stat]), index=[0])

        param_path = os.path.join(Parameters.output_path, "runs.csv")
        mode = 'a' if os.path.exists(param_path) else 'w'
        paramdf.to_csv(param_path, mode=mode, index=False, header=mode == 'w')
