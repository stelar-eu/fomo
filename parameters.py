import logging
import os
import time
from typing import Any, List

import numpy as np
import pandas as pd

import utils.lib as lib


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
    loglevel: str = 'INFO'
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
    save_logs: bool = False  # If True the logging will be printed to the console, else it will be saved to a file in the output directory

    # Inferred attributes
    output_dir: str = None

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
            'ape': pd.Series(dtype='float'),
            'cluster_id': pd.Series(dtype='int'),
            'model_id': pd.Series(dtype='int')})

    mean_rmse: Stat = None
    mean_smape: Stat = None

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
        assert Parameters.prio_strategy in ['rmse', 'smape', 'random'], f"Invalid prioritization strategy: {Parameters.prio_strategy}"

        #         Create an output directory
        output_dir = f"{Parameters.output_path}/{int(time.time())}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving the run output to {output_dir}")
        Parameters.output_dir = output_dir

        # Setup logger
        FORMAT = '%(asctime)s.%(msecs)03d - [%(levelname)s] %(message)s'
        logging.basicConfig(format=FORMAT, datefmt='%d/%m/%Y %H:%M:%S')
        logging.getLogger().setLevel(Parameters.loglevel)

        if Parameters.save_logs:
            log_path = os.path.join(output_dir, "log.txt")
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter(FORMAT))
            logging.getLogger().addHandler(file_handler)

            #         Remove printing to the console
            # logging.getLogger().removeHandler(logging.getLogger().handlers[0])

    @staticmethod
    def get_rmses(df, gb: str = 'stream_name'):
        """
        Get the RMSE of all forecasts
        """
        return np.sqrt(df.groupby(gb).squared_error.mean())

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

        lib.print_line()
        logging.info(f"Parameters:")
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
        logging.info(f"Run statistics:")
        lib.printdict(statdict)
        lib.print_line()

    @staticmethod
    def save():
        #         Save the predictions
        Parameters.forecast_history.to_csv(f"{Parameters.output_dir}/predictions.csv", index=False)

        #        Append the parameters and statistics to runs.csv file
        paramdf = pd.DataFrame(Parameters.get_attributes([str, int, bool, float, Stat]), index=[0])

        param_path = os.path.join(Parameters.output_path, "runs.csv")
        mode = 'a' if os.path.exists(param_path) else 'w'
        paramdf.to_csv(param_path, mode=mode, index=False, header=mode == 'w')
