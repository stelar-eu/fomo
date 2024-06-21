# Forecasting Model Orchestrator (FOMO)

## Overview

The Forecasting Model Orchestrator (FOMO) is designed to efficiently manage and optimize a portfolio of forecasting
models for time series data.
The primary objective is to maximize the overall performance of the models while keeping the total time for re-training
and inference within a predefined budget.

## Problem Definition

FOMO addresses the challenge of updating and managing multiple time series forecasting models under time constraints.
Given a set of time series, the goal is to partition these into disjoint subsets and assign a model to each subset.
The performance of the models should be optimized, and the total time for updating the models should be kept below a
certain threshold.

The problem can be formulated as an optimization problem:

- **Objective:** Minimize the sum of squared errors of the models.
- **Constraints:** Ensure the total time taken to update the models is within a budget.

## Solution Description

### Summary

The algorithm leverages dependencies among time series to improve forecasting accuracy.
It dynamically updates the models based on recent data while adhering to a time budget. The solution involves:

1. Clustering time series based on similarity.
2. Assigning and updating models for each cluster.
3. Optimizing the update schedule to stay within the time budget.

### Components

- **Clustering:** Time series are clustered to exploit dependencies.
- **Model Assignment:** Each cluster is assigned a forecasting model.
- **Update Scheduling:** A binary decision variable determines whether a model is updated at each step.

## Usage Instructions

FOMO can be run using different strategies for model selection and model prioritization. The following model selection
strategies are supported:

- **Singleton:** Each time series is modeled by a separate model. This is the baseline strategy.
- **ODAC:** The Online Divisive-Agglomerative Clustering algorithm is used to cluster the time series and maintain (
  aggregate) models over them to reduce the number of models that need to be updated.

The following model prioritization strategies are supported:

- **Random:** The models are updated in a random order. This is the baseline strategy.
- **RMSE:** The models are updated in order of their RMSE over the last 10 time steps.
- **SMAPE:** The models are updated in order of their SMAPE over the last 10 time steps.

### Requirements

Ensure you have the necessary Python packages installed in requirement.txt. You can install them using the following
command:

```bash
pip install -r requirements.txt
```

### Parameters

The FOMO application accepts several parameters to customize its behavior. Here's an overview of all the parameters:

- `-i`, `--input_path`: Path to the CSV file containing the stream data. This parameter is required.
- `-o`, `--output_path`: Path to the output directory. The default is the current working directory.
- `-b`, `--budget`: Time budget in milliseconds we have every timestep to manage & maintain forecasts. The default is
  20.
- `-w`, `--window`: Window size for the stream. The default is 100.
- `-n`, `--n_streams`: Number of streams to consider. The default is None, which means all streams in the input file
  will be considered.
- `-d`, `--duration`: Duration of the stream; how many timesteps we want to simulate. The default is 100.
- `-m`, `--metric`: Distance metric to use for clustering. The default is "manhattan".
- `--selection`: The strategy for selecting the different models. Choose from [odac, singleton, random]. The default
  is 'odac'.
- `--prio`: The prioritization method for updating the forecasts of different models. Choose from [rmse, smape, random].
  The default is 'smape'.
- `-t`, `--tau`: Threshold for the cluster tree. The default is 1.
- `--warmup`: Number of time steps after which we will start building models. The default is 30.
- `--index`: Flag to indicate if the input file has an index column. The default is False.
- `--header`: Flag to indicate if the input file has a header. The default is False.
- `--loglevel`: Log level for the logger. The default is 'INFO'.
- `--savelogs`: Flag to indicate if the logs should be saved. The default is False.

Please refer to the `main.py` script for a detailed description of each argument.

### Running the Algorithm

You can run the algorithm by executing the script with the required parameters.
If no parameters are provided, default values will be used.

#### Example Command

```bash
python main.py --input_path "path/to/input.csv" --output_path "path/to/output" --metric "manhattan" --window 100 --budget 20 --n_streams 300 --duration 800 --warmup 100 --selection "odac" --prio "smape" --tau 1 --index --header --save_logs False --loglevel "INFO"
```

### Main Function

The main function orchestrates the following steps:

1. Print the parameters.
2. Load the data from the input CSV file.
3. Simulate a stream of data and maintain a cluster tree.
4. Compute aggregate statistics.
5. Print and save the final run statistics and parameters.

### Logging

The script uses Python's `logging` module to provide information about the execution process. L
ogs can be saved to a file if `save_logs` is set to `True`.

### Output

The output of the script includes:

- **Run Statistics:** The final statistics and parameters of the run, including the total time taken, the number of models updated, and
  the total number of models. This information is printed to the console and saved to `runs.csv`.
- **Log File:** A log file containing information about the execution process if `save_logs` is set to `True`. The log file is saved as `output_dir/{timestamp}/log.txt`.
- **Predictions:** The predictions made by the models for each time step for each time series. The predictions are saved as `output_dir/{timestamp}/predictions.csv`.

## Contributing

Contributions are welcome. Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Conclusion

The Forecasting Model Orchestrator (FOMO) provides a robust framework for managing multiple time series forecasting
models efficiently.
By leveraging dependencies among time series and optimizing the update schedule within a time budget, FOMO ensures high
performance and timely updates.