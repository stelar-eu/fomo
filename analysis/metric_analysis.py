import sys
import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr, kendalltau
from ..methods.model import Model

def get_data(path:str, index:bool=False, header:bool=False, n:int=None):
    """
    Get data from csv file
    """
    index_col = 0 if index else None
    header = 0 if header else None

    if n is None:
        return pd.read_csv(path, header=header, index_col=index_col)
    
    return pd.read_csv(path, header=header, index_col=index_col, usecols=range(n))

def train_test_split(df, test_perc=0.1):
    m,n = df.shape
    test_size = np.ceil(test_perc*m).astype(int)

    X_train = df.iloc[:-test_size]
    X_test = df.iloc[-test_size:]

    return X_train, X_test

# Define helper functions
from sklearn.metrics import pairwise_distances
from prophet import Prophet
from sklearn.metrics import root_mean_squared_error
import logging

logging.getLogger("cmdstanpy").disabled = True #  turn 'cmdstanpy' logs off

def evaluate(ytrue, ypred):
    return root_mean_squared_error(ytrue, ypred)

def flat_triu(matrix):
    n = matrix.shape[0]
    return matrix[np.triu_indices(n, k=1)]

def main(input_path, freq='W', n=None, index=True, header=True):
    print(f"Running analysis for {input_path}")

    # Load the data
    df = get_data(input_path, index, header, n)
    df.index = pd.to_datetime(df.index)

    # Rename the index
    df.index.name = "ds"

    # Split the data
    X_train, X_test = train_test_split(df)
    test_size = X_test.shape[0]

    # Set metrics
    metrics = ['chebyshev', 'cosine', 'matching', 'euclidean', 'braycurtis', 'cityblock',  'jaccard', 'russellrao', 'hamming', 'rogerstanimoto', 'minkowski', 'manhattan', 'correlation', 'yule', 'canberra', 'sokalsneath', 'sokalmichener', 'dice']

    metrics = {k: k for k in metrics}

    metrics["spearman"] = lambda x,y: spearmanr(x,y)[0]
    metrics['kendall'] = lambda x,y: kendalltau(x,y)[0]

    # Get the RMSEs of all pairwise models
    print("Calculating pairwise RMSEs")
    n = X_train.shape[1]
    pair_rmse = np.zeros((n,n))
    for i in range(n):
        print(f"Calculating for {i}/{n}", end="\r")
        for j in range(i,n):
            ytrain = (X_train.iloc[:,i] + X_train.iloc[:,j]) / 2
            ytest = (X_test.iloc[:,i] + X_test.iloc[:,j]) / 2
            m = Model()
            ypred = m.fit_forecast(ytrain, test_size, freq=freq, include_history=False)
            rmse = evaluate(ytest, ypred)
            pair_rmse[i,j] = rmse
            pair_rmse[j,i] = rmse

    # Save the results
    np.save(f"A2_model_manager/src/timeseries_clustering/output/pairwise_rmse_{os.path.basename(input_path).replace('.csv', '.npy')}", pair_rmse)

    # Get all relative errors
    rel_errors = pair_rmse / pair_rmse.diagonal()[:, None]

    # Get the correlations between the different distance metrics and the paired RMSEs
    corrs = {}
    rel_errors_flat = flat_triu(rel_errors)
    for k, m in metrics.items():
        D = flat_triu(pairwise_distances(df.T.values, metric=m))
        corr = np.corrcoef(D, rel_errors_flat)[0,1]
        corrs[k] = corr

    # Print sorted correlations
    sorted_corrs = sorted(corrs.items(), key=lambda x: x[1], reverse=True)

    print("Correlations")
    for k,v in sorted_corrs:
        print(f"{k}: {v}")

if __name__ == "__main__":
    DATA_PATH = "/home/jens/ownCloud/Documents/3.Werk/0.TUe_Research/0.STELAR/1.Agroknow/data"

    main(os.path.join(DATA_PATH, "monthly.csv"), "M", index=True,header=True)
    main(os.path.join(DATA_PATH, "weekly.csv"), "W", index=True,header=True)
    main(os.path.join(DATA_PATH, "weekly_syn_r.csv"), "W", index=False, header=False, n=50)
