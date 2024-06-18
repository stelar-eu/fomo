from dataclasses import dataclass
import numpy as np
from prophet import Prophet
import pandas as pd
import pickle
from prophet.serialize import model_from_json
import logging

logging.getLogger("cmdstanpy").disabled = True #  turn 'cmdstanpy' logs off
logging.getLogger("prophet").disabled = True #  turn 'cmdstanpy' logs off

@dataclass
class Model:
    """
    A class representing a Prophet model and providing methods to interact with it.
    """

    W: pd.DataFrame # Pointer to the DataFrame containing the sliding windows over the data
    names: np.ndarray # Array containing the names of the streams over which the model is built

    # Optional attributes
    agg_function: str = 'avg' # Aggregation function to use for the predictions
    freq: str = 'W' # Frequency of the data

    # Model attributes
    predictions: pd.Series = None # Predictions made by the model

    # ------------- Helper functions -------------
    @staticmethod
    def date_range(start, periods, freq='W'):
        if freq == 'W':
            return pd.DatetimeIndex([start + pd.DateOffset(weeks=i) for i in range(1,periods+1)])
        elif freq == 'D':
            return pd.DatetimeIndex([start + pd.DateOffset(days=i) for i in range(1,periods+1)])
        elif freq == 'M':
            return pd.DatetimeIndex([start + pd.DateOffset(months=i) for i in range(1,periods+1)])
        else:
            raise ValueError("Invalid frequency")

    def make_future_dataframe(self, periods, include_history=False):
        last_date = self.model.history['ds'].max()
        dates = Model.date_range(last_date, periods, self.freq)
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:periods]  # Return correct number of periods

        if include_history:
            dates = np.concatenate((np.array(self.model.history_dates), dates))

        return pd.DataFrame({'ds': dates})
    
    # ------------- Forecasting -------------
    def init_model(self):
        """
        Build the model
        """
        self.model = Prophet()

    def prepare_training_data(self):
        """
        Prepare the training data for the model
        """
        # Get and aggregate the data
        Wf = self.W[self.names]

        if self.agg_function == 'avg':
            y = Wf.mean(axis=1)
        elif self.agg_function == 'sum':
            y = Wf.sum(axis=1)
        elif self.agg_function == 'max':
            y = Wf.max(axis=1)
        elif self.agg_function == 'min':
            y = Wf.min(axis=1)
        else:
            raise ValueError("Invalid aggregation function")
        
        # Prepare for prophet
        y = y.reset_index()
        y.columns = ['ds', 'y']

        return y

    def fit_forecast(self, periods):
        """
        Fit the model and forecast
        """
        self.init_model()

        # Prepare the training data
        y = self.prepare_training_data()

        # Fit the model
        self.model.fit(y) 

        # Predict
        future = self.make_future_dataframe(periods=periods, include_history=False)
        ypred = self.model.predict(future)[['ds','yhat']]
        ypred.set_index(ypred.ds, inplace=True)
        ypred = ypred['yhat']

        # Make things non-negative integers 
        ypred = ypred.round().astype(int)
        ypred[ypred < 0] = 0

        # Store the predictions
        self.predictions = ypred

        return self.predictions

