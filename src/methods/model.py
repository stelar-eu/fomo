import logging
from dataclasses import dataclass, field
from itertools import count
from scipy import stats

import numpy as np
import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").disabled = True  # turn 'cmdstanpy' logs off
logging.getLogger("prophet").disabled = True  # turn 'cmdstanpy' logs off

# Turn off pandas warning
pd.options.mode.chained_assignment = None  # default='warn'


@dataclass
class Model:
    """
    A class representing a Prophet model and providing methods to interact with it.
    """

    W: pd.DataFrame  # Pointer to the DataFrame containing the sliding windows over the data
    names: np.ndarray  # Array containing the names of the streams over which the model is built

    # Optional attributes
    agg_function: str = 'avg'  # Aggregation function to use for the predictions
    freq: str = 'W'  # Frequency of the data

    # Model attributes
    idx: int = field(default_factory=count().__next__)
    agg_forecasts: pd.Series = None  # Aggregate forecasts made by the model, indexed by the timestamps
    curr_kpi: float = None  # Latest KPI of the model

    # ------------- Helper functions -------------
    @staticmethod
    def date_range(periods: int, start: pd.Timestamp = None, end: pd.Timestamp = None, freq: str = 'W'):
        if end is not None:
            if freq == 'W':
                start = end - pd.DateOffset(weeks=periods)
            elif freq == 'D':
                start = end - pd.DateOffset(days=periods)
            elif freq == 'M':
                start = end - pd.DateOffset(months=periods)

        if freq == 'W':
            return pd.DatetimeIndex([start + pd.DateOffset(weeks=i) for i in range(1, periods + 1)])
        elif freq == 'D':
            return pd.DatetimeIndex([start + pd.DateOffset(days=i) for i in range(1, periods + 1)])
        elif freq == 'M':
            return pd.DatetimeIndex([start + pd.DateOffset(months=i) for i in range(1, periods + 1)])
        else:
            raise ValueError("Invalid frequency")

    def make_future_dataframe(self, periods, include_history=False):
        last_date = self.model.history['ds'].max()
        dates = Model.date_range(start=last_date, periods=periods, freq=self.freq)
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

        # Remove outliers (z-score > 3 or < -3)
        z = np.abs(stats.zscore(Wf))
        eps = 3
        Wf[(z > eps)] = np.nan

        if self.agg_function == 'avg':
            # y = np.nanmean(Wf, axis=1)
            y = Wf.mean(axis=1)
        elif self.agg_function == 'sum':
            # y = np.nansum(Wf, axis=1)
            y = Wf.sum(axis=1)
        elif self.agg_function == 'max':
            # y = np.nanmax(Wf, axis=1)
            y = Wf.max(axis=1)
        elif self.agg_function == 'min':
            # y = np.nanmin(Wf, axis=1)
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
        ypred = self.model.predict(future)[['ds', 'yhat']]
        ypred.set_index(ypred.ds, inplace=True)
        ypred = ypred['yhat']

        # Make things non-negative integers 
        # ypred = ypred.round().astype(int)
        ypred[ypred < 0] = 0

        # Store the predictions
        self.agg_forecasts = ypred

        return self.agg_forecasts

    def get_forecast(self, ts: pd.Timestamp):
        """
        Get the forecast for a specific timestamp
        """
        try:
            return self.agg_forecasts.loc[ts]
        except KeyError:
            return 0