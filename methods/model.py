from dataclasses import dataclass
import numpy as np
from prophet import Prophet
import pandas as pd
import pickle
from prophet.serialize import model_from_json

@dataclass
class Model:
    """
    A class representing a Prophet model and providing methods to interact with it.
    """

    W: np.ndarray # Pointer to the array containing the sliding windows over the data
    ids: np.ndarray # Array containing the ids of the streams over which the model is built

    model: Prophet = Prophet() # Prophet model
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

    def make_future_dataframe(self, periods, freq='W', include_history=False):
        last_date = self.model.history['ds'].max()
        dates = Model.date_range(last_date, periods, freq)
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:periods]  # Return correct number of periods

        if include_history:
            dates = np.concatenate((np.array(self.model.history_dates), dates))

        return pd.DataFrame({'ds': dates})
    
    # ------------- Forecasting -------------
    def fit_forecast(self, y: pd.Series, periods: int = 10, freq='W'):
        # Get a very complicated model
        model = Prophet()

        # Prepare for prophet
        y.name = "y"
        y = y.reset_index()

        # Fit the model
        model.fit(y) 

        # Predict
        future = Model.make_future_dataframe(model, periods=periods, freq=freq, include_history=False)
        ypred = model.predict(future)
        ypred = model.predict(future)[['ds','yhat']]
        ypred.set_index(ypred.ds, inplace=True)
        ypred = ypred['yhat']

        # Make things non-negative integers 
        ypred = ypred.round().astype(int)
        ypred[ypred < 0] = 0

        # Store the predictions
        self.predictions = ypred

        return self.predictions

