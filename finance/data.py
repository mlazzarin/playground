"""
Author: Marco Lazzarin
Description: Data loading and preprocessing for "Deep learning for portfolio optimization"
"""

# Libraries
import numpy as np
import pandas as pd


def load_data(filename, columns, add_returns=True,
              start_date="03/01/2011", end_date="03/01/2018"):
    """Load data from a CSV file and return a NumPy array

    Args:
        filename: path to the CSV file
        columns: columns to extract (e.g. "Close")
        add_returns: whether to add the Returns feature
        start_date: initial date for the historical series
        end_date: final date for the historical series

    Return:
        NumPy array of shape (timesteps, columns)

    """

    # Load CSV from file
    data = pd.read_csv(filename)

    # Filter element by data
    data["Date"] = pd.to_datetime(data["Date"])
    data = data[data['Date'] > pd.to_datetime(start_date)]
    data = data[data['Date'] < pd.to_datetime(end_date)]

    # Convert data to NumPy array
    np_data = data[columns].to_numpy()

    # Add returns if needed
    if add_returns:
        returns = np_data[1:] / np_data[:-1] - 1
        returns = np.concatenate(([[0.0]], returns))
        np_data = np.stack((np_data, returns), axis=1)

    return np_data
