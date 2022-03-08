"""
Author: Marco Lazzarin
Description: Unit testing for "Deep learning for portfolio optimization"
"""

# Libraries
import numpy as np
import pytest

from data import load_data
from metrics import portfolio_evolution, portfolio_returns, portfolio_returns_annualized


@pytest.mark.parametrize("target_asset", (0, 1, 2, 3))
def test_portfolio_returns(target_asset):
    """Test metrics::portfolio_returns"""

    # Retrive the historical series and concatenate them in a single
    # NumPy array of shape (timesteps, 2 * n_assets)
    data = np.concatenate((load_data("data/VTI.csv", ["Close"]),
                           load_data("data/AGG.csv", ["Close"]),
                           load_data("data/DBC.csv", ["Close"]),
                           load_data("data/VIX.csv", ["Close"])), axis=1)
    data = np.squeeze(data)
    data = data.reshape(1, *data.shape) # reshape the data to expose the batch size (one)

    # Allocate everything into the target asset
    allocations = np.zeros(data[:, :, 1::2].shape)
    allocations[:, :, target_asset] = 1.0

    # Compare the portfolio return with historical data
    port_re = portfolio_returns(data[:, :, 1::2], allocations)
    ground_truth = data[0, -1, 2*target_asset] / data[0, 0, 2*target_asset] - 1.0
    np.testing.assert_allclose(port_re, ground_truth, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("target_asset", (0, 1, 2 ,3))
def test_portfolio_returns_annualized(target_asset):
    """Test metrics::portfolio_returns_annualized"""

    # Retrive the historical series and concatenate them in a single
    # NumPy array of shape (timesteps, 2 * n_assets)
    data = np.concatenate((load_data("data/VTI.csv", ["Close"]),
                           load_data("data/AGG.csv", ["Close"]),
                           load_data("data/DBC.csv", ["Close"]),
                           load_data("data/VIX.csv", ["Close"])), axis=1)
    data = np.squeeze(data)
    data = data.reshape(1, *data.shape) # reshape the data to expose the batch size (one)

    # Allocate everything into the target asset
    allocations = np.zeros(data[:, :, 1::2].shape)
    allocations[:, :, target_asset] = 1.0

    # Compare the portfolio return with historical data
    port_re = portfolio_returns_annualized(data[:, :, 1::2], allocations)
    ground_truth = (data[0, -1, 2*target_asset] / data[0, 0, 2*target_asset]) ** (365.0 / (data.shape[1]-1)) - 1.0
    np.testing.assert_allclose(port_re, ground_truth, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("target_asset", (0, 1, 2, 3))
def test_portfolio_evolution(target_asset):
    """Test metrics::portfolio_evolution"""

    # Retrive the historical series and concatenate them in a single
    # NumPy array of shape (timesteps, 2 * n_assets)
    data = np.concatenate((load_data("data/VTI.csv", ["Close"]),
                           load_data("data/AGG.csv", ["Close"]),
                           load_data("data/DBC.csv", ["Close"]),
                           load_data("data/VIX.csv", ["Close"])), axis=1)
    data = np.squeeze(data)

    # Allocate everything into the target asset
    allocations = np.zeros(data[:, 1::2].shape)
    allocations[:, target_asset] = 1.0

    # Compare the portfolio evolution with historical data
    port_ev = portfolio_evolution(data[:, 1::2], allocations)
    ground_truth = data[:, 2*target_asset] / data[0, 2*target_asset] - 1.0
    np.testing.assert_allclose(port_ev, ground_truth[1:], atol=1e-10, rtol=1e-10)
    