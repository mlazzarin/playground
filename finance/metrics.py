"""
Author: Marco Lazzarin
Description: Custom metrics and losses for "Deep learning for portfolio optimization"
"""

# Libraries
import numpy as np
import tensorflow as tf


def sharpe(y_true, y_pred):
    """
    (Negative) Sharpe ratio loss function.
    Expects input of shape (batch_size, time steps, n_assets).

    Args:
        y_true: asset returns
        y_pred: allocation weights
    """

    # Compute realized portfolio returns (eq. 3 and below from the original paper)
    # sum over assets
    port_re = tf.reduce_sum(tf.multiply(y_true[:, 1:, :], y_pred[:, :-1, :]), axis=2)

    # Compute loss function (eq. 1 from the original paper)
    mean, variance = tf.nn.moments(port_re, axes=1)
    loss = mean / tf.sqrt(variance)

    return -loss


def portfolio_returns(y_true, y_pred):
    """
    Portfolio returns.
    Expects input of shape (batch_size, time steps, n_assets).

    Args:
        y_true: asset returns
        y_pred: allocation weights
    """

    # Compute realized portfolio returns (eq. 3 and below from the original paper)
    # sum over assets
    port_re = tf.reduce_sum(tf.multiply(y_true[:, 1:, :], y_pred[:, :-1, :]), axis=2)
    port_re = port_re + 1.0
    port_re = tf.reduce_prod(port_re, axis=1) - 1.0 # sum over time steps

    return port_re


def portfolio_returns_annualized(y_true, y_pred):
    """
    Annualized portfolio return.
    Expects input of shape (batch_size, time steps, n_assets).

    Args:
        y_true: asset returns
        y_pred: allocation weights
    """

    port_re = portfolio_returns(y_true, y_pred)
    return (port_re + 1.0) ** (365.0 / (y_true.shape[1]-1)) - 1


def portfolio_evolution(returns, allocations):
    """
    Compute the evolution of the portfolio value.
    Expect input of shape (time steps, n_assets)

    Args:
        returns: asset returns
        allocations: allocation weights
    """

    # sum over assets
    port_ev = tf.reduce_sum(tf.multiply(returns[1:, :], allocations[:-1, :]), axis=1)
    port_ev = port_ev + 1.0
    port_ev = tf.math.cumprod(port_ev, axis=0) - 1.0 # sum over time steps
    port_ev = np.concatenate(([0.0], port_ev), axis=0) # first step is 0

    return port_ev
