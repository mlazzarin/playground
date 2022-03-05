"""
Author: Marco Lazzarin
Description: Custom metrics and losses for "Deep learning for portfolio optimization"
"""

# Libraries
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
    port_re = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=2) # sum over assets

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
    port_re = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=2) # sum over assets
    port_re = tf.reduce_sum(port_re, axis=1) # sum over time steps

    return port_re
