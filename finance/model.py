"""
Author: Marco Lazzarin
Description: Deep learning model from "Deep learning for portfolio optimization"
"""
# pylint: disable=import-error
# pylint: disable=no-name-in-module

# Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, TimeDistributed

from metrics import sharpe, portfolio_returns


class PortfolioOptimizer():
    """Deep learning model for portfolio optimization"""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, data):
        """Initialize data members"""

        # Train-valid-test split
        sequences_len  = data.shape[1]
        self.train_set = data[:, :sequences_len//2, :]
        self.valid_set = data[:, sequences_len//2:3*sequences_len//4, :]
        self.test_set  = data[: ,3*sequences_len//4:, :]

        # Compute returns
        self.train_returns = self.train_set[:, :, 1::2]
        self.valid_returns = self.valid_set[:, :, 1::2]
        self.test_returns  = self.test_set[:, :, 1::2]

        # Various
        self.n_features = data.shape[2]
        self.history    = None
        self.model      = None

    def build(self):
        """Build the deep learning model."""
        self.model = Sequential()
        self.model.add(Input(shape=(None, self.n_features)))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(TimeDistributed(Dense(4, activation="softmax")))
        self.model.compile(loss=sharpe, optimizer="adam", metrics=[portfolio_returns])
        self.model.summary()

    def fit(self, epochs=100):
        """Fit the deep learning model."""
        self.history = self.model.fit(self.train_set, self.train_returns,
                                      validation_data=(self.valid_set, self.valid_returns),
                                      epochs=epochs)
