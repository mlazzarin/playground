"""
Author: Marco Lazzarin
Description: Deep learning model from "Deep learning for portfolio optimization"
"""
# pylint: disable=import-error
# pylint: disable=no-name-in-module

# Libraries
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
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

    def fit(self, epochs=1000):
        """Fit the deep learning model."""

        early_stopping = EarlyStopping(monitor="val_loss",
                                       patience=30,
                                       restore_best_weights=True)
        self.history = self.model.fit(self.train_set, self.train_returns,
                                      validation_data=(self.valid_set, self.valid_returns),
                                      epochs=epochs, callbacks=[early_stopping])

    def plot(self):
        """Various plot to analise the results."""

        # Plot training history
        fig, axes = plt.subplots(2, 1, figsize=(12, 5))
        axes[0].grid()
        axes[0].plot(self.history.history["loss"], label='Training set')
        axes[0].plot(self.history.history["val_loss"], label='Validation set')
        # axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Sharpe ratio")
        # axes[0].legend()
        axes[1].grid()
        axes[1].plot(self.history.history["portfolio_returns"], label='Training set')
        axes[1].plot(self.history.history["val_portfolio_returns"], label='Validation set')
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Portfolio returns")
        axes[1].legend()
        fig.savefig('history.png', dpi=300, bbox_inches='tight')
        plt.close()
