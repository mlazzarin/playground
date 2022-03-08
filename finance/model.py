"""
Author: Marco Lazzarin
Description: Deep learning model from "Deep learning for portfolio optimization"
"""
# pylint: disable=import-error
# pylint: disable=no-name-in-module

# Libraries
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, TimeDistributed

from metrics import portfolio_returns_annualized, sharpe, portfolio_returns, portfolio_evolution


class PortfolioOptimizer(ABC):
    """Abstract portfolio optimizer"""

    def __init__(self, data):
        """Initialize data members"""

        # Train-valid-test split
        self.data      = data
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

    @abstractmethod
    def build(self):
        """Build the deep learning model."""
        raise(NotImplementedError)

    @abstractmethod
    def fit(self):
        """Fit the model to the dataset."""
        raise(NotImplementedError)

    @abstractmethod
    def predict(self, data):
        """Predict asset allocation."""
        raise(NotImplementedError)

    def plot(self):
        """Various plot to analise the results."""

        # Plot training history
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].grid()
        axes[0].plot(self.history["loss"], label='Training set')
        axes[0].plot(self.history["val_loss"], label='Validation set')
        # axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Sharpe ratio")
        # axes[0].legend()
        axes[1].grid()
        axes[1].plot(self.history["portfolio_returns"], label='Training set')
        axes[1].plot(self.history["val_portfolio_returns"], label='Validation set')
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Portfolio returns")
        axes[1].legend()
        fig.savefig('history.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Compute overall gain / loss
        prediction = self.predict(self.data)
        prediction = np.squeeze(prediction)
        port_ev    = portfolio_evolution(self.data[0, :, 1::2], prediction)

        # Plot allocations and gain/loss on the whole dataset
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].plot(self.data[0, :, 0] / self.data[0, 0, 0] - 1, label="VTI")
        axes[0].plot(self.data[0, :, 2] / self.data[0, 0, 2] - 1, label="AGG")
        axes[0].plot(self.data[0, :, 4] / self.data[0, 0, 4] - 1, label="DBC")
        axes[0].plot(self.data[0, :, 6] / self.data[0, 0, 6] - 1, label="SGOL")
        axes[0].plot(port_ev, label="Portfolio", color='k', ls='--')
        axes[0].title.set_text("Gain / loss")
        # axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("Gain / loss")
        axes[0].legend()
        axes[1].plot(prediction[:, 0], label="VTI")
        axes[1].plot(prediction[:, 1], label="AGG")
        axes[1].plot(prediction[:, 2], label="DBC")
        axes[1].plot(prediction[:, 3], label="SGOL")
        axes[1].title.set_text("Allocations")
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Portfolio weight")
        fig.savefig('allocations.png', dpi=300, bbox_inches='tight')

        # Plot gain/loss on training / validation / test set independently
        fig, axes = plt.subplots(3, 1, figsize=(16, 9))
        for index, data in enumerate((self.train_set, self.valid_set, self.test_set)):
            axes[index].plot(data[0, :, 0] / data[0, 0, 0] - 1, label="VTI")
            axes[index].plot(data[0, :, 2] / data[0, 0, 2] - 1, label="AGG")
            axes[index].plot(data[0, :, 4] / data[0, 0, 4] - 1, label="DBC")
            axes[index].plot(data[0, :, 6] / data[0, 0, 6] - 1, label="SGOL")
            prediction = self.predict(data)
            prediction = np.squeeze(prediction)
            port_ev    = portfolio_evolution(data[0, :, 1::2], prediction)
            axes[index].plot(port_ev, label="Portfolio", color='k', ls='--')
            axes[index].title.set_text("Gain / loss")
            axes[index].set_ylabel("Gain / loss")
        axes[2].set_xlabel("Time step")
        axes[2].legend()
        fig.savefig('returns.png', dpi=300, bbox_inches='tight')


class DeepLearningOptimizer(PortfolioOptimizer):
    """Deep learning model for portfolio optimization"""
    # pylint: disable=too-many-instance-attributes

    def build(self):
        """Build the deep learning model."""
        self.model = Sequential()
        self.model.add(Input(shape=(None, self.n_features)))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(TimeDistributed(Dense(4, activation="softmax")))
        self.model.compile(loss=sharpe, optimizer="adam",
                           metrics=[portfolio_returns, portfolio_returns_annualized])
        self.model.summary()

    def fit(self, epochs=1000):
        """Fit the deep learning model."""

        early_stopping = EarlyStopping(monitor="val_loss",
                                       patience=100,
                                       restore_best_weights=True)
        self.history = self.model.fit(self.train_set, self.train_returns,
                                      validation_data=(self.valid_set, self.valid_returns),
                                      epochs=epochs, callbacks=[early_stopping]).history
        test_results = self.model.evaluate(self.test_set, self.test_returns)
        print(f"\n\n ==] Test Sharpe ratio: {test_results[0]}")
        print(f" ==] Test return: {test_results[1]}")
        print(f" ==] Test annualized return: {test_results[2]}")

    def predict(self, data):
        """Predict asset allocation."""
        return self.model.predict(data)