"""
Author: Marco Lazzarin
Description: Main file for "Deep learning for portfolio optimization"
"""

# Set env variables for reproducibility
# pylint: disable=wrong-import-position
import os
os.environ['PYTHONHASHSEED'] = str(2022)
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Libraries
import random

import numpy as np
import tensorflow as tf

from data import load_data
from model import DeepLearningOptimizer

# https://keras.io/getting_started/faq/
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(2022)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
random.seed(2022)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(2022)


def main():
    """Main function."""

    # ----- LOAD AND PREPROCESS DATA -----

    # Retrive data the historical series and concatenate them in a single
    # NumPy array of shape (1, timesteps, 2 * n_assets)
    data = np.concatenate((load_data("data/VTI.csv", ["Close"]),
                           load_data("data/AGG.csv", ["Close"]),
                           load_data("data/DBC.csv", ["Close"]),
                           load_data("data/SGOL.csv", ["Close"])), axis=1)
    data = np.squeeze(data)
    data = data.reshape(1, *data.shape) # reshape the data to expose the batch size (one)

    # Extract the returns (third index, odd numbers) and double-check the correctness
    returns = data[:, :, 1::2]
    test_returns = data[:, 1:, 0::2] / data[:, :-1, 0::2] - 1.0
    np.testing.assert_equal(returns[:, 1:, :], test_returns)

    # ----- BUILD AND TRAIN MODEL -----

    model = DeepLearningOptimizer(data)
    model.build()
    model.fit()
    model.plot()

if __name__ == "__main__":
    main()
