import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import LSTM, Conv1D, GlobalMaxPooling1D, RepeatVector, TimeDistributed, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam


class TimeSeriesEmbedding:
    def __init__(self, data, window_size=100):
        self.window_size = window_size
        self.data = data
        self.X = self.create_sequences(self.data, self.window_size)
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], 1))

    def create_sequences(self, data, window_size):
        X = []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
        return np.array(X)

    def train_lstm(self, epochs=20):
        model = Sequential([
            LSTM(50, activation='tanh', input_shape=(self.window_size, 1), kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal', bias_initializer='zeros'),
            Dense(1, activation='linear')
        ])

        # Use Adam optimizer with a smaller learning rate
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)

        model.compile(optimizer=optimizer, loss='mse')

        # Check if any NaNs in input data
        assert not np.isnan(self.X).any(), "Input data contains NaN values."

        # Check for NaNs during training
        callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch + 1}, Loss: {logs['loss']}"))

        model.fit(self.X, self.X[:, -1], epochs=epochs, verbose=1, callbacks=[callback])
        self.encoder_model = Sequential([model.layers[0]])

    def get_embeddings(self):
        embeddings = self.encoder_model.predict(self.X)
        return embeddings
