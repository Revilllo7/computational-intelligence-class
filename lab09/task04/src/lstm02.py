# Small LSTM Network to Generate Text for Alice in Wonderland
from pathlib import Path

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from numpy.typing import NDArray

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = Path(filename).read_text(encoding="utf-8").lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
print("Characters: ")
print(chars)
char_to_int = dict((c, i) for i, c in enumerate(chars))
print("CharacterToNumbers: ")
print(char_to_int)

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i : i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X: NDArray[np.float64] = np.reshape(
    dataX,
    (n_patterns, seq_length, 1),
).astype(np.float64)
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y: np.ndarray = np.eye(n_vocab, dtype=np.float64)[dataY]
# define the LSTM model
timesteps = int(X.shape[1])
features = int(X.shape[2])
output_dim = int(y.shape[1])

model = Sequential()
model.add(LSTM(256, input_shape=(timesteps, features)))
model.add(Dropout(0.2))
model.add(Dense(output_dim, activation="softmax"))
# filename = "weights-improvement-03-2.7711.hdf5"
# model.load_weights(filename)
model.compile(loss="categorical_crossentropy", optimizer="adam")
# define the checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=5, batch_size=128, callbacks=callbacks_list)
