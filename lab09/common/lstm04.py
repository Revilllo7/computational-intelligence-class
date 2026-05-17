# Small LSTM Network to Generate Text for Alice in Wonderland
from pathlib import Path

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from nltk.tokenize import wordpunct_tokenize
from numpy.typing import NDArray

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = Path(filename).read_text(encoding="utf-8").lower()
# create mapping of unique chars to integers
tokenized_text = wordpunct_tokenize(raw_text)
tokens = sorted(list(dict.fromkeys(tokenized_text)))

# print("Tokens: ")
# print(tokens)
tok_to_int = dict((c, i) for i, c in enumerate(tokens))
# print("TokensToNumbers: ")
print(tok_to_int)

# summarize the loaded data
n_tokens = len(tokenized_text)
n_token_vocab = len(tokens)
print("Total Tokens: ", n_tokens)
print("Unique Tokens (Token Vocab): ", n_token_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_tokens - seq_length, 1):
    seq_in = tokenized_text[i : i + seq_length]
    seq_out = tokenized_text[i + seq_length]
    dataX.append([tok_to_int[tok] for tok in seq_in])
    dataY.append(tok_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X: NDArray[np.float64] = np.reshape(
    dataX,
    (n_patterns, seq_length, 1),
).astype(np.float64)
# normalize
X = X / float(n_token_vocab)
# one hot encode the output variable
y: np.ndarray = np.eye(n_token_vocab, dtype=np.float64)[dataY]
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation="softmax"))
filename = "big-token-model-30-2.3772.hdf5"
model.load_weights(filename)
model.compile(loss="categorical_crossentropy", optimizer="adam")
# define the checkpoint
filepath = "big-token-model-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
