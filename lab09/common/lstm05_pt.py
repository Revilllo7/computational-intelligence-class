# Load word-token LSTM network and generate text (PyTorch)
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import wordpunct_tokenize

# ── data (same preprocessing as training) ─────────────────────────────────────
raw_text = Path("wonderland.txt").read_text(encoding="utf-8").lower()
tokenized_text = wordpunct_tokenize(raw_text)
tokens = sorted(dict.fromkeys(tokenized_text))

tok_to_int = {t: i for i, t in enumerate(tokens)}
int_to_tok = {i: t for i, t in enumerate(tokens)}
n_tokens = len(tokenized_text)
n_token_vocab = len(tokens)
print(f"Total Tokens: {n_tokens}  |  Unique Tokens (Vocab): {n_token_vocab}")

seq_length = 100
dataX = []
for i in range(n_tokens - seq_length):
    dataX.append([tok_to_int[t] for t in tokenized_text[i : i + seq_length]])


# ── model ─────────────────────────────────────────────────────────────────────
class TokenLSTM(nn.Module):
    def __init__(self, n_vocab, hidden=256, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_vocab)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TokenLSTM(n_token_vocab).to(device)

# pick the checkpoint with the lowest loss
checkpoints = Path("./").glob("big-token-model-*.pt")
if not checkpoints:
    sys.exit("No checkpoint found. Train with lstm04_pt.py first.")
checkpoint = min(
    checkpoints,
    key=lambda f: float(f.stem.rsplit("-", 1)[-1]),
)
print(f"Loading: {checkpoint}")
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

# ── generate ──────────────────────────────────────────────────────────────────
start = np.random.randint(0, len(dataX))
pattern = list(dataX[start])
print("Seed:")
print('"', " ".join(int_to_tok[v] for v in pattern), '"')
print("\nGenerated text:")

for _ in range(100):
    x = torch.tensor(pattern, dtype=torch.float32).reshape(1, seq_length, 1) / n_token_vocab
    with torch.no_grad():
        logits = model(x.to(device))
    idx = int(torch.argmax(logits, dim=1).item())
    sys.stdout.write(int_to_tok[idx] + " ")
    sys.stdout.flush()
    pattern.append(idx)
    pattern = pattern[1:]

print("\nDone.")
