# Load LSTM network and generate text (PyTorch)
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── data (same preprocessing as training) ─────────────────────────────────────
filename = "wonderland.txt"
raw_text = Path(filename).read_text(encoding="utf-8").lower()

chars = sorted(set(raw_text))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}
n_vocab = len(chars)
n_chars = len(raw_text)

seq_length = 100
dataX = []
for i in range(n_chars - seq_length):
    dataX.append([char_to_int[c] for c in raw_text[i : i + seq_length]])


# ── model ─────────────────────────────────────────────────────────────────────
class CharLSTM(nn.Module):
    def __init__(self, n_vocab, hidden=256, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_vocab)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharLSTM(n_vocab).to(device)

# pick the checkpoint with the lowest loss (last field before .pt)
checkpoints = Path().glob("weights-improvement-*.pt")
if not checkpoints:
    sys.exit("No checkpoint found. Train with lstm02_pt.py first.")
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
print('"', "".join(int_to_char[v] for v in pattern), '"')
print()

for _ in range(500):
    x = torch.tensor(pattern, dtype=torch.float32).reshape(1, seq_length, 1) / n_vocab
    with torch.no_grad():
        logits = model(x.to(device))
    idx = int(torch.argmax(logits, dim=1).item())
    sys.stdout.write(int_to_char[idx])
    sys.stdout.flush()
    pattern.append(idx)
    pattern = pattern[1:]

print("\nDone.")
