# Word-token LSTM to Generate Text for Alice in Wonderland (PyTorch)
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import wordpunct_tokenize
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── data ──────────────────────────────────────────────────────────────────────
raw_text = Path("wonderland.txt").read_text(encoding="utf-8").lower()
tokenized_text = wordpunct_tokenize(raw_text)
tokens = sorted(dict.fromkeys(tokenized_text))  # unique, order-preserving

tok_to_int = {t: i for i, t in enumerate(tokens)}
n_tokens = len(tokenized_text)
n_token_vocab = len(tokens)
print(f"Total Tokens: {n_tokens}  |  Unique Tokens (Vocab): {n_token_vocab}")

seq_length = 100
dataX, dataY = [], []
for i in range(n_tokens - seq_length):
    dataX.append([tok_to_int[t] for t in tokenized_text[i : i + seq_length]])
    dataY.append(tok_to_int[tokenized_text[i + seq_length]])
print(f"Total Patterns: {len(dataX)}")

X = np.array(dataX, dtype=np.float32).reshape(-1, seq_length, 1) / n_token_vocab
y = np.array(dataY, dtype=np.int64)

dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
loader = DataLoader(dataset, batch_size=128, shuffle=True)


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
print(f"Using device: {device}")

model = TokenLSTM(n_token_vocab).to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# resume from best existing checkpoint if available
checkpoints = Path().glob("big-token-model-*.pt")
if checkpoints:
    ckpt = min(
        checkpoints,
        key=lambda f: float(f.stem.rsplit("-", 1)[-1]),
    )
    print(f"Resuming from: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))

# ── training ──────────────────────────────────────────────────────────────────
best_loss = float("inf")
epochs = 6

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    bar = tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs}", unit="batch")
    for xb, yb in bar:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(xb)
        bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(dataset)
    bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        ckpt = f"big-token-model-{epoch:02d}-{avg_loss:.4f}.pt"
        torch.save(model.state_dict(), ckpt)
        tqdm.write(f"  -> saved {ckpt}")
