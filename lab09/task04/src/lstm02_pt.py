from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_torch = cast(Any, torch)
torch_from_numpy = _torch.from_numpy
torch_device = _torch.device


class CharLSTM(nn.Module):
    def __init__(self, n_vocab: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


def prepare_char_dataset(filename: Path | str, seq_length: int = 100):
    filename = Path(filename)
    raw_text = filename.read_text(encoding="utf-8").lower()
    chars = sorted(set(raw_text))
    char_to_int = {c: i for i, c in enumerate(chars)}
    n_chars = len(raw_text)
    n_vocab = len(chars)

    dataX, dataY = [], []
    for i in range(n_chars - seq_length):
        dataX.append([char_to_int[c] for c in raw_text[i : i + seq_length]])
        dataY.append(char_to_int[raw_text[i + seq_length]])

    X = np.array(dataX, dtype=np.float32).reshape(-1, seq_length, 1) / n_vocab
    y = np.array(dataY, dtype=np.int64)
    dataset = TensorDataset(torch_from_numpy(X), torch_from_numpy(y))

    return dataset, chars, char_to_int, n_vocab


def train_char_model(
    text_path: Path | str,
    epochs: int,
    seq_length: int = 100,
    batch_size: int = 128,
    hidden: int = 256,
    dropout: float = 0.2,
    lr: float = 1e-3,
    device: str | None = None,
    output_dir: Path | str = ".",
) -> tuple[Path, float]:
    text_path = Path(text_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, _, _, n_vocab = prepare_char_dataset(text_path, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = (
        torch_device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch_device(device)
    )
    model = CharLSTM(n_vocab, hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        bar = tqdm(loader, desc=f"Char epoch {epoch:02d}/{epochs}", unit="batch")
        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
            bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(dataset)  # pyright: ignore[reportPossiblyUnboundVariable]
    checkpoint = output_dir / f"char-model-{epochs:02d}-epochs-{avg_loss:.4f}.pt"
    torch.save(model.state_dict(), checkpoint)
    return checkpoint, avg_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the char-level LSTM model.")
    parser.add_argument("text_path", nargs="?", default="data/text.txt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    ckpt, loss = train_char_model(
        args.text_path,
        epochs=args.epochs,
        output_dir=Path(args.output_dir),
    )
    print(f"Saved checkpoint: {ckpt}")
    print(f"Final loss: {loss:.4f}")
