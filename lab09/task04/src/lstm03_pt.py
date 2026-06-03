from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

_torch = cast(Any, torch)
torch_device = _torch.device
torch_tensor = _torch.tensor
torch_argmax = _torch.argmax
torch_float32 = _torch.float32


class CharLSTM(nn.Module):
    def __init__(self, n_vocab: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


def prepare_char_data(filename: Path | str, seq_length: int = 100):
    filename = Path(filename)
    raw_text = filename.read_text(encoding="utf-8").lower()
    chars = sorted(set(raw_text))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    n_vocab = len(chars)
    n_chars = len(raw_text)

    dataX = []
    for i in range(n_chars - seq_length):
        dataX.append([char_to_int[c] for c in raw_text[i : i + seq_length]])

    return dataX, int_to_char, n_vocab


def generate_char_text(
    checkpoint_path: Path | str,
    text_path: Path | str,
    seq_length: int = 100,
    gen_length: int = 500,
    seed_index: int | None = None,
    device: str | None = None,
) -> tuple[str, str]:
    checkpoint_path = Path(checkpoint_path)
    dataX, int_to_char, n_vocab = prepare_char_data(text_path, seq_length)

    device = (
        torch_device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch_device(device)
    )
    model = CharLSTM(n_vocab).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if seed_index is None:
        seed_index = int(np.random.randint(0, len(dataX)))

    pattern = list(dataX[seed_index])
    seed_text = "".join(int_to_char[v] for v in pattern)
    generated_chars: list[str] = []

    for _ in range(gen_length):
        x = torch_tensor(pattern, dtype=torch_float32).reshape(1, seq_length, 1) / n_vocab
        with torch.no_grad():
            logits = model(x.to(device))
        idx = int(torch_argmax(logits, dim=1).item())
        generated_chars.append(int_to_char[idx])
        pattern.append(idx)
        pattern = pattern[1:]

    return seed_text, "".join(generated_chars)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate char-level text from a trained checkpoint."
    )
    parser.add_argument("checkpoint", help="Checkpoint file path")
    parser.add_argument("--text-path", default="data/text.txt")
    parser.add_argument("--length", type=int, default=500)
    args = parser.parse_args()

    seed_text, generated = generate_char_text(
        args.checkpoint, args.text_path, gen_length=args.length
    )
    print("Seed:")
    print(seed_text)
    print()
    print(generated)
    sys.exit(0)
