from __future__ import annotations

import re
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


class TokenLSTM(nn.Module):
    def __init__(self, n_vocab: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


def prepare_token_data(filename: Path | str, seq_length: int = 100):
    filename = Path(filename)
    raw_text = filename.read_text(encoding="utf-8").lower()
    tokenized_text = re.findall(r"\w+|[^\w\s]", raw_text, flags=re.UNICODE)
    tokens = sorted(dict.fromkeys(tokenized_text))
    tok_to_int = {t: i for i, t in enumerate(tokens)}
    int_to_tok = {i: t for i, t in enumerate(tokens)}
    n_tokens = len(tokenized_text)
    n_token_vocab = len(tokens)

    dataX = []
    for i in range(n_tokens - seq_length):
        dataX.append([tok_to_int[t] for t in tokenized_text[i : i + seq_length]])

    return dataX, int_to_tok, n_token_vocab


def generate_token_text(
    checkpoint_path: Path | str,
    text_path: Path | str,
    seq_length: int = 100,
    gen_length: int = 100,
    seed_index: int | None = None,
    device: str | None = None,
) -> tuple[str, str]:
    checkpoint_path = Path(checkpoint_path)
    dataX, int_to_tok, n_token_vocab = prepare_token_data(text_path, seq_length)

    device = (
        torch_device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch_device(device)
    )
    model = TokenLSTM(n_token_vocab).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if seed_index is None:
        seed_index = int(np.random.randint(0, len(dataX)))

    pattern = list(dataX[seed_index])
    seed_text = " ".join(int_to_tok[v] for v in pattern)
    generated_tokens: list[str] = []

    for _ in range(gen_length):
        x = torch_tensor(pattern, dtype=torch_float32).reshape(1, seq_length, 1) / n_token_vocab
        with torch.no_grad():
            logits = model(x.to(device))
        idx = int(torch_argmax(logits, dim=1).item())
        generated_tokens.append(int_to_tok[idx])
        pattern.append(idx)
        pattern = pattern[1:]

    return seed_text, " ".join(generated_tokens)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate token-level text from a trained checkpoint."
    )
    parser.add_argument("checkpoint", help="Checkpoint file path")
    parser.add_argument("--text-path", default="data/text.txt")
    parser.add_argument("--length", type=int, default=100)
    args = parser.parse_args()

    seed_text, generated = generate_token_text(
        args.checkpoint, args.text_path, gen_length=args.length
    )
    print("Seed:")
    print(seed_text)
    print()
    print(generated)
    sys.exit(0)
