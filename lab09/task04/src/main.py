from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .lstm02_pt import train_char_model
from .lstm03_pt import generate_char_text
from .lstm04_pt import train_token_model
from .lstm05_pt import generate_token_text

DEFAULT_TEXT_PATH = Path(__file__).resolve().parents[1] / "data" / "text.txt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
CHAR_SEQ_LENGTH = 100
TOKEN_SEQ_LENGTH = 100


def run_full_pipeline(
    text_path: Path | str = DEFAULT_TEXT_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    epochs_list: Sequence[int] = (1, 5, 15),
) -> int:
    text_path = Path(text_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    char_output_dir = output_dir / "char"
    token_output_dir = output_dir / "token"
    char_output_dir.mkdir(exist_ok=True)
    token_output_dir.mkdir(exist_ok=True)

    print(f"Using text file: {text_path}")
    print(f"Saving checkpoints and output under: {output_dir}")

    for epochs in epochs_list:
        print(f"\n=== Training char-level model for {epochs} epoch(s) ===")
        char_checkpoint, char_loss = train_char_model(
            text_path,
            epochs=epochs,
            seq_length=CHAR_SEQ_LENGTH,
            output_dir=char_output_dir,
        )
        print(f"Saved char checkpoint: {char_checkpoint.name} (loss={char_loss:.4f})")

        seed_text, generated = generate_char_text(
            char_checkpoint,
            text_path,
            seq_length=CHAR_SEQ_LENGTH,
            gen_length=300,
        )
        print("\nChar generator seed excerpt:")
        print(seed_text[:200].replace("\n", "\\n"))
        print("\nGenerated char text:")
        print(generated)

    for epochs in epochs_list:
        print(f"\n=== Training token-level model for {epochs} epoch(s) ===")
        token_checkpoint, token_loss = train_token_model(
            text_path,
            epochs=epochs,
            seq_length=TOKEN_SEQ_LENGTH,
            output_dir=token_output_dir,
        )
        print(f"Saved token checkpoint: {token_checkpoint.name} (loss={token_loss:.4f})")

        seed_text, generated = generate_token_text(
            token_checkpoint,
            text_path,
            seq_length=TOKEN_SEQ_LENGTH,
            gen_length=100,
        )
        print("\nToken generator seed excerpt:")
        print(seed_text)
        print("\nGenerated token text:")
        print(generated)

    return 0


def main() -> int:
    return run_full_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
