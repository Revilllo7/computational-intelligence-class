from pathlib import Path

from src.utils.torch_runtime import prepare_torch_import

prepare_torch_import()

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.training.trainer import train_model  # noqa: E402


class ToyImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image = torch.randn(3, 16, 16)
        label = torch.tensor(index % 2, dtype=torch.long)
        return image, label, f"sample_{index}.jpg"


def test_train_model_saves_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(42)

    train_loader = DataLoader(ToyImageDataset(length=16), batch_size=4, shuffle=True)
    validation_loader = DataLoader(ToyImageDataset(length=8), batch_size=4, shuffle=False)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 16 * 16, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

    checkpoint = tmp_path / "toy_model.pt"
    result = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        learning_rate=0.001,
        epochs=2,
        weight_decay=0.0,
        checkpoint_path=checkpoint,
        device=torch.device("cpu"),
    )

    assert checkpoint.exists()
    assert len(result.history) == 2
    assert 0.0 <= result.best_validation_accuracy <= 1.0
