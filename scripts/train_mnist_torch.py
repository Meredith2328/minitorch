import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from mnist import MNIST
from torch.utils.data import DataLoader, TensorDataset


IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CLASSES = 10


class MnistConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple CNN on MNIST with a PyTorch DataLoader workflow."
    )
    parser.add_argument(
        "--data-dir",
        default="project/data",
        help="Directory containing train-images-idx3-ubyte and train-labels-idx1-ubyte.",
    )
    parser.add_argument("--output-dir", default="outputs/mnist-torch")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-stop", type=int, default=5000)
    parser.add_argument("--eval-start", type=int, default=10000)
    parser.add_argument("--eval-stop", type=int, default=10500)
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_mnist(data_dir: str):
    loader = MNIST(data_dir)
    images, labels = loader.load_training()
    x = torch.tensor(images, dtype=torch.float32).view(-1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
    x = x / 255.0
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def build_dataloaders(args: argparse.Namespace):
    x, y = load_mnist(args.data_dir)
    train_x = x[args.train_start : args.train_stop]
    train_y = y[args.train_start : args.train_stop]
    eval_x = x[args.eval_start : args.eval_stop]
    eval_y = y[args.eval_start : args.eval_stop]

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(
        TensorDataset(eval_x, eval_y),
        batch_size=args.batch_size,
        shuffle=False,
    )
    return train_loader, eval_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    train_loader, eval_loader = build_dataloaders(args)
    model = MnistConvNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        eval_loss, eval_acc = evaluate(model, eval_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "eval_loss": eval_loss,
            "eval_accuracy": eval_acc,
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2%} "
            f"eval_loss={eval_loss:.4f} "
            f"eval_acc={eval_acc:.2%}"
        )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
