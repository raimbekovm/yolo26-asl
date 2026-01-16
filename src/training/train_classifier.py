"""ASL keypoint classifier training."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentation import get_train_augmentation, get_val_augmentation
from src.data.dataset import ASLKeypointDataset
from src.models.classifier import create_classifier
from src.utils.constants import ASL_CLASSES, OUTPUTS_DIR, WEIGHTS_DIR
from src.utils.device import get_device, set_seed


class ClassifierTrainer:
    """
    Trainer for ASL keypoint classifier.

    Handles training loop, validation, checkpointing, and metrics.

    Example:
        >>> trainer = ClassifierTrainer(model_type="mlp")
        >>> trainer.train(data_dir="data/processed/classifier_dataset")
    """

    def __init__(
        self,
        model_type: str = "mlp",
        device: Optional[str] = None,
        seed: int = 42,
        **model_kwargs,
    ):
        """
        Initialize trainer.

        Args:
            model_type: Model architecture ('mlp' or 'transformer').
            device: Device to train on.
            seed: Random seed.
            **model_kwargs: Additional model arguments.
        """
        self.device = torch.device(device) if device else get_device()
        self.seed = seed
        set_seed(seed)

        # Create model
        self.model = create_classifier(model_type, **model_kwargs)
        self.model.to(self.device)
        self.model_type = model_type

        logger.info(f"Created {model_type} classifier on {self.device}")
        logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(
        self,
        data_dir: Path,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        label_smoothing: float = 0.1,
        patience: int = 10,
        output_dir: Optional[Path] = None,
        run_name: str = "classifier_training",
    ) -> dict:
        """
        Run training.

        Args:
            data_dir: Directory with train/val/test splits.
            epochs: Number of training epochs.
            batch_size: Batch size.
            learning_rate: Initial learning rate.
            weight_decay: Weight decay for optimizer.
            warmup_epochs: Number of warmup epochs.
            label_smoothing: Label smoothing factor.
            patience: Early stopping patience.
            output_dir: Output directory.
            run_name: Name for this training run.

        Returns:
            Dictionary with training history.
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir) if output_dir else OUTPUTS_DIR / "classifier_training"
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Setup data
        train_dataset = ASLKeypointDataset(
            data_dir / "train",
            transform=get_train_augmentation(),
        )
        val_dataset = ASLKeypointDataset(
            data_dir / "val",
            transform=get_val_augmentation(),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")

        # Setup training
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # Training loop
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)

            # Validation
            val_loss, val_acc = self._validate(val_loader, criterion)

            # Update scheduler
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

            # Log metrics
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["learning_rate"].append(lr)

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | "
                f"LR: {lr:.6f}"
            )

            # Checkpointing
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                # Save best model
                self._save_checkpoint(
                    run_dir / "best.pt",
                    epoch,
                    optimizer,
                    val_acc,
                )
                logger.info(f"New best model: {val_acc:.2%}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    run_dir / f"epoch_{epoch+1}.pt",
                    epoch,
                    optimizer,
                    val_acc,
                )

        # Save final model
        self._save_checkpoint(run_dir / "last.pt", epochs - 1, optimizer, val_acc)

        # Copy best model to weights dir
        best_weights = run_dir / "best.pt"
        if best_weights.exists():
            target = WEIGHTS_DIR / "asl_classifier.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy(best_weights, target)
            logger.info(f"Best model saved to {target}")

        # Save history
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Training complete. Best val accuracy: {best_val_acc:.2%}")

        return history

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float]:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc="Training", leave=False)
        for keypoints, labels in pbar:
            keypoints = keypoints.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            logits = self.model(keypoints)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        return total_loss / total, correct / total

    def _validate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for keypoints, labels in loader:
                keypoints = keypoints.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(keypoints)
                loss = criterion(logits, labels)

                total_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        return total_loss / total, correct / total

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        val_acc: float,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "config": self.model.get_config(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "val_accuracy": val_acc,
        }
        torch.save(checkpoint, path)

    def evaluate(
        self,
        data_dir: Path,
        split: str = "test",
        batch_size: int = 64,
    ) -> dict:
        """
        Evaluate model on test set.

        Args:
            data_dir: Data directory.
            split: Split to evaluate on.
            batch_size: Batch size.

        Returns:
            Evaluation metrics.
        """
        from sklearn.metrics import classification_report, confusion_matrix

        dataset = ASLKeypointDataset(
            Path(data_dir) / split,
            transform=get_val_augmentation(),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for keypoints, labels in tqdm(loader, desc="Evaluating"):
                keypoints = keypoints.to(self.device)
                logits = self.model(keypoints)
                preds = logits.argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        accuracy = (all_preds == all_labels).mean()
        report = classification_report(
            all_labels,
            all_preds,
            target_names=ASL_CLASSES,
            output_dict=True,
        )
        cm = confusion_matrix(all_labels, all_preds)

        logger.info(f"Test Accuracy: {accuracy:.2%}")
        logger.info(f"\n{classification_report(all_labels, all_preds, target_names=ASL_CLASSES)}")

        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": all_preds.tolist(),
            "labels": all_labels.tolist(),
        }


def main():
    """CLI entry point for classifier training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train ASL classifier")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="classifier_training")

    args = parser.parse_args()

    trainer = ClassifierTrainer(model_type=args.model_type, device=args.device)
    trainer.train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
