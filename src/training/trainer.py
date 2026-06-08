import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from evaluation.metrics import compute_all_metrics
from training.losses import get_loss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "auto",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        if device == "auto":
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device

        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

        self.loss_fn = get_loss(config.get("loss", "structure"))
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.history = []

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_config = self.config.get("optimizer", {})
        name = opt_config.get("name", "adam").lower()
        lr = float(opt_config.get("lr", 1e-4))
        weight_decay = float(opt_config.get("weight_decay", 1e-5))

        if name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        if name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        if name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        raise ValueError(f"Unrecognized optimizer '{name}'.")

    def _build_scheduler(self):
        sched_config = self.config.get("scheduler", {})
        name = sched_config.get("name", "cosine").lower()
        epochs = self.config.get("epochs", 200)
        t_max = int(sched_config.get("t_max", epochs))
        min_lr = float(sched_config.get("min_lr", 1e-6))

        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=min_lr,
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get("step_size", 50),
                gamma=sched_config.get("gamma", 0.1),
            )
        if name == "none":
            return None
        raise ValueError(f"Unrecognized scheduler '{name}'.")

    def _train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(images)

            if isinstance(preds, (list, tuple)):
                loss = sum(self.loss_fn(p, masks) for p in preds)
                preds = preds[0]
            else:
                loss = self.loss_fn(preds, masks)

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                binary_preds = (torch.sigmoid(preds) > 0.5).float()
                metrics = compute_all_metrics(binary_preds.cpu(), masks.cpu())
                total_dice += metrics["dice"]

            total_loss += loss.item()

        n = len(self.train_loader)
        return {
            "loss": total_loss / n,
            "dice": total_dice / n,
        }

    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            preds = self.model(images)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            loss = self.loss_fn(preds, masks)
            total_loss += loss.item()

            binary_preds = (torch.sigmoid(preds) > 0.5).float()
            metrics = compute_all_metrics(binary_preds.cpu(), masks.cpu())
            for k in all_metrics:
                all_metrics[k] += metrics[k]

        n = len(self.val_loader)
        return {
            "loss": total_loss / n,
            **{k: v / n for k, v in all_metrics.items()},
        }

    def _save_checkpoint(self, epoch: int, metrics: dict):
        path = self.checkpoint_dir / "best.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            path,
        )
        print(f"  Checkpoint saved (epoch {epoch}, loss={metrics['loss']:.6f})")

    def _log_epoch(self, epoch: int, epochs: int, train: dict, val: dict, elapsed: float):
        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"loss={train['loss']:.4f} "
            f"dice={train['dice']:.4f} | "
            f"val_loss={val['loss']:.4f} "
            f"val_dice={val['dice']:.4f} "
            f"val_iou={val['iou']:.4f} "
            f"({elapsed:.1f}s)"
        )

    def fit(self):
        epochs = self.config.get("epochs", 200)
        log_every = self.config.get("log_every_n_epochs", 1)

        print(f"\nStarting training for {epochs} epochs...")
        print("=" * 70)

        for epoch in range(1, epochs + 1):
            start = time.time()

            train_metrics = self._train_epoch()
            val_metrics = self._val_epoch()

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - start

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(record)

            if epoch % log_every == 0:
                self._log_epoch(epoch, epochs, train_metrics, val_metrics, elapsed)

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, val_metrics)

        print("=" * 70)
        print(f"Training completed. Best validation loss: {self.best_val_loss:.6f}")
        return self.history
