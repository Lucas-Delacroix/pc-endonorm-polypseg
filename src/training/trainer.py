import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
from evaluation.metrics import compute_all_metrics
from training.consistency import consistency_loss, soft_iou
from training.dino_teacher import DinoTeacher, feature_distillation_loss
from training.ema import ModelEMA
from training.losses import build_loss

class Trainer:

    def __init__(self, model, train_loader, val_loader, config, device='auto'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
        print(f'Using device: {self.device}')
        self.model = self.model.to(self.device)
        self.loss_fn = build_loss(config.get('loss', 'structure'))
        distillation_config = config.get('distillation', {}) or {}
        self.use_distillation = distillation_config.get('enabled', False)
        if self.use_distillation:
            self.teacher = DinoTeacher(model_name=distillation_config.get('model_name', 'dinov2_vits14'), image_size=distillation_config.get('teacher_image_size', 350), device=self.device)
            student_dim = self.model.backbone.embed_dims[-1]
            self.distill_head = nn.Conv2d(student_dim, self.teacher.embed_dim, kernel_size=1).to(self.device)
            self.distill_weight = float(distillation_config.get('weight', 1.0))
        else:
            self.teacher = None
            self.distill_head = None
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.epochs = self.config.get('epochs', 200)
        consistency_config = config.get('consistency', {}) or {}
        self.consistency_weight = float(consistency_config.get('consistency_weight', 1.0))
        ema_config = config.get('ema', {}) or {}
        self.use_ema = ema_config.get('enabled', False)
        self.ema = ModelEMA(self.model, decay=ema_config.get('decay', 0.999)).to(self.device) if self.use_ema else None
        swa_config = config.get('swa', {}) or {}
        self.use_swa = swa_config.get('enabled', False)
        self.swa_start = int(self.epochs * swa_config.get('start_epoch_ratio', 0.75))
        self.swa_update_bn = swa_config.get('update_bn', True)
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=float(swa_config.get('lr', 5e-05)))
        else:
            self.swa_model = None
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        self.best_raw_dice = float('-inf')
        self.best_ema_dice = float('-inf')
        self.history = []

    def _trainable_parameters(self):
        parameters = list(self.model.parameters())
        if self.distill_head is not None:
            parameters += list(self.distill_head.parameters())
        return parameters

    def _build_optimizer(self):
        opt_config = self.config.get('optimizer', {})
        name = opt_config.get('name', 'adam').lower()
        lr = float(opt_config.get('lr', 0.0001))
        weight_decay = float(opt_config.get('weight_decay', 1e-05))
        parameters = self._trainable_parameters()
        if name == 'adam':
            return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        if name == 'adamw':
            return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        raise ValueError(f"Unrecognized optimizer '{name}'.")

    def _build_scheduler(self):
        sched_config = self.config.get('scheduler', {})
        epochs = self.config.get('epochs', 200)
        t_max = int(sched_config.get('t_max', epochs))
        min_lr = float(sched_config.get('min_lr', 1e-06))
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=min_lr)

    def _segmentation_loss(self, preds, masks):
        if isinstance(preds, (list, tuple)):
            return (sum((self.loss_fn(p, masks) for p in preds)), preds[0])
        return (self.loss_fn(preds, masks), preds)

    def _distillation_loss(self, images, features):
        student = self.distill_head(features[-1])
        teacher = self.teacher.extract(images)
        teacher = F.interpolate(teacher, size=student.shape[-2:], mode='bilinear', align_corners=False)
        return feature_distillation_loss(student, teacher)

    def _supervised_step(self, batch):
        masks = batch['mask'].to(self.device)
        images = batch['image'].to(self.device)
        if self.use_distillation:
            preds, features = self.model(images, return_features=True)
            loss, main_pred = self._segmentation_loss(preds, masks)
            loss = loss + self.distill_weight * self._distillation_loss(images, features)
        else:
            preds = self.model(images)
            loss, main_pred = self._segmentation_loss(preds, masks)
        return (loss, main_pred, masks)

    def _consistency_step(self, batch):
        masks = batch['mask'].to(self.device)
        weak = batch['image_weak'].to(self.device)
        strong = batch['image_strong'].to(self.device)
        seg_loss, weak_main = self._segmentation_loss(self.model(weak), masks)
        strong_preds = self.model(strong)
        strong_main = strong_preds[0] if isinstance(strong_preds, (list, tuple)) else strong_preds
        weak_probability = torch.sigmoid(weak_main)
        strong_probability = torch.sigmoid(strong_main)
        weight = soft_iou(weak_probability.detach(), masks)
        consistency = consistency_loss(strong_probability, weak_probability.detach())
        loss = (1.0 - weight) * seg_loss + weight * self.consistency_weight * consistency
        return (loss, weak_main, masks)

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            if 'image_weak' in batch:
                loss, main_pred, masks = self._consistency_step(batch)
            else:
                loss, main_pred, masks = self._supervised_step(batch)
            loss.backward()
            self.optimizer.step()
            if self.use_ema:
                self.ema.update(self.model)
            with torch.no_grad():
                binary_preds = (torch.sigmoid(main_pred) > 0.5).float()
                total_dice += compute_all_metrics(binary_preds.cpu(), masks.cpu())['dice']
            total_loss += loss.item()
        n = len(self.train_loader)
        return {'loss': total_loss / n, 'dice': total_dice / n}

    @torch.no_grad()
    def _evaluate(self, model):
        model.eval()
        total_loss = 0.0
        all_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            preds = model(images)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            total_loss += self.loss_fn(preds, masks).item()
            binary_preds = (torch.sigmoid(preds) > 0.5).float()
            metrics = compute_all_metrics(binary_preds.cpu(), masks.cpu())
            for k in all_metrics:
                all_metrics[k] += metrics[k]
        n = len(self.val_loader)
        return {'loss': total_loss / n, **{k: v / n for k, v in all_metrics.items()}}

    def _update_swa_batchnorm(self):
        batchnorm_modules = [module for module in self.swa_model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        if not batchnorm_modules:
            return
        momenta = {}
        for module in batchnorm_modules:
            module.reset_running_stats()
            momenta[module] = module.momentum
            module.momentum = None
        self.swa_model.train()
        with torch.no_grad():
            for batch in self.train_loader:
                self.swa_model(batch['image'].to(self.device))
        for module, momentum in momenta.items():
            module.momentum = momentum

    def _save(self, filename, model, epoch, metrics):
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics, 'config': self.config}, self.checkpoint_dir / filename)

    def _log_epoch(self, epoch, epochs, train, val, ema, elapsed):
        ema_text = f" ema_dice={ema['dice']:.4f}" if ema else ''
        print(f"Epoch [{epoch:03d}/{epochs}] loss={train['loss']:.4f} dice={train['dice']:.4f} | val_loss={val['loss']:.4f} val_dice={val['dice']:.4f} val_iou={val['iou']:.4f}{ema_text} ({elapsed:.1f}s)")

    def fit(self):
        epochs = self.epochs
        log_every = self.config.get('log_every_n_epochs', 1)
        print(f'\nStarting training for {epochs} epochs...')
        print('=' * 70)
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_metrics = self._train_epoch()
            val_metrics = self._evaluate(self.model)
            ema_metrics = self._evaluate(self.ema.module) if self.use_ema else None
            if self.use_swa and epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            elif self.scheduler:
                self.scheduler.step()
            elapsed = time.time() - start
            record = {'epoch': epoch, 'train': train_metrics, 'val': val_metrics, 'lr': self.optimizer.param_groups[0]['lr']}
            if ema_metrics:
                record['ema'] = ema_metrics
            self.history.append(record)
            if epoch % log_every == 0:
                self._log_epoch(epoch, epochs, train_metrics, val_metrics, ema_metrics, elapsed)
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save('best.pth', self.model, epoch, val_metrics)
            if self.use_ema or self.use_swa:
                self._save('last.pt', self.model, epoch, val_metrics)
                if val_metrics['dice'] > self.best_raw_dice:
                    self.best_raw_dice = val_metrics['dice']
                    self._save('best_raw.pt', self.model, epoch, val_metrics)
                if ema_metrics and ema_metrics['dice'] > self.best_ema_dice:
                    self.best_ema_dice = ema_metrics['dice']
                    self._save('best_ema.pt', self.ema.module, epoch, ema_metrics)
        if self.use_swa:
            if self.swa_update_bn:
                self._update_swa_batchnorm()
            swa_metrics = self._evaluate(self.swa_model)
            self._save('best_swa.pt', self.swa_model, epochs, swa_metrics)
            print(f"SWA validation: dice={swa_metrics['dice']:.4f} iou={swa_metrics['iou']:.4f}")
        print('=' * 70)
        print(f'Training completed. Best validation loss: {self.best_val_loss:.6f}')
        return self.history
