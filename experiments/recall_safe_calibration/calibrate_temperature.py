import torch
import torch.nn.functional as F
from experiments.recall_safe_calibration.metrics_calibration import compute_metrics
from experiments.recall_safe_calibration.utils import resolve_device

def fit_temperature(logits, targets, *, device, max_iter, lr):
    calibration_device = resolve_device(device) if device == 'auto' else device
    logits_tensor = torch.as_tensor(logits.reshape(-1), dtype=torch.float32, device=calibration_device)
    targets_tensor = torch.as_tensor(targets.reshape(-1), dtype=torch.float32, device=calibration_device)
    log_temperature = torch.zeros((), dtype=torch.float32, device=calibration_device, requires_grad=True)
    with torch.no_grad():
        initial_loss = F.binary_cross_entropy_with_logits(logits_tensor, targets_tensor).item()
    optimizer = torch.optim.LBFGS([log_temperature], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=0.001, max=100.0)
        loss = F.binary_cross_entropy_with_logits(logits_tensor / temperature, targets_tensor)
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        temperature = torch.exp(log_temperature).clamp(min=0.001, max=100.0)
        final_loss = F.binary_cross_entropy_with_logits(logits_tensor / temperature, targets_tensor).item()
    return {'temperature': float(temperature.detach().cpu().item()), 'log_temperature': float(log_temperature.detach().cpu().item()), 'initial_nll': float(initial_loss), 'final_nll': float(final_loss)}

def temperature_payload(cache, fit, *, config_path, checkpoint, ece_bins, lesion_dilation, lr, max_iter):
    before = compute_metrics(cache['logits'], cache['targets'], threshold=0.5, temperature=1.0, ece_bins=ece_bins, lesion_dilation=lesion_dilation, include_boundary=False)
    after = compute_metrics(cache['logits'], cache['targets'], threshold=0.5, temperature=fit['temperature'], ece_bins=ece_bins, lesion_dilation=lesion_dilation, include_boundary=False)
    return {'method': 'temperature_scaling', 'objective': 'BCEWithLogitsLoss(logits / exp(log_T), target)', 'temperature': fit['temperature'], 'log_temperature': fit['log_temperature'], 'optimization': {'parameterization': 'T = exp(log_T), clamped to [1e-3, 100]', 'optimizer': 'LBFGS', 'lr': lr, 'max_iter': max_iter, 'initial_T': 1.0, 'initial_nll': fit['initial_nll'], 'final_nll': fit['final_nll']}, 'validation_cache_metadata': cache.get('metadata', {}), 'validation_metrics_at_threshold_0_5': {'before_temperature_scaling': before, 'after_temperature_scaling': after}, 'config': {'config_path': config_path, 'checkpoint': checkpoint, 'ece_bins': ece_bins, 'lesion_dilation': lesion_dilation}}
