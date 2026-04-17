"""calibration — 参数拟合、Bootstrap 置信区间、误差验证"""
from .fitting    import fit_model, FitResult
from .validation import compute_metrics, ValidationResult

__all__ = ["fit_model", "FitResult", "compute_metrics", "ValidationResult"]
