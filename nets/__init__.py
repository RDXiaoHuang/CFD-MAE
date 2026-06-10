"""
TSDA-Debias Neural Network Modules

This package contains two implementations:
- custom: Custom PyTorch YOLO11 implementation
- ultralytics: Ultralytics YOLO11/YOLO26 wrappers (supports n/s/m/l/x variants)
"""

# Import nn module for weight loading compatibility
from .custom import nn

# Custom PyTorch YOLO11 implementation
from .custom import (
    YOLO, yolo_v11_n, yolo_v11_s, yolo_v11_m, yolo_v11_l, yolo_v11_x,
)

# Ultralytics YOLO wrappers
from .ultralytics import (
    YOLO11Wrapper, yolo_v11_ultralytics,
    YOLO26Wrapper, yolo_v26,
)

__all__ = [
    # nn module (for weight loading)
    'nn',
    # Custom PyTorch
    'YOLO', 'yolo_v11_n', 'yolo_v11_s', 'yolo_v11_m', 'yolo_v11_l', 'yolo_v11_x',
    # Ultralytics
    'YOLO11Wrapper', 'yolo_v11_ultralytics',
    'YOLO26Wrapper', 'yolo_v26',
]
