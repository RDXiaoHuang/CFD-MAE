"""
Ultralytics YOLO Wrappers (YOLO11 / YOLO26)
Supports all variants: n, s, m, l, x
"""
from .yolo11_wrapper import YOLO11Wrapper, yolo_v11_ultralytics
from .yolo26_wrapper import YOLO26Wrapper, yolo_v26

__all__ = [
    'YOLO11Wrapper', 'yolo_v11_ultralytics',
    'YOLO26Wrapper', 'yolo_v26'
]
