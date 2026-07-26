"""FCD 波场分析系统 - 后端计算引擎包。

提供 FCD 解调、Sylvester 积分、时频域分析、定标及交互式 UI 选择器。
"""

from backend.core import FCDCore, sine_fit_func
from backend.ui_selectors import MasterCircleSelector, MasterLineSelector, InteractiveMeasurer

__all__ = [
    'FCDCore',
    'sine_fit_func',
    'MasterCircleSelector',
    'MasterLineSelector',
    'InteractiveMeasurer',
]
