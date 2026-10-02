"""BIURI shared palette and spacing — single source for shell styling."""

from __future__ import annotations


def prefers_reduced_motion() -> bool:
    """Respect OS 'reduce animations' when detectable (Windows client-area flag)."""
    try:
        import sys

        if sys.platform != "win32":
            return False
        import ctypes

        enabled = ctypes.c_int(1)
        # SPI_GETCLIENTAREAANIMATION = 0x1042
        if ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0):
            return enabled.value == 0
    except Exception:
        return False
    return False


# Brand
BRAND = "#9c27b0"
BRAND_DARK = "#7b1fa2"
BRAND_LIGHT = "#ba68c8"
ACCENT = "#e91e63"

# Surfaces
SURFACE = "#f5f5f5"
SURFACE_RAISED = "#ffffff"
SURFACE_SUBTLE = "#faf7fc"
BORDER = "#e0e0e0"
BORDER_SOFT = "#e8d5ef"

# Text (tuned for ≥4.5:1 on light surfaces)
TEXT = "#212121"
TEXT_SECONDARY = "#5b2a6d"  # brand-tinted, not gray #666
TEXT_ON_BRAND = "#ffffff"
TEXT_ON_BRAND_MUTED = "rgba(255, 255, 255, 0.92)"
TEXT_DISABLED = "#6d6d6d"

# Semantic action colors
ACTION_LOAD = "#4caf50"
ACTION_TRAIN = "#ff9800"
ACTION_EXPLAIN = BRAND
ACTION_METRICS = "#f44336"
ACTION_NL = "#00bcd4"
ACTION_CF = "#673ab7"
ACTION_IMPROVE = "#3f51b5"
ACTION_EXPORT = "#795548"

# Hover / pressed companions for ActionButton
ACTION_HOVER = {
    ACTION_LOAD: "#66bb6a",
    ACTION_TRAIN: "#ffb74d",
    ACTION_EXPLAIN: BRAND_LIGHT,
    ACTION_METRICS: "#ef5350",
    ACTION_NL: "#26c6da",
    ACTION_CF: "#7e57c2",
    ACTION_IMPROVE: "#5c6bc0",
    ACTION_EXPORT: "#8d6e63",
    "#2196f3": "#42a5f5",
}
ACTION_PRESSED = {
    ACTION_LOAD: "#388e3c",
    ACTION_TRAIN: "#f57c00",
    ACTION_EXPLAIN: BRAND_DARK,
    ACTION_METRICS: "#d32f2f",
    ACTION_NL: "#0097a7",
    ACTION_CF: "#512da8",
    ACTION_IMPROVE: "#303f9f",
    ACTION_EXPORT: "#5d4037",
    "#2196f3": "#1976d2",
}

SURFACE_CONTROL = "#f8f9fa"
BORDER_CONTROL = "#dee2e6"
DISABLED_FILL = "#e0e0e0"

# Spacing scale (px)
SPACE_1 = 4
SPACE_2 = 8
SPACE_3 = 12
SPACE_4 = 16
SPACE_5 = 24

RADIUS = 8
FOCUS_RING = f"2px solid {BRAND_DARK}"


def lighten_action(color: str) -> str:
    return ACTION_HOVER.get(color, color)


def darken_action(color: str) -> str:
    return ACTION_PRESSED.get(color, color)

APP_QSS = f"""
QMainWindow {{
    background-color: {SURFACE};
}}
QWidget {{
    font-family: 'Segoe UI', 'Segoe UI Symbol', sans-serif;
}}
QPushButton:focus, QToolButton:focus, QComboBox:focus,
QCheckBox:focus, QSlider:focus, QDoubleSpinBox:focus, QTabBar::tab:focus {{
    outline: none;
    border: {FOCUS_RING};
}}
QLineEdit:focus, QTextEdit:focus {{
    border: {FOCUS_RING};
}}
QStatusBar {{
    background-color: {SURFACE_SUBTLE};
    color: {TEXT_SECONDARY};
    border-top: 1px solid {BORDER_SOFT};
    padding: 4px 10px;
    font-size: 12px;
}}
"""
