"""
General-purpose helper utilities.
"""

import sys
import platform


def load_class_names(class_mapping: dict | None = None) -> list[str]:
    """Return ordered list of class names.

    Parameters
    ----------
    class_mapping : dict, optional
        A mapping of ``{int_label: str_name}``. When *None*, defaults to
        ``{0: "Male", 1: "Female"}``.
    """
    if class_mapping is None:
        return ["Male", "Female"]
    return [class_mapping[i] for i in sorted(class_mapping.keys())]


def get_device_info() -> str:
    """Return a human-readable summary of the compute environment."""
    info = {
        "Python": sys.version.split()[0],
        "Platform": platform.platform(),
        "CPU cores": __import__("os").cpu_count(),
    }

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        info["TensorFlow"] = tf.__version__
        info["GPU available"] = len(gpus) > 0
        if gpus:
            info["GPU"] = gpus[0].name
    except ImportError:
        info["TensorFlow"] = "not installed"
        info["GPU available"] = False

    parts = [f"{k}: {v}" for k, v in info.items()]
    return " | ".join(parts)
