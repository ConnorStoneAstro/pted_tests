import importlib.util
import numpy as np
from dataclasses import dataclass


def _grab_config(path: str):
    spec = importlib.util.spec_from_file_location("config", path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)  # type: ignore
    return config_module.CONFIG


@dataclass(frozen=True)
class TwoSampleProblem:
    x: np.ndarray
    y: np.ndarray
    severity: float
    deviation: str


def _annotate(records: list[dict[str, object]], **extra: object) -> None:
    for record in records:
        record.update(extra)
