"""Preset datasets and metadata for the dashboard API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from src.simulation import build_sine_training_data

PresetName = Literal["xor", "sine"]


class PresetDefinition(BaseModel):
    """Describes a supported preset for UI and API consumers."""

    key: PresetName
    title: str
    description: str
    default_structure: list[int]
    default_learning_rate: float


def load_xor_data() -> tuple[list[list[float]], list[list[float]]]:
    """Return the XOR dataset."""
    inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    targets = [
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ]
    return inputs, targets


def load_sine_data() -> tuple[list[list[float]], list[list[float]]]:
    """Return the normalized sine regression dataset."""
    return build_sine_training_data()


PRESET_DEFINITIONS: dict[PresetName, PresetDefinition] = {
    "xor": PresetDefinition(
        key="xor",
        title="XOR",
        description="Binary classification sanity check for nonlinear separation.",
        default_structure=[2, 2, 1],
        default_learning_rate=1.0,
    ),
    "sine": PresetDefinition(
        key="sine",
        title="Sine Approximation",
        description="Small regression task using normalized sine samples from x=0..7.",
        default_structure=[1, 2, 1],
        default_learning_rate=0.275,
    ),
}

PRESET_LOADERS = {
    "xor": load_xor_data,
    "sine": load_sine_data,
}
