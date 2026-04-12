"""Pydantic models for the Neuroinformatik API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.api.presets import PresetName

PositiveIntList = Annotated[list[int], Field(min_length=2)]


class TopologyState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure: list[int]
    learning_rate: float | None


class LayerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    input_size: int
    output_size: int
    weights: list[list[float]] | None
    biases: list[float] | None
    last_input: list[float] | None
    last_pre_activation: list[float] | None
    last_output: list[float] | None
    last_gradient: list[float] | None
    learning_rate: float


class CurrentSampleState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    input: list[float]
    target: list[float]
    prediction: list[float] | None
    loss: float | None


class TrainingState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_epoch: int
    current_sample_index: int
    current_layer_index: int
    phase: str
    mode: str
    epoch_losses: list[float]
    total_samples: int
    total_layers: int


class SessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: PresetName | None
    topology: TopologyState
    layers: list[LayerState]
    network_output: list[float] | None
    training: TrainingState
    current_sample: CurrentSampleState | None


class SummaryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    training: TrainingState
    network_output: list[float] | None


class PresetSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: PresetName
    title: str
    description: str
    default_structure: list[int]
    default_learning_rate: float
    sample_count: int
    input_size: int
    output_size: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]


class ConfigureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure: PositiveIntList
    learning_rate: Annotated[float, Field(gt=0.0)]
    preset: PresetName


class StepEpochRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: Annotated[int, Field(ge=1, le=10000)] = 1


class StepForwardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_index: int | None = Field(default=None, ge=0)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speed_ms: Annotated[int, Field(ge=10, le=5000)] = 50


class StateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["state"] = "state"
    data: SessionState


class SummaryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["summary"] = "summary"
    data: SummaryState


class ErrorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    message: str


DashboardEvent = StateEvent | SummaryEvent | ErrorEvent
