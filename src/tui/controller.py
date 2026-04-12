"""Stateful controller for the Neuroinformatik terminal UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.neuralnet import NeuralNetwork
from src.simulation import build_sine_training_data

PresetName = Literal["xor", "sine"]
ModeName = Literal["inspect", "run"]


@dataclass(frozen=True)
class PresetConfig:
    key: PresetName
    title: str
    description: str
    default_structure: list[int]
    default_learning_rate: float


@dataclass(frozen=True)
class LayerSnapshot:
    index: int
    input_size: int
    output_size: int
    weights: list[list[float]] | None
    biases: list[float] | None
    last_input: list[float] | None
    last_pre_activation: list[float] | None
    last_output: list[float] | None
    last_gradient: list[float] | None


@dataclass(frozen=True)
class SampleSnapshot:
    index: int
    input: list[float]
    target: list[float]
    prediction: list[float] | None
    loss: float | None


@dataclass(frozen=True)
class SessionSnapshot:
    preset: PresetName | None
    structure: list[int]
    learning_rate: float | None
    current_epoch: int
    current_sample_index: int
    current_layer_index: int
    phase: str
    mode: ModeName
    epoch_losses: list[float]
    layers: list[LayerSnapshot]
    current_sample: SampleSnapshot | None
    total_samples: int
    total_layers: int


PRESETS: dict[PresetName, PresetConfig] = {
    "xor": PresetConfig(
        key="xor",
        title="XOR",
        description="Binary classification sanity check for nonlinear separation.",
        default_structure=[2, 2, 1],
        default_learning_rate=1.0,
    ),
    "sine": PresetConfig(
        key="sine",
        title="Sine Approximation",
        description="Small regression task using normalized sine samples from x = 0..7.",
        default_structure=[1, 2, 1],
        default_learning_rate=0.275,
    ),
}


def load_xor_data() -> tuple[list[list[float]], list[list[float]]]:
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
    return build_sine_training_data()


PRESET_LOADERS = {
    "xor": load_xor_data,
    "sine": load_sine_data,
}


class TuiController:
    """Wrap the handwritten network with inspectable training session state."""

    def __init__(self) -> None:
        self.network: NeuralNetwork | None = None
        self.training_inputs: list[list[float]] | None = None
        self.training_targets: list[list[float]] | None = None
        self.preset: PresetName | None = None
        self.epoch_losses: list[float] = []
        self.current_epoch = 0
        self.current_sample_index = 0
        self.current_layer_index = 0
        self.phase = "idle"
        self.mode: ModeName = "inspect"
        self.epoch_loss_accumulator = 0.0
        self.samples_in_current_epoch = 0
        self._last_config: tuple[list[int], float, PresetName] | None = None
        self._backward_gradient: list[float] | None = None
        self._backward_layer_index: int | None = None
        self._layer_forward_input: list[float] | None = None

    def configure(self, structure: list[int], learning_rate: float, preset: PresetName) -> SessionSnapshot:
        if preset not in PRESET_LOADERS:
            raise ValueError(f"Unsupported preset: {preset}")

        self.network = NeuralNetwork(structure, learning_rate)
        self.training_inputs, self.training_targets = PRESET_LOADERS[preset]()
        self.preset = preset
        self._last_config = (list(structure), learning_rate, preset)
        self._reset_training_state()
        return self.snapshot()

    def reset(self) -> SessionSnapshot:
        if self._last_config is None:
            raise ValueError("No configuration to reset. Configure a preset first.")

        structure, learning_rate, preset = self._last_config
        self.network = NeuralNetwork(structure, learning_rate)
        self.preset = preset
        self._reset_training_state()
        return self.snapshot()

    def start_run(self) -> SessionSnapshot:
        self._require_network()
        self.mode = "run"
        return self.snapshot()

    def pause_run(self) -> SessionSnapshot:
        self.mode = "inspect"
        return self.snapshot()

    def run_epoch(self) -> SessionSnapshot:
        self._require_network()
        self.step_epoch(1)
        self.mode = "run"
        return self.snapshot()

    def run_epochs(self, count: int) -> SessionSnapshot:
        self._require_network()
        if count <= 0:
            raise ValueError("Epoch batch must be a positive integer.")

        self.step_epoch(count)
        self.mode = "run"
        return self.snapshot()

    def step_epoch(self, count: int = 1) -> SessionSnapshot:
        self._require_network()
        assert self.network is not None
        assert self.training_inputs is not None
        assert self.training_targets is not None

        self.mode = "inspect"
        for _ in range(count):
            epoch_loss_sum = 0.0
            for sample_input, sample_target in zip(self.training_inputs, self.training_targets, strict=True):
                self.network.forward(sample_input)
                epoch_loss_sum += self.network.calculate_sample_loss(sample_target)
                self.network.backward(sample_target)

            self.epoch_losses.append(epoch_loss_sum / len(self.training_inputs))
            self.current_epoch += 1

        self.current_sample_index = 0
        self.current_layer_index = 0
        self.phase = "idle"
        self.epoch_loss_accumulator = 0.0
        self.samples_in_current_epoch = 0
        self._backward_gradient = None
        self._backward_layer_index = None
        self._layer_forward_input = None
        return self.snapshot()

    def step_forward(self, sample_index: int | None = None) -> SessionSnapshot:
        self._require_network()
        assert self.network is not None
        assert self.training_inputs is not None

        self.mode = "inspect"
        if sample_index is not None:
            self.current_sample_index = sample_index

        if self.current_sample_index >= len(self.training_inputs):
            self.current_sample_index = 0

        self.network.forward(self.training_inputs[self.current_sample_index])
        self.phase = "forward_done"
        self.current_layer_index = 0
        self._backward_gradient = None
        self._backward_layer_index = None
        self._layer_forward_input = None
        return self.snapshot()

    def step_backward(self) -> SessionSnapshot:
        self._require_network()
        assert self.network is not None
        assert self.training_inputs is not None
        assert self.training_targets is not None

        self.mode = "inspect"
        if self.phase != "forward_done":
            raise ValueError("Cannot step backward before a forward pass completes.")

        sample_target = self.training_targets[self.current_sample_index]
        self.epoch_loss_accumulator += self.network.calculate_sample_loss(sample_target)
        self.samples_in_current_epoch += 1
        self.network.backward(sample_target)
        self.phase = "backward_done"
        self.current_sample_index += 1

        if self.current_sample_index >= len(self.training_inputs):
            self._finalize_epoch()

        return self.snapshot()

    def step_layer_forward(self) -> SessionSnapshot:
        self._require_network()
        assert self.network is not None
        assert self.training_inputs is not None

        self.mode = "inspect"
        layers = self.network.layers
        if self._layer_forward_input is None:
            if self.current_sample_index >= len(self.training_inputs):
                self.current_sample_index = 0
            self._layer_forward_input = self.training_inputs[self.current_sample_index]
            self.current_layer_index = 0

        if self.current_layer_index >= len(layers):
            return self.snapshot()

        layer = layers[self.current_layer_index]
        output = layer.forward(self._layer_forward_input)
        self.current_layer_index += 1

        if self.current_layer_index >= len(layers):
            self.network.last_output = layers[-1].last_output
            self.phase = "forward_done"
            self._layer_forward_input = None
        else:
            self._layer_forward_input = output

        return self.snapshot()

    def step_layer_backward(self) -> SessionSnapshot:
        self._require_network()
        assert self.network is not None
        assert self.training_inputs is not None
        assert self.training_targets is not None

        self.mode = "inspect"
        if self.phase != "forward_done" and self._backward_gradient is None:
            raise ValueError("Cannot step layer backward before a forward pass completes.")

        layers = self.network.layers
        if self._backward_gradient is None:
            sample_target = self.training_targets[self.current_sample_index]
            self.epoch_loss_accumulator += self.network.calculate_sample_loss(sample_target)
            self.samples_in_current_epoch += 1
            self._backward_gradient = [
                self.network.last_output[index] - sample_target[index]
                for index in range(len(sample_target))
            ]
            self._backward_layer_index = len(layers) - 1

        assert self._backward_layer_index is not None
        if self._backward_layer_index < 0:
            return self.snapshot()

        layer = layers[self._backward_layer_index]
        self._backward_gradient = layer.backward(self._backward_gradient)
        self.current_layer_index = self._backward_layer_index
        self._backward_layer_index -= 1

        if self._backward_layer_index < 0:
            self.phase = "backward_done"
            self._backward_gradient = None
            self.current_sample_index += 1
            if self.current_sample_index >= len(self.training_inputs):
                self._finalize_epoch()

        return self.snapshot()

    def snapshot(self) -> SessionSnapshot:
        self._require_network()
        assert self.network is not None
        assert self.training_inputs is not None
        assert self.training_targets is not None

        layers = [
            LayerSnapshot(
                index=index,
                input_size=layer.input_size,
                output_size=layer.output_size,
                weights=_copy_matrix(layer.weights),
                biases=_copy_list(layer.biases),
                last_input=_copy_list(layer.last_input),
                last_pre_activation=_copy_list(layer.last_pre_activation_values),
                last_output=_copy_list(layer.last_output),
                last_gradient=_copy_list(layer.last_gradient_from_next_layer),
            )
            for index, layer in enumerate(self.network.layers)
        ]

        current_sample = None
        if 0 <= self.current_sample_index < len(self.training_inputs):
            sample_input = self.training_inputs[self.current_sample_index]
            sample_target = self.training_targets[self.current_sample_index]
            prediction = _copy_list(self.network.last_output) if self.network.last_output else None
            loss = None
            if self.network.last_output and self.phase == "forward_done":
                loss = self.network.calculate_sample_loss(sample_target)
            current_sample = SampleSnapshot(
                index=self.current_sample_index,
                input=_copy_list(sample_input) or [],
                target=_copy_list(sample_target) or [],
                prediction=prediction,
                loss=loss,
            )

        structure = [self.network.layers[0].input_size]
        structure.extend(layer.output_size for layer in self.network.layers)
        return SessionSnapshot(
            preset=self.preset,
            structure=structure,
            learning_rate=self.network.learning_rate,
            current_epoch=self.current_epoch,
            current_sample_index=self.current_sample_index,
            current_layer_index=self.current_layer_index,
            phase=self.phase,
            mode=self.mode,
            epoch_losses=[float(loss_value) for loss_value in self.epoch_losses],
            layers=layers,
            current_sample=current_sample,
            total_samples=len(self.training_inputs),
            total_layers=len(self.network.layers),
        )

    def _reset_training_state(self) -> None:
        self.epoch_losses = []
        self.current_epoch = 0
        self.current_sample_index = 0
        self.current_layer_index = 0
        self.phase = "idle"
        self.mode = "inspect"
        self.epoch_loss_accumulator = 0.0
        self.samples_in_current_epoch = 0
        self._backward_gradient = None
        self._backward_layer_index = None
        self._layer_forward_input = None

    def _finalize_epoch(self) -> None:
        if self.samples_in_current_epoch > 0:
            self.epoch_losses.append(self.epoch_loss_accumulator / self.samples_in_current_epoch)
            self.current_epoch += 1
        self.current_sample_index = 0
        self.current_layer_index = 0
        self.epoch_loss_accumulator = 0.0
        self.samples_in_current_epoch = 0
        self.phase = "idle"

    def _require_network(self) -> None:
        if self.network is None or self.training_inputs is None or self.training_targets is None:
            raise ValueError("No network configured. Configure a preset first.")


def _copy_list(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [float(value) for value in values]


def _copy_matrix(matrix: list[list[float]] | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return [_copy_list(row) or [] for row in matrix]
