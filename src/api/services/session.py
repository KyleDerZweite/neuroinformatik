"""Session orchestration for the Neuroinformatik dashboard API."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable

from src.api.presets import PRESET_DEFINITIONS, PRESET_LOADERS, PresetName
from src.api.schemas import (
    ConfigureRequest,
    CurrentSampleState,
    DashboardEvent,
    ErrorEvent,
    LayerState,
    PresetSummary,
    SessionState,
    StateEvent,
    SummaryEvent,
    SummaryState,
    TopologyState,
    TrainingState,
)
from src.neuralnet import NeuralNetwork

PRECISION = 6


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, PRECISION)


def _round_list(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [_round(item) for item in values]


def _round_matrix(matrix: list[list[float]] | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return [_round_list(row) or [] for row in matrix]


class TrainingSession:
    """Stateful session around the handwritten neural network implementation."""

    def __init__(self) -> None:
        self.network: NeuralNetwork | None = None
        self.training_inputs: list[list[float]] | None = None
        self.training_targets: list[list[float]] | None = None
        self.epoch_losses: list[float] = []
        self.current_epoch = 0
        self.current_sample_index = 0
        self.current_layer_index = 0
        self.phase = "idle"
        self.mode = "inspect"
        self.preset: PresetName | None = None
        self.epoch_loss_accumulator = 0.0
        self.samples_in_current_epoch = 0
        self._last_config: tuple[list[int], float, PresetName] | None = None
        self._backward_gradient: list[float] | None = None
        self._backward_layer_index: int | None = None
        self._layer_forward_input: list[float] | None = None
        self._stop_flag = False

    def configure(self, request: ConfigureRequest) -> SessionState:
        """Configure a fresh network for the requested preset."""
        if request.preset not in PRESET_LOADERS:
            raise ValueError(f"Unsupported preset: {request.preset}")

        self._stop_flag = True
        self.network = NeuralNetwork(request.structure, request.learning_rate)
        self.training_inputs, self.training_targets = PRESET_LOADERS[request.preset]()
        self._last_config = (list(request.structure), request.learning_rate, request.preset)
        self.preset = request.preset
        self._reset_training_state()
        return self.get_state()

    def reset(self) -> SessionState:
        """Reset the current configuration with randomized parameters."""
        if self._last_config is None:
            raise ValueError("No configuration to reset. Configure a preset first.")

        structure, learning_rate, preset = self._last_config
        self._stop_flag = True
        self.network = NeuralNetwork(structure, learning_rate)
        self.preset = preset
        self._reset_training_state()
        return self.get_state()

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

    def _require_network(self) -> None:
        if self.network is None or self.training_inputs is None or self.training_targets is None:
            raise ValueError("No network configured. Configure a preset first.")

    def get_state(self) -> SessionState:
        """Return the full serializable session state."""
        network = self.network
        layers: list[LayerState] = []
        structure: list[int] = []

        if network is not None:
            for index, layer in enumerate(network.layers):
                layers.append(
                    LayerState(
                        index=index,
                        input_size=layer.input_size,
                        output_size=layer.output_size,
                        weights=_round_matrix(layer.weights),
                        biases=_round_list(layer.biases),
                        last_input=_round_list(layer.last_input),
                        last_pre_activation=_round_list(layer.last_pre_activation_values),
                        last_output=_round_list(layer.last_output),
                        last_gradient=_round_list(layer.last_gradient_from_next_layer),
                        learning_rate=layer.learning_rate,
                    )
                )

            if network.layers:
                structure.append(network.layers[0].input_size)
                for layer in network.layers:
                    structure.append(layer.output_size)

        current_sample: CurrentSampleState | None = None
        if (
            network is not None
            and self.training_inputs is not None
            and self.training_targets is not None
            and 0 <= self.current_sample_index < len(self.training_inputs)
        ):
            sample_input = self.training_inputs[self.current_sample_index]
            sample_target = self.training_targets[self.current_sample_index]
            prediction = _round_list(network.last_output) if network.last_output else None
            loss = None
            if network.last_output and self.phase == "forward_done":
                loss = _round(network.calculate_sample_loss(sample_target))

            current_sample = CurrentSampleState(
                index=self.current_sample_index,
                input=_round_list(sample_input) or [],
                target=_round_list(sample_target) or [],
                prediction=prediction,
                loss=loss,
            )

        return SessionState(
            preset=self.preset,
            topology=TopologyState(
                structure=structure,
                learning_rate=_round(network.learning_rate) if network is not None else None,
            ),
            layers=layers,
            network_output=_round_list(network.last_output) if network and network.last_output else None,
            training=TrainingState(
                current_epoch=self.current_epoch,
                current_sample_index=self.current_sample_index,
                current_layer_index=self.current_layer_index,
                phase=self.phase,
                mode=self.mode,
                epoch_losses=[_round(loss) or 0.0 for loss in self.epoch_losses],
                total_samples=len(self.training_inputs) if self.training_inputs else 0,
                total_layers=len(network.layers) if network is not None else 0,
            ),
            current_sample=current_sample,
        )

    def get_summary(self) -> SummaryState:
        """Return a lighter state payload for streaming updates."""
        state = self.get_state()
        return SummaryState(training=state.training, network_output=state.network_output)

    def step_epoch(self, count: int = 1) -> SessionState:
        self._require_network()
        assert self.network is not None
        assert self.training_inputs is not None
        assert self.training_targets is not None

        for _ in range(count):
            epoch_loss_sum = 0.0
            for sample_input, sample_target in zip(self.training_inputs, self.training_targets, strict=True):
                self.network.forward(sample_input)
                epoch_loss_sum += self.network.calculate_sample_loss(sample_target)
                self.network.backward(sample_target)

            self.epoch_losses.append(epoch_loss_sum / len(self.training_inputs))
            self.current_epoch += 1

        self.current_sample_index = 0
        self.phase = "idle"
        self.epoch_loss_accumulator = 0.0
        self.samples_in_current_epoch = 0
        self._backward_gradient = None
        self._backward_layer_index = None
        self._layer_forward_input = None
        return self.get_state()

    def step_forward(self, sample_index: int | None = None) -> SessionState:
        self._require_network()
        assert self.network is not None
        assert self.training_inputs is not None

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
        return self.get_state()

    def step_backward(self) -> SessionState:
        self._require_network()
        assert self.network is not None
        assert self.training_targets is not None

        if self.phase != "forward_done":
            raise ValueError("Cannot step backward before a forward pass completes.")

        sample_target = self.training_targets[self.current_sample_index]
        self.epoch_loss_accumulator += self.network.calculate_sample_loss(sample_target)
        self.samples_in_current_epoch += 1
        self.network.backward(sample_target)
        self.phase = "backward_done"
        self.current_sample_index += 1

        if self.training_inputs is not None and self.current_sample_index >= len(self.training_inputs):
            self._finalize_epoch()

        return self.get_state()

    def _finalize_epoch(self) -> None:
        if self.samples_in_current_epoch > 0:
            self.epoch_losses.append(self.epoch_loss_accumulator / self.samples_in_current_epoch)
            self.current_epoch += 1

        self.current_sample_index = 0
        self.epoch_loss_accumulator = 0.0
        self.samples_in_current_epoch = 0
        self.phase = "idle"

    def step_layer_forward(self) -> SessionState:
        self._require_network()
        assert self.network is not None
        assert self.training_inputs is not None

        layers = self.network.layers
        if self._layer_forward_input is None:
            if self.current_sample_index >= len(self.training_inputs):
                self.current_sample_index = 0
            self._layer_forward_input = self.training_inputs[self.current_sample_index]
            self.current_layer_index = 0

        if self.current_layer_index >= len(layers):
            return self.get_state()

        layer = layers[self.current_layer_index]
        output = layer.forward(self._layer_forward_input)
        self.current_layer_index += 1

        if self.current_layer_index >= len(layers):
            self.network.last_output = layers[-1].last_output
            self.phase = "forward_done"
            self._layer_forward_input = None
        else:
            self._layer_forward_input = output

        return self.get_state()

    def step_layer_backward(self) -> SessionState:
        self._require_network()
        assert self.network is not None
        assert self.training_targets is not None
        assert self.training_inputs is not None

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
            return self.get_state()

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

        return self.get_state()

    def begin_run(self) -> SessionState:
        """Enter run mode before background training starts."""
        self._require_network()
        self._stop_flag = False
        self.mode = "run"
        return self.get_state()

    def run_epoch(self) -> SessionState:
        """Run a single epoch in continuous mode."""
        self._require_network()
        assert self.network is not None
        assert self.training_inputs is not None
        assert self.training_targets is not None

        epoch_loss_sum = 0.0
        for sample_input, sample_target in zip(self.training_inputs, self.training_targets, strict=True):
            self.network.forward(sample_input)
            epoch_loss_sum += self.network.calculate_sample_loss(sample_target)
            self.network.backward(sample_target)

        self.epoch_losses.append(epoch_loss_sum / len(self.training_inputs))
        self.current_epoch += 1
        return self.get_state()

    def pause(self) -> None:
        self._stop_flag = True

    def should_stop(self) -> bool:
        return self._stop_flag

    def finish_run(self) -> SessionState:
        self.mode = "inspect"
        self.phase = "idle"
        self.current_sample_index = 0
        self.current_layer_index = 0
        self.epoch_loss_accumulator = 0.0
        self.samples_in_current_epoch = 0
        self._backward_gradient = None
        self._backward_layer_index = None
        self._layer_forward_input = None
        return self.get_state()


class SessionService:
    """Thread-safe orchestration layer for HTTP and WebSocket consumers."""

    def __init__(self) -> None:
        self._session = TrainingSession()
        self._lock = asyncio.Lock()
        self._listeners: set[asyncio.Queue[DashboardEvent]] = set()
        self._run_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        """Boot the service with a default XOR session."""
        async with self._lock:
            request = ConfigureRequest(
                structure=PRESET_DEFINITIONS["xor"].default_structure,
                learning_rate=PRESET_DEFINITIONS["xor"].default_learning_rate,
                preset="xor",
            )
            self._session.configure(request)

    async def shutdown(self) -> None:
        """Stop background work during application shutdown."""
        await self.pause()

    async def list_presets(self) -> list[PresetSummary]:
        """Return API-friendly preset metadata."""
        presets: list[PresetSummary] = []
        for key, definition in PRESET_DEFINITIONS.items():
            inputs, targets = PRESET_LOADERS[key]()
            presets.append(
                PresetSummary(
                    key=definition.key,
                    title=definition.title,
                    description=definition.description,
                    default_structure=definition.default_structure,
                    default_learning_rate=definition.default_learning_rate,
                    sample_count=len(inputs),
                    input_size=len(inputs[0]) if inputs else 0,
                    output_size=len(targets[0]) if targets else 0,
                )
            )
        return presets

    async def get_state(self) -> SessionState:
        async with self._lock:
            return self._session.get_state()

    async def configure(self, request: ConfigureRequest) -> SessionState:
        await self.pause()
        async with self._lock:
            state = self._session.configure(request)
        await self._broadcast(StateEvent(data=state))
        return state

    async def reset(self) -> SessionState:
        await self.pause()
        async with self._lock:
            state = self._session.reset()
        await self._broadcast(StateEvent(data=state))
        return state

    async def step_epoch(self, count: int) -> SessionState:
        await self.pause()
        async with self._lock:
            state = self._session.step_epoch(count)
        await self._broadcast(StateEvent(data=state))
        return state

    async def step_forward(self, sample_index: int | None) -> SessionState:
        await self.pause()
        async with self._lock:
            state = self._session.step_forward(sample_index)
        await self._broadcast(StateEvent(data=state))
        return state

    async def step_backward(self) -> SessionState:
        await self.pause()
        async with self._lock:
            state = self._session.step_backward()
        await self._broadcast(StateEvent(data=state))
        return state

    async def step_layer_forward(self) -> SessionState:
        await self.pause()
        async with self._lock:
            state = self._session.step_layer_forward()
        await self._broadcast(StateEvent(data=state))
        return state

    async def step_layer_backward(self) -> SessionState:
        await self.pause()
        async with self._lock:
            state = self._session.step_layer_backward()
        await self._broadcast(StateEvent(data=state))
        return state

    async def start_run(self, speed_ms: int) -> SessionState:
        await self.pause()
        async with self._lock:
            state = self._session.begin_run()
            self._run_task = asyncio.create_task(self._run_loop(speed_ms))
        await self._broadcast(StateEvent(data=state))
        return state

    async def pause(self) -> SessionState:
        task: asyncio.Task[None] | None
        async with self._lock:
            self._session.pause()
            task = self._run_task
            self._run_task = None

        if task is not None:
            await task

        async with self._lock:
            state = self._session.get_state()
        return state

    async def subscribe(self) -> asyncio.Queue[DashboardEvent]:
        """Register a new WebSocket subscriber and seed it with current state."""
        queue: asyncio.Queue[DashboardEvent] = asyncio.Queue()
        async with self._lock:
            self._listeners.add(queue)
            await queue.put(StateEvent(data=self._session.get_state()))
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[DashboardEvent]) -> None:
        async with self._lock:
            self._listeners.discard(queue)

    async def _run_loop(self, speed_ms: int) -> None:
        epoch_count = 0
        try:
            while True:
                async with self._lock:
                    if self._session.should_stop():
                        break
                    state = self._session.run_epoch()
                    summary = self._session.get_summary()

                epoch_count += 1
                if epoch_count % 10 == 0:
                    await self._broadcast(StateEvent(data=state))
                else:
                    await self._broadcast(SummaryEvent(data=summary))

                await asyncio.sleep(speed_ms / 1000.0)
        except Exception as exc:  # pragma: no cover - defensive broadcast path
            await self._broadcast(ErrorEvent(message=str(exc)))
            raise
        finally:
            async with self._lock:
                final_state = self._session.finish_run()
                self._run_task = None
            await self._broadcast(StateEvent(data=final_state))

    async def _broadcast(self, event: DashboardEvent) -> None:
        listeners = list(self._listeners)
        if not listeners:
            return

        stale_queues: list[asyncio.Queue[DashboardEvent]] = []
        for queue in listeners:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - default queues are unbounded
                stale_queues.append(queue)

        if stale_queues:
            async with self._lock:
                for queue in stale_queues:
                    self._listeners.discard(queue)
