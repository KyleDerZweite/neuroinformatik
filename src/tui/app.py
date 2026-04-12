"""Textual terminal UI for the Neuroinformatik project."""

from __future__ import annotations

from typing import cast

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import Button, Footer, Header, Input, Log, Markdown, Select, Sparkline, Static, TabPane, TabbedContent

from src.tui.controller import PRESETS, PresetName, SessionSnapshot, TuiController
from src.tui.help_text import HELP_MARKDOWN


def format_number(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def format_vector(values: list[float] | None, digits: int = 4) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(f"{value:.{digits}f}" for value in values) + "]"


def format_matrix(matrix: list[list[float]] | None, digits: int = 4) -> str:
    if not matrix:
        return "[]"
    return "\n".join(format_vector(row, digits=digits) for row in matrix)


class NeuroTuiApp(App[None]):
    """Single-process TUI for training and inspection."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        height: 1fr;
    }

    #sidebar {
        width: 34;
        min-width: 34;
        padding: 1 1;
        border: round $primary;
        background: $surface;
    }

    #content {
        width: 1fr;
        padding-left: 1;
    }

    .section-title {
        margin-top: 1;
        text-style: bold;
        color: $text-muted;
    }

    .field-label {
        margin-top: 1;
        color: $text-muted;
    }

    .button-grid {
        layout: grid;
        grid-size: 2 5;
        grid-gutter: 1 1;
        margin-top: 1;
    }

    Button {
        width: 1fr;
    }

    #stats {
        height: auto;
        border: round $primary;
        padding: 1 2;
        margin-bottom: 1;
        background: $panel;
    }

    #loss-panel,
    #sample-panel,
    #layers-panel,
    #log-panel {
        border: round $accent;
        padding: 1 2;
        background: $panel;
        margin-top: 1;
    }

    #loss-curve {
        width: 100%;
        height: 4;
        margin-top: 1;
    }

    #sample-panel,
    #layers-panel,
    #help-panel,
    #log-panel {
        height: 1fr;
    }

    Log {
        height: 1fr;
    }

    Input,
    Select {
        width: 100%;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "run_training", "Run"),
        ("p", "pause_training", "Pause"),
        ("e", "step_epoch", "Epoch"),
        ("f", "step_forward", "Forward"),
        ("b", "step_backward", "Backward"),
        ("l", "step_layer_forward", "Layer +"),
        ("k", "step_layer_backward", "Layer -"),
        ("c", "configure_session", "Configure"),
        ("x", "reset_session", "Reset"),
    ]

    run_timer: Timer

    def __init__(self) -> None:
        super().__init__()
        self.controller = TuiController()
        default = PRESETS["xor"]
        self.controller.configure(default.default_structure, default.default_learning_rate, default.key)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with VerticalScroll(id="sidebar"):
                yield Static("Configuration", classes="section-title")
                yield Static("Preset", classes="field-label")
                yield Select(
                    [(config.title, key) for key, config in PRESETS.items()],
                    value="xor",
                    id="preset-select",
                    allow_blank=False,
                )
                yield Static("Structure", classes="field-label")
                yield Input("2, 2, 1", id="structure-input")
                yield Static("Learning rate", classes="field-label")
                yield Input("1.0", id="learning-rate-input", type="number")
                yield Static("Run interval (ms)", classes="field-label")
                yield Input("80", id="speed-input", type="integer")
                yield Static("Epoch batch", classes="field-label")
                yield Input("1", id="epoch-count-input", type="integer")

                yield Static("Actions", classes="section-title")
                with Vertical(classes="button-grid"):
                    yield Button("Configure", id="configure-button", variant="primary")
                    yield Button("Reset", id="reset-button")
                    yield Button("Run", id="run-button", variant="success")
                    yield Button("Pause", id="pause-button", variant="warning")
                    yield Button("Epoch", id="step-epoch-button")
                    yield Button("Forward", id="step-forward-button")
                    yield Button("Backward", id="step-backward-button")
                    yield Button("Layer +", id="step-layer-forward-button")
                    yield Button("Layer -", id="step-layer-backward-button")

            with Vertical(id="content"):
                yield Static(id="stats")
                with TabbedContent(initial="overview"):
                    with TabPane("Overview", id="overview"):
                        with Vertical(id="loss-panel"):
                            yield Static(id="loss-summary")
                            yield Sparkline([], id="loss-curve")
                        yield Static(id="sample-panel")
                    with TabPane("Inspect", id="inspect"):
                        yield Static(id="layers-panel")
                    with TabPane("Help", id="help"):
                        yield Markdown(HELP_MARKDOWN, id="help-panel")
                with Vertical(id="log-panel"):
                    yield Static("Event log")
                    yield Log(id="event-log", auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        self.run_timer = self.set_interval(0.08, self._tick_training, pause=True)
        self.refresh_all()
        self._log("TUI initialized with XOR preset.")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "preset-select":
            return
        preset_key = cast(PresetName, event.value)
        preset = PRESETS[preset_key]
        self.query_one("#structure-input", Input).value = ", ".join(str(item) for item in preset.default_structure)
        self.query_one("#learning-rate-input", Input).value = str(preset.default_learning_rate)
        self._log(f"Loaded preset defaults for {preset.title}.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        handlers = {
            "configure-button": self.action_configure_session,
            "reset-button": self.action_reset_session,
            "run-button": self.action_run_training,
            "pause-button": self.action_pause_training,
            "step-epoch-button": self.action_step_epoch,
            "step-forward-button": self.action_step_forward,
            "step-backward-button": self.action_step_backward,
            "step-layer-forward-button": self.action_step_layer_forward,
            "step-layer-backward-button": self.action_step_layer_backward,
        }
        handler = handlers.get(event.button.id)
        if handler is not None:
            handler()

    def action_configure_session(self) -> None:
        try:
            structure = self._parse_structure()
            learning_rate = self._parse_learning_rate()
            preset = self._current_preset()
            self.run_timer.pause()
            snapshot = self.controller.configure(structure, learning_rate, preset)
            self.refresh_all(snapshot)
            self._log(f"Configured {preset} with structure {structure} and lr={learning_rate:.4f}.")
        except ValueError as exc:
            self._log(f"Configure failed: {exc}")

    def action_reset_session(self) -> None:
        try:
            self.run_timer.pause()
            snapshot = self.controller.reset()
            self.refresh_all(snapshot)
            self._log("Reset network with the last configuration.")
        except ValueError as exc:
            self._log(f"Reset failed: {exc}")

    def action_run_training(self) -> None:
        try:
            speed_ms = self._parse_speed_ms(default=80)
            epoch_batch = self._parse_epoch_batch()
            self.controller.start_run()
            self.run_timer.interval = speed_ms / 1000.0
            self.run_timer.resume()
            self.refresh_all()
            self._log(
                f"Continuous training started at {speed_ms} ms per update with batch size {epoch_batch}."
            )
        except ValueError as exc:
            self._log(f"Run failed: {exc}")

    def action_pause_training(self) -> None:
        self.run_timer.pause()
        self.controller.pause_run()
        self.refresh_all()
        self._log("Continuous training paused.")

    def action_step_epoch(self) -> None:
        try:
            count = self._parse_epoch_batch()
            self.run_timer.pause()
            snapshot = self.controller.step_epoch(count)
            self.refresh_all(snapshot)
            self._log(f"Stepped {count} epoch(s).")
        except ValueError as exc:
            self._log(f"Epoch step failed: {exc}")

    def action_step_forward(self) -> None:
        try:
            self.run_timer.pause()
            snapshot = self.controller.step_forward()
            self.refresh_all(snapshot)
            self._log("Stepped one forward pass.")
        except ValueError as exc:
            self._log(f"Forward step failed: {exc}")

    def action_step_backward(self) -> None:
        try:
            self.run_timer.pause()
            snapshot = self.controller.step_backward()
            self.refresh_all(snapshot)
            self._log("Stepped one backward pass.")
        except ValueError as exc:
            self._log(f"Backward step failed: {exc}")

    def action_step_layer_forward(self) -> None:
        try:
            self.run_timer.pause()
            snapshot = self.controller.step_layer_forward()
            self.refresh_all(snapshot)
            self._log("Stepped one layer forward.")
        except ValueError as exc:
            self._log(f"Layer forward failed: {exc}")

    def action_step_layer_backward(self) -> None:
        try:
            self.run_timer.pause()
            snapshot = self.controller.step_layer_backward()
            self.refresh_all(snapshot)
            self._log("Stepped one layer backward.")
        except ValueError as exc:
            self._log(f"Layer backward failed: {exc}")

    def refresh_all(self, snapshot: SessionSnapshot | None = None) -> None:
        snapshot = snapshot or self.controller.snapshot()
        self._update_stats(snapshot)
        self._update_loss(snapshot)
        self._update_sample(snapshot)
        self._update_layers(snapshot)

    def _tick_training(self) -> None:
        try:
            epoch_batch = self._parse_epoch_batch()
        except ValueError as exc:
            self.run_timer.pause()
            self.controller.pause_run()
            self.refresh_all()
            self._log(f"Continuous training paused: {exc}")
            return

        snapshot = self.controller.run_epochs(epoch_batch)
        self.refresh_all(snapshot)
        latest_loss = snapshot.epoch_losses[-1] if snapshot.epoch_losses else None
        self._log(
            f"Epoch {snapshot.current_epoch} complete after batch {epoch_batch}. "
            f"Loss={format_number(latest_loss)}"
        )

    def _update_stats(self, snapshot: SessionSnapshot) -> None:
        latest_loss = snapshot.epoch_losses[-1] if snapshot.epoch_losses else None
        stats = (
            f"Preset: {snapshot.preset or '—'}\n"
            f"Structure: {' → '.join(str(item) for item in snapshot.structure)}\n"
            f"Learning rate: {format_number(snapshot.learning_rate, digits=4)}\n"
            f"Epoch: {snapshot.current_epoch}\n"
            f"Sample: {min(snapshot.current_sample_index + 1, snapshot.total_samples)}/{snapshot.total_samples}\n"
            f"Layer: {snapshot.current_layer_index + 1 if snapshot.total_layers else 0}/{snapshot.total_layers}\n"
            f"Phase: {snapshot.phase}\n"
            f"Mode: {snapshot.mode}\n"
            f"Latest loss: {format_number(latest_loss)}"
        )
        self.query_one("#stats", Static).update(stats)

    def _update_loss(self, snapshot: SessionSnapshot) -> None:
        self.query_one("#loss-curve", Sparkline).data = snapshot.epoch_losses or [0.0]
        latest_loss = snapshot.epoch_losses[-1] if snapshot.epoch_losses else None
        best_loss = min(snapshot.epoch_losses) if snapshot.epoch_losses else None
        summary = (
            f"Loss history: {len(snapshot.epoch_losses)} epoch(s)\n"
            f"Latest: {format_number(latest_loss)}\n"
            f"Best: {format_number(best_loss)}"
        )
        self.query_one("#loss-summary", Static).update(summary)

    def _update_sample(self, snapshot: SessionSnapshot) -> None:
        sample = snapshot.current_sample
        if sample is None:
            content = "No sample selected."
        else:
            content = "\n".join(
                [
                    f"Current sample index: {sample.index}",
                    f"Input:      {format_vector(sample.input)}",
                    f"Target:     {format_vector(sample.target)}",
                    f"Prediction: {format_vector(sample.prediction)}",
                    f"Loss:       {format_number(sample.loss)}",
                ]
            )
        self.query_one("#sample-panel", Static).update(content)

    def _update_layers(self, snapshot: SessionSnapshot) -> None:
        if not snapshot.layers:
            self.query_one("#layers-panel", Static).update("No layers configured.")
            return

        sections: list[str] = []
        for layer in snapshot.layers:
            sections.extend(
                [
                    f"Layer {layer.index + 1}: {layer.input_size} -> {layer.output_size}",
                    f"  Weights:\n{_indent_block(format_matrix(layer.weights), '    ')}",
                    f"  Biases: {format_vector(layer.biases)}",
                    f"  Last input: {format_vector(layer.last_input)}",
                    f"  Pre-activation: {format_vector(layer.last_pre_activation)}",
                    f"  Output: {format_vector(layer.last_output)}",
                    f"  Gradient: {format_vector(layer.last_gradient)}",
                    "",
                ]
            )

        self.query_one("#layers-panel", Static).update("\n".join(sections).rstrip())

    def _parse_structure(self) -> list[int]:
        raw = self.query_one("#structure-input", Input).value
        numbers = [int(part.strip()) for part in raw.split(",") if part.strip()]
        if len(numbers) < 2 or any(number <= 0 for number in numbers):
            raise ValueError("Structure must contain at least two positive integers.")
        return numbers

    def _parse_learning_rate(self) -> float:
        raw = self.query_one("#learning-rate-input", Input).value.strip()
        value = float(raw)
        if value <= 0:
            raise ValueError("Learning rate must be greater than zero.")
        return value

    def _parse_epoch_batch(self) -> int:
        raw = self.query_one("#epoch-count-input", Input).value.strip() or "1"
        value = int(raw)
        if value <= 0:
            raise ValueError("Epoch batch must be a positive integer.")
        return value

    def _parse_speed_ms(self, default: int) -> int:
        raw = self.query_one("#speed-input", Input).value.strip() or str(default)
        value = int(raw)
        if value < 10:
            raise ValueError("Run interval must be at least 10 ms.")
        return value

    def _current_preset(self) -> PresetName:
        return cast(PresetName, self.query_one("#preset-select", Select).value)

    def _log(self, message: str) -> None:
        self.query_one("#event-log", Log).write_line(message)


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())
