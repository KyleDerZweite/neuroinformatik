"""NetworkController: wraps NeuralNetwork for granular step-by-step execution.

Decomposes the monolithic train() loop into individually callable steps:
  - step_epoch()         -> full epoch (all samples, forward+backward)
  - step_forward()       -> single sample forward pass
  - step_backward()      -> single sample backward pass (requires forward first)
  - step_layer_forward() -> forward through one layer at a time
  - step_layer_backward()-> backward through one layer at a time
  - run_continuous()     -> async loop for real-time training
"""

import asyncio
import os
import sys

# Add src/ to path so we can import neuralnet and simulation
_src_dir = os.path.join(os.path.dirname(__file__), "..")
if _src_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_src_dir))

from neuralnet import NeuralNetwork
from simulation import build_sine_training_data
from web.serializer import serialize_full_state, serialize_summary


# --- Preset training data ---

def _load_xor_data():
    """Returns (inputs, targets) for the XOR problem."""
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


def _load_sine_data():
    """Returns (inputs, targets) for the sine approximation problem."""
    return build_sine_training_data()


PRESETS = {
    "xor": _load_xor_data,
    "sine": _load_sine_data,
}


class NetworkController:
    """Wraps a NeuralNetwork for interactive, step-by-step training.

    All public methods return the serialized state dict after their operation.
    Phase guards prevent invalid operation sequences (e.g., backward without forward).
    """

    def __init__(self):
        self.network = None
        self.training_inputs = None
        self.training_targets = None
        self.epoch_losses = []
        self.current_epoch = 0
        self.current_sample_index = 0
        self.current_layer_index = 0
        self.phase = "idle"  # "idle", "forward_done", "backward_done"
        self.epoch_loss_accumulator = 0.0
        self.samples_in_current_epoch = 0
        self.mode = "inspect"  # "inspect" or "run"
        self._stop_flag = False
        self._last_config = None
        self._backward_gradient = None
        self._backward_layer_index = None
        self._layer_forward_input = None

    def configure(self, structure, learning_rate, preset):
        """Create a fresh network and load training data."""
        self._stop_flag = True

        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")

        self._last_config = (structure, learning_rate, preset)
        self.network = NeuralNetwork(structure, learning_rate)
        self.training_inputs, self.training_targets = PRESETS[preset]()
        self._reset_training_state()
        return self.get_state()

    def reset(self):
        """Re-initialize the network with the same config (re-randomized weights)."""
        if self._last_config is None:
            raise ValueError("No configuration to reset. Call configure() first.")

        self._stop_flag = True
        structure, learning_rate, preset = self._last_config
        self.network = NeuralNetwork(structure, learning_rate)
        self._reset_training_state()
        return self.get_state()

    def _reset_training_state(self):
        """Reset all training tracking to initial state."""
        self.epoch_losses = []
        self.current_epoch = 0
        self.current_sample_index = 0
        self.current_layer_index = 0
        self.phase = "idle"
        self.epoch_loss_accumulator = 0.0
        self.samples_in_current_epoch = 0
        self._backward_gradient = None
        self._backward_layer_index = None
        self._layer_forward_input = None

    def get_state(self):
        """Return the full serialized state dict."""
        return serialize_full_state(self)

    def get_summary(self):
        """Return lightweight state for run-mode updates."""
        return serialize_summary(self)

    def _require_network(self):
        """Guard: raise if no network configured."""
        if self.network is None:
            raise ValueError("No network configured. Call configure() first.")

    # --- Epoch-level stepping ---

    def step_epoch(self, count=1):
        """Run one or more full epochs."""
        self._require_network()

        for _ in range(count):
            epoch_loss_sum = 0.0
            for sample_idx in range(len(self.training_inputs)):
                sample_input = self.training_inputs[sample_idx]
                sample_target = self.training_targets[sample_idx]
                self.network.forward(sample_input)
                epoch_loss_sum += self.network.calculate_sample_loss(sample_target)
                self.network.backward(sample_target)

            average_loss = epoch_loss_sum / len(self.training_inputs)
            self.epoch_losses.append(average_loss)
            self.current_epoch += 1

        # After stepping epochs, reset sample-level state
        self.current_sample_index = 0
        self.phase = "idle"
        self.epoch_loss_accumulator = 0.0
        self.samples_in_current_epoch = 0
        self._backward_gradient = None
        self._backward_layer_index = None
        self._layer_forward_input = None

        return self.get_state()

    # --- Sample-level stepping ---

    def step_forward(self, sample_index=None):
        """Run forward pass for a single sample. Does NOT update weights."""
        self._require_network()

        if sample_index is not None:
            self.current_sample_index = sample_index

        idx = self.current_sample_index
        if idx >= len(self.training_inputs):
            idx = 0
            self.current_sample_index = 0

        sample_input = self.training_inputs[idx]
        self.network.forward(sample_input)

        self.phase = "forward_done"
        self.current_layer_index = 0
        self._backward_gradient = None
        self._backward_layer_index = None
        self._layer_forward_input = None

        return self.get_state()

    def step_backward(self):
        """Run backward pass for the current sample. Requires forward_done.

        Updates weights and biases via backpropagation. Advances to next sample.
        If this was the last sample in the epoch, finalizes the epoch.
        """
        self._require_network()

        if self.phase != "forward_done":
            raise ValueError(
                "Cannot step backward: no forward pass completed. "
                "Run step_forward() first."
            )

        idx = self.current_sample_index
        sample_target = self.training_targets[idx]

        self.epoch_loss_accumulator += self.network.calculate_sample_loss(sample_target)
        self.samples_in_current_epoch += 1

        self.network.backward(sample_target)
        self.phase = "backward_done"

        # Advance to next sample
        self.current_sample_index += 1
        if self.current_sample_index >= len(self.training_inputs):
            self._finalize_epoch()

        return self.get_state()

    def _finalize_epoch(self):
        """Record average loss and reset for next epoch."""
        if self.samples_in_current_epoch > 0:
            average_loss = self.epoch_loss_accumulator / self.samples_in_current_epoch
            self.epoch_losses.append(average_loss)
            self.current_epoch += 1

        self.current_sample_index = 0
        self.epoch_loss_accumulator = 0.0
        self.samples_in_current_epoch = 0
        self.phase = "idle"

    # --- Per-layer stepping ---

    def step_layer_forward(self):
        """Forward through a single layer.

        Call repeatedly to step through the network one layer at a time.
        After the last layer completes, sets network.last_output and phase="forward_done".
        """
        self._require_network()

        layers = self.network.layers

        if self._layer_forward_input is None:
            idx = self.current_sample_index
            if idx >= len(self.training_inputs):
                idx = 0
                self.current_sample_index = 0
            self._layer_forward_input = self.training_inputs[idx]
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

    def step_layer_backward(self):
        """Backward through a single layer.

        Call repeatedly to step through backpropagation one layer at a time.
        After the first layer (index 0) completes, accumulates loss and advances sample.
        """
        self._require_network()

        if self.phase != "forward_done" and self._backward_gradient is None:
            raise ValueError(
                "Cannot step layer backward: no forward pass completed. "
                "Complete a forward pass first."
            )

        layers = self.network.layers

        if self._backward_gradient is None:
            idx = self.current_sample_index
            sample_target = self.training_targets[idx]

            self.epoch_loss_accumulator += self.network.calculate_sample_loss(sample_target)
            self.samples_in_current_epoch += 1

            self._backward_gradient = [
                self.network.last_output[i] - sample_target[i]
                for i in range(len(sample_target))
            ]
            self._backward_layer_index = len(layers) - 1

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

    # --- Continuous run mode ---

    async def run_continuous(self, send_callback, speed_ms=50):
        """Run training epochs continuously, sending state updates via callback."""
        self._require_network()
        self._stop_flag = False
        self.mode = "run"

        epoch_count = 0
        while not self._stop_flag:
            epoch_loss_sum = 0.0
            for sample_idx in range(len(self.training_inputs)):
                sample_input = self.training_inputs[sample_idx]
                sample_target = self.training_targets[sample_idx]
                self.network.forward(sample_input)
                epoch_loss_sum += self.network.calculate_sample_loss(sample_target)
                self.network.backward(sample_target)

            average_loss = epoch_loss_sum / len(self.training_inputs)
            self.epoch_losses.append(average_loss)
            self.current_epoch += 1
            epoch_count += 1

            # Full state every 10 epochs, summary otherwise
            if epoch_count % 10 == 0:
                await send_callback({"type": "state", "data": self.get_state()})
            else:
                await send_callback({"type": "summary", "data": self.get_summary()})

            await asyncio.sleep(speed_ms / 1000.0)

        self.mode = "inspect"
        self.phase = "idle"
        self.current_sample_index = 0
        self.epoch_loss_accumulator = 0.0
        self.samples_in_current_epoch = 0

        await send_callback({"type": "state", "data": self.get_state()})

    def pause(self):
        """Signal the continuous run loop to stop."""
        self._stop_flag = True
