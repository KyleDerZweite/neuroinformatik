"""Converts network state to JSON-serializable dicts.

Pure functions — no side effects. All floats rounded to PRECISION decimals.
None values pass through as None (become JSON null).
"""

PRECISION = 6


def _round(value):
    if value is None:
        return None
    return round(value, PRECISION)


def _round_list(values):
    if values is None:
        return None
    return [_round(v) for v in values]


def _round_matrix(matrix):
    if matrix is None:
        return None
    return [_round_list(row) for row in matrix]


def serialize_layer(layer, index):
    """Serialize a single Layer object to a dict."""
    return {
        "index": index,
        "input_size": layer.input_size,
        "output_size": layer.output_size,
        "weights": _round_matrix(layer.weights),
        "biases": _round_list(layer.biases),
        "last_input": _round_list(layer.last_input),
        "last_pre_activation": _round_list(layer.last_pre_activation_values),
        "last_output": _round_list(layer.last_output),
        "last_gradient": _round_list(layer.last_gradient_from_next_layer),
        "learning_rate": layer.learning_rate,
    }


def serialize_full_state(controller):
    """Serialize the complete controller state for the frontend."""
    network = controller.network
    layers_data = []
    structure = []

    if network is not None:
        for layer_index, layer in enumerate(network.layers):
            layers_data.append(serialize_layer(layer, layer_index))
        if network.layers:
            structure.append(network.layers[0].input_size)
            for layer in network.layers:
                structure.append(layer.output_size)

    # Current sample info
    current_sample = None
    if (controller.training_inputs is not None
            and controller.current_sample_index < len(controller.training_inputs)):
        sample_input = controller.training_inputs[controller.current_sample_index]
        sample_target = controller.training_targets[controller.current_sample_index]
        prediction = _round_list(network.last_output) if network and network.last_output else None
        loss = None
        if network and network.last_output and controller.phase == "forward_done":
            try:
                loss = _round(network.calculate_sample_loss(sample_target))
            except (ValueError, AttributeError):
                loss = None
        current_sample = {
            "index": controller.current_sample_index,
            "input": _round_list(sample_input),
            "target": _round_list(sample_target),
            "prediction": prediction,
            "loss": loss,
        }

    return {
        "topology": {
            "structure": structure,
            "learning_rate": _round(controller.network.learning_rate) if network else None,
        },
        "layers": layers_data,
        "network_output": _round_list(network.last_output) if network and network.last_output else None,
        "training": {
            "current_epoch": controller.current_epoch,
            "current_sample_index": controller.current_sample_index,
            "current_layer_index": controller.current_layer_index,
            "phase": controller.phase,
            "epoch_losses": [_round(loss_value) for loss_value in controller.epoch_losses],
            "mode": controller.mode,
            "total_samples": len(controller.training_inputs) if controller.training_inputs else 0,
            "total_layers": len(network.layers) if network else 0,
        },
        "current_sample": current_sample,
    }


def serialize_summary(controller):
    """Lightweight state for run-mode updates."""
    latest_loss = None
    if controller.epoch_losses:
        latest_loss = _round(controller.epoch_losses[-1])

    return {
        "training": {
            "current_epoch": controller.current_epoch,
            "latest_loss": latest_loss,
            "phase": controller.phase,
            "mode": controller.mode,
        },
        "network_output": _round_list(
            controller.network.last_output
        ) if controller.network and controller.network.last_output else None,
    }
