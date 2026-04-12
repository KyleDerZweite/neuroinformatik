from src.tui.controller import TuiController


def test_configure_snapshot() -> None:
    controller = TuiController()
    snapshot = controller.configure([2, 3, 1], 0.25, "xor")

    assert snapshot.preset == "xor"
    assert snapshot.structure == [2, 3, 1]
    assert snapshot.learning_rate == 0.25
    assert snapshot.total_samples == 4
    assert snapshot.total_layers == 2


def test_forward_backward_flow() -> None:
    controller = TuiController()
    controller.configure([2, 2, 1], 1.0, "xor")

    forward = controller.step_forward()
    assert forward.phase == "forward_done"
    assert forward.current_sample is not None
    assert forward.current_sample.prediction is not None

    backward = controller.step_backward()
    assert backward.phase == "backward_done"
    assert backward.current_sample_index == 1


def test_epoch_and_layer_steps() -> None:
    controller = TuiController()
    controller.configure([1, 2, 1], 0.275, "sine")

    first_layer = controller.step_layer_forward()
    assert first_layer.current_layer_index == 1

    controller.step_layer_forward()
    layer_backward = controller.step_layer_backward()
    assert layer_backward.current_layer_index >= 0

    epoch = controller.step_epoch(2)
    assert epoch.current_epoch == 2
    assert len(epoch.epoch_losses) == 2
