"""Runnable entrypoint for the Neuroinformatik TUI."""

from src.tui.app import NeuroTuiApp


def run() -> None:
    NeuroTuiApp().run()


if __name__ == "__main__":
    run()
