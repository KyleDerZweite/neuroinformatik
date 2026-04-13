# Neuroinformatik

Study repository for a neural network built from scratch.
The project was coded fully with AI.

## Scope

The project is intentionally narrow:

- feedforward neural network from scratch
- no ML frameworks or auto-diff
- XOR classification as the first task
- sine approximation as the second task

The goal is to keep the math, implementation, and evaluation easy to explain end to end.

## Structure

```text
neuroinformatik/
├── README.md
├── docs/
│   └── research/
└── src/
    ├── tui/
    └── ...
```

## Running

TUI:

```bash
uv run python -m src.tui
```

## Notes

- `docs/research/` contains the planning notes, math notes, and code-aligned explanations.
- `src/` contains the neural network implementation and the terminal UI.
- `src/tui/` contains the Textual app and the session controller.
- The current implementation uses a small feedforward network with manual forward pass, manual backpropagation, sigmoid activations, and MSE loss.
