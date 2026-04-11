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
    ├── web/
    └── ...
```

## Web

The repository includes a small web dashboard in `src/web/` for interacting with the network in the browser.
It serves static files and a WebSocket endpoint from the same port.

## Notes

- `docs/research/` contains the planning notes, math notes, and code-aligned explanations.
- `src/` contains the current implementation.
- `src/web/` contains the browser-based UI and server entry point.
- The current implementation uses a small feedforward network with manual forward pass, manual backpropagation, sigmoid activations, and MSE loss.
