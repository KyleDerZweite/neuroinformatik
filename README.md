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
    ├── api/
    ├── web/
    └── ...
```

## Dashboard Architecture

The dashboard is now split into two parts:

- `src/api/`: FastAPI backend exposing typed HTTP and WebSocket endpoints.
- `src/web/`: Vite + React frontend scaffold.

The neural network implementation still lives in `src/` and is used by the API layer.

## Running

API:

```bash
uv run uvicorn src.api.main:app --reload
```

Frontend scaffold:

```bash
cd src/web
pnpm install
pnpm dev
```

Both together:

```bash
uv run python scripts/dev.py
```

## Notes

- `docs/research/` contains the planning notes, math notes, and code-aligned explanations.
- `src/` contains the neural network implementation plus the dashboard API.
- `src/web/` contains the frontend application scaffold.
- The current implementation uses a small feedforward network with manual forward pass, manual backpropagation, sigmoid activations, and MSE loss.
