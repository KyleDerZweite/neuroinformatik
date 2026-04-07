# Neuroinformatik: Neural Network From Scratch

This repository is intentionally reset to a study-first baseline for a **from-scratch neural network** project.

Context for this repo:

- Studiengang: `Energieinformatik`
- Fachsemester: `7`
- Studienform: `dual praxisintegriert`
- Modultyp: `Wahlmodul`

The goal is not to let AI produce the project for you. The goal is to structure the work so you can:

- understand the theory behind a neural network from scratch,
- map the lecture content to a realistic project scope,
- document your reasoning and decisions,
- keep implementation separate until the theory and plan are clear.

At this stage there is no implementation code. `src/` exists on purpose but stays empty.

## Implementation Focus

The implementation target for this repo is intentionally narrow:

- a neural network built **from scratch**,
- **no ML libraries** and no ready-made neural-network tooling,
- only **very basic libraries** if needed, such as standard-library utilities,
- two concrete tasks: **XOR classification** and **sine approximation**.

The point is to understand every part yourself:

- forward pass,
- activation functions,
- loss calculation,
- backpropagation,
- parameter updates,
- why the network succeeds or fails.

## Why This Structure

The official HRW module handbook for **Neuroinformatik** describes the practical core as:

- understanding the foundations of neuroinformatics,
- designing and training **feedforward neural networks**,
- developing a deeper understanding of **supervised learning**,
- transferring that understanding into a **practice-oriented software project**,
- while also covering **self-organizing maps**, **recurrent networks**, and **dynamic neural fields** as broader course topics.

That means the repo should first support:

1. theory capture,
2. lecture-slide organization,
3. project scoping,
4. research notes,
5. only then implementation.

## Repository Structure

```text
neuroinformatik/
├── .gitignore
├── README.md
├── docs/
│   └── research/
│       ├── 00-module-scope.md
│       ├── 01-learning-roadmap.md
│       ├── 02-project-strategy.md
│       ├── 03-source-map.md
│       └── 04-self-check-questions.md
└── src/
```

## Docs

The `docs/` directory is now intentionally small:

- `docs/research/`: focused notes for understanding and planning the from-scratch implementation.

Start with:

1. [docs/research/00-module-scope.md](docs/research/00-module-scope.md)
2. [docs/research/01-learning-roadmap.md](docs/research/01-learning-roadmap.md)
3. [docs/research/02-project-strategy.md](docs/research/02-project-strategy.md)
4. [docs/vorlesung-folien/README.md](docs/vorlesung-folien/README.md)

## Scope Recommendation

The safest first implementation target for this module is:

- a small **multilayer feedforward network**,
- implemented **from scratch**,
- trained with **backpropagation** and **gradient descent**,
- first validated on **XOR**,
- then extended to **sine approximation**,
- written without ML frameworks or numerical libraries that hide the core logic,
- documented well enough that you can defend every design choice in an oral exam.

Everything beyond that should be added only if the assignment or lecturer explicitly requires it.

## Sources Behind This Structure

The research notes in `docs/research/` are based on:

- the HRW module handbook entry for **Neuroinformatik**,
- standard neural-network references such as **Haykin**,
- seminal backpropagation literature,
- established explanatory sources for feedforward networks, SOMs, and related topics.

See [docs/research/03-source-map.md](docs/research/03-source-map.md) for the full list.
