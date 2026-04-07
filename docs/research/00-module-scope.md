# Module Scope

## Repository Context

This repository is aligned with the HRW module handbook entry for **Neuroinformatik** in the `Mensch-Technik-Interaktion BPO2017` handbook.

## Official Anchor

Relevant points from pages 152-153:

- Module name: `Neuroinformatik`
- Lecturer: `Prof. Dr. Uwe Handmann`
- Workload: `180 h`, `6 credits`
- Format: lecture, exercises, and project work
- Assessment: `mündliche Prüfung inkl. Dokumentation der Projektarbeit`

Core learning outcomes listed there:

- students understand the foundations of neuroinformatics,
- students can design and train **feedforward neural networks**,
- students develop deeper understanding in **supervised learning tasks**,
- students transfer that knowledge into a **practice-oriented software project**,
- students implement a self-designed training approach.

The module content listed there emphasizes:

- biological motivation and simple neuron models,
- **feedforward neural networks** as the main focus,
- supervised learning in multilayer networks,
- learning strategies and optimization approaches,
- **self-organizing maps** as a second focus,
- additional discussion of **recurrent networks** and **dynamic neural fields**.

## Practical Interpretation

Inference from the official module description:

- The implementation target should begin with a **multilayer perceptron / feedforward network**, not with a broad framework.
- The project should be small enough to explain end to end.
- Documentation matters because the assessment explicitly includes project documentation and an oral exam.
- It is risky to start with performance engineering, multi-language rewrites, or a large project layout before the theory is stable.

For this repository, that gets narrowed further to:

- one from-scratch implementation path,
- no ML libraries,
- XOR as the first sanity-check task,
- sine approximation as the second concrete task.

## What This Means For Your Repo

This repository should support four things before any coding starts:

1. understanding the theory,
2. organizing course material,
3. narrowing the actual project scope,
4. preparing to defend the work orally.

That is why the current structure is documentation-heavy and keeps `src/` empty.

## Immediate Recommendation

Treat the project in three layers:

1. `Theory`: neuron model, linear algebra, activations, loss, gradient descent, backpropagation.
2. `Project scope`: XOR first, then sine approximation, with one small feedforward network architecture you can explain fully.
3. `Implementation`: only after the first two are written down clearly.

## Source

- HRW handbook PDF: https://www.hochschule-ruhr-west.de/content/download/4693/file/Modulhandbuch_Mensch-Technik-InteraktionBPO2017.pdf

Important note:

- This handbook entry is the best official source found during this research pass.
- The semester/program allocation in handbooks can differ across study regulations, so the imported course material should be treated as the stronger practical guide for the current run of the module.
