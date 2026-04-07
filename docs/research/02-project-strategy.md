# Project Strategy

## Main Recommendation

Keep the first complete project intentionally narrow:

- one language,
- one clean implementation path,
- one clearly documented model,
- two concrete supervised learning tasks: XOR and sine approximation.

Your previous structure split the work into Python and Rust. That is interesting technically, but it increases cognitive load before the core module objective is secured.

Inference from the module description:

- your first priority is not language comparison,
- your first priority is showing that you understand and can explain a feedforward network from first principles.

For this repo, that also means:

- no neural-network libraries,
- no automatic differentiation,
- no numerical frameworks that hide the matrix and gradient logic,
- only very basic helper libraries if the module permits them.

## Suggested Project Sequence

### Step 1

Write down the mathematical model in your own words:

- neuron equation,
- layer equation,
- activation choice,
- loss function,
- training rule.

### Step 2

Choose the smallest valid supervised tasks.

Required order for this repo:

- `XOR` first,
- `sine approximation` second.

A good rule:

- if you cannot manually inspect the data and manually reason about the target, the task is too large for the first implementation.

### Step 3

Define success criteria before coding.

Examples:

- the loss decreases consistently,
- the model fits XOR,
- the model approximates a sine curve with visibly improved predictions,
- the model generalizes on a held-out split,
- you can explain one full training iteration line by line.

### Step 4

Plan observability before implementation.

You should know in advance what you want to inspect:

- parameter shapes,
- intermediate activations,
- loss per epoch,
- gradients per layer,
- predictions before and after training.

### Step 5

Only then start implementing in `src/`.

## Deliverable Mindset

Because the assessment includes documentation and an oral component, your project should be defendable from three angles:

- `Theory`: why the model works.
- `Implementation`: how the math maps to code.
- `Evaluation`: what the model learned and where it fails.

## What To Avoid

- adding frameworks before the manual implementation is understood,
- using ML libraries that solve the core problem for you,
- optimizing performance before correctness is proven,
- skipping derivations and relying on memory,
- building a large CLI or package layout too early,
- mixing exploratory notebooks, production code, and lecture notes in the same place.
