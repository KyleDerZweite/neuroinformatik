# Learning Roadmap

This roadmap is optimized for understanding a neural network from scratch before implementing anything.

## Phase 1: Mathematical Minimum

You should be able to explain and derive:

- vectors, matrices, matrix multiplication,
- affine transformation: `z = W x + b`,
- activation functions and why nonlinearity matters,
- loss functions for regression vs classification,
- partial derivatives and the chain rule.

If any of these is weak, implementation will turn into copying patterns instead of understanding.

## Phase 2: Single Neuron To Multilayer Network

Master this dependency chain in order:

1. perceptron and decision boundary,
2. sigmoid or tanh neuron as differentiable alternative,
3. one hidden layer network,
4. forward pass,
5. loss computation,
6. backward pass,
7. parameter update.

The key milestone is not “I can run it.” The milestone is “I can explain why each tensor shape and derivative is correct.”

## Phase 3: Training Dynamics

Before coding, you should understand:

- why initialization matters,
- why learning rate matters,
- why saturated activations can hurt learning,
- how overfitting differs from underfitting,
- why train/validation/test separation exists,
- what failure looks like when the network does not learn.

## Phase 4: Minimal Project Target

Start with a target you can fully defend:

- one dataset,
- one network family,
- one training procedure,
- clear metrics,
- clear visualizations or tables for results.

For this module, the default safe target is:

- a small **feedforward network** for two simple supervised tasks.

The recommended order in this repo is:

1. `XOR` as the first sanity check for nonlinear separation.
2. `sine approximation` as the first small regression task.

Both tasks are small enough that you can inspect the behavior directly and explain the results.

Examples of bad first tasks:

- large image datasets,
- many architectural variants at once,
- combining multiple languages or runtimes,
- trying to reproduce a modern deep-learning stack,
- using libraries that hide the forward or backward pass.

## Phase 5: Broader Module Topics

The handbook suggests these should be understood at least conceptually:

- self-organizing maps,
- recurrent networks,
- dynamic neural fields.

Recommendation:

- keep these in theory notes unless the actual assignment requires implementation.

## Stop Conditions Before Coding

Do not start the implementation until you can answer all of these without looking things up:

- Why does a hidden layer help with nonlinear separation?
- What exactly is passed forward through the network?
- What exactly flows backward during backpropagation?
- Which quantities are parameters and which are intermediate values?
- What shapes do `W`, `b`, activations, and gradients have at each layer?
