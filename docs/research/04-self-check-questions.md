# Self-Check Questions

Use these questions before you write code and again before any presentation or oral exam.

## Core Theory

- What is the difference between a perceptron and a differentiable neuron?
- Why is nonlinearity required in hidden layers?
- What does a weight matrix do geometrically?
- Why can a single linear layer not solve XOR?
- What is the difference between regression loss and classification loss?

## Forward Pass

- What is computed in each layer before activation?
- What is the role of the bias term?
- What shape does each intermediate result have?
- Which values must be stored for backpropagation?

## Backpropagation

- What does the chain rule mean in this context?
- Why is backpropagation efficient compared with naive differentiation?
- What gradient do you need for the output layer?
- How is the gradient propagated to earlier layers?
- Why do you compute gradients with respect to both weights and biases?

## Training

- What does the learning rate control?
- What signs tell you the learning rate is too high or too low?
- Why does initialization matter?
- What is the difference between batch, mini-batch, and stochastic gradient descent?
- How do you detect overfitting in a small project?

## Project Defense

- Why did you choose this dataset?
- Why did you choose this architecture?
- Why is this activation function appropriate here?
- How do you know the network actually learned something meaningful?
- What are the limitations of your implementation?
