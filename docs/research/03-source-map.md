# Source Map

This is the reading list behind the current repo strategy.

## Priority 1: Official Module Scope

### HRW module handbook

Why it matters:

- This is the strongest official description of what the module expects.
- It anchors the project around feedforward networks, supervised learning, and project documentation.

Link:

- https://www.hochschule-ruhr-west.de/content/download/4693/file/Modulhandbuch_Mensch-Technik-InteraktionBPO2017.pdf

## Priority 2: Core Feedforward Network Understanding

### Simon Haykin, *Neural Networks and Learning Machines*

Why it matters:

- It is explicitly listed in the handbook literature section.
- Good anchor for canonical terminology and standard neural-network treatment.

### Rumelhart, Hinton, Williams (1986), *Learning representations by back-propagating errors*

Why it matters:

- Seminal backpropagation paper.
- Useful for understanding the historical and conceptual core of multilayer training.

Link:

- https://www.nature.com/articles/323533a0

### Michael Nielsen, *Neural Networks and Deep Learning*

Why it matters:

- One of the clearest learning-oriented explanations of feedforward nets and backpropagation.
- Especially useful when you want intuition, derivations, and small-scale from-scratch thinking.

Link:

- http://neuralnetworksanddeeplearning.com/

### Goodfellow, Bengio, Courville, *Deep Learning*

Why it matters:

- Strong reference for the bigger conceptual picture.
- Helps place feedforward networks, optimization, regularization, and representation learning in context.

Link:

- https://www.deeplearningbook.org/

## Priority 3: Supporting Theory

### Cybenko (1989), *Approximation by superpositions of a sigmoidal function*

Why it matters:

- Classical result related to universal approximation for feedforward networks.
- Helpful if you want a mathematically grounded explanation for why multilayer networks are expressive.

Link:

- https://doi.org/10.1287/moor.14.2.303

### Rosenblatt (1958), *The Perceptron*

Why it matters:

- Historical base for understanding the single-layer perceptron before multilayer networks.

Link:

- https://doi.org/10.1037/h0042519

## Priority 4: Secondary Module Topics

### Scholarpedia: Kohonen network

Why it matters:

- Good compact reference for self-organizing maps, which the handbook lists as a major secondary topic.

Link:

- http://www.scholarpedia.org/article/Kohonen_network

### Scholarpedia: Neural fields

Why it matters:

- Good reference point for the dynamic neural fields mentioned in the handbook.

Link:

- http://www.scholarpedia.org/article/Neural_fields

## How To Use This List

Read in this order:

1. HRW handbook entry,
2. Nielsen for intuition,
3. Haykin for canonical structure,
4. Rumelhart et al. for backprop foundations,
5. Goodfellow for broader context,
6. SOM and neural-field references only after the feedforward core is stable.

## Important Boundary

This repository is about understanding and implementing a small neural network from scratch, not about reproducing modern deep-learning tooling.
