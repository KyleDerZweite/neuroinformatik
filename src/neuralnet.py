
from matrix import create_bias_vector, create_weight_matrix


class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = create_weight_matrix(self.input_size, self.output_size)
        self.biases = create_bias_vector(self.output_size)

    def forward(self, input_data):
        pass

    def backward(self, target_output):
        pass

class NeuralNetwork:
    def __init__(self, structure: list[int]):
        """Build layers from sizes: [input, hidden..., output].

        Examples:
        - XOR: [2, 2, 1] 
        - Sine approximation: [1, 8, 1]
        """
        self.layers = []
        for i in range(len(structure) - 1):
            input_size = structure[i]
            output_size = structure[i + 1]
            self.layers.append(Layer(input_size, output_size))

    def forward(self, input_data):
        pass

    def backward(self, target_output):
        pass

    def train(self, training_inputs, training_targets, epochs):
        for epoch in range(epochs):
            for i in range(len(training_inputs)):
                input_data = training_inputs[i]
                target_output = training_targets[i]
                self.forward(input_data)
                self.backward(target_output)