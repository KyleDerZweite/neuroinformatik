
import math

try:
    from matrix import create_bias_vector, create_weight_matrix
except ModuleNotFoundError:
    from src.matrix import create_bias_vector, create_weight_matrix


class Layer:
    """Eine einfache Dense-Layer mit Sigmoid-Aktivierung.

    Vorwärts:
    pre_activation = input * weights + biases
    output = sigmoid(pre_activation)
    """

    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = create_weight_matrix(self.input_size, self.output_size)
        self.biases = create_bias_vector(self.output_size)

        # Keep values from the last run so we can learn/debug step by step.
        self.last_input = None
        self.last_pre_activation_values = None
        self.last_output = None
        self.last_gradient_from_next_layer = None

    def _sigmoid(self, value):
        return 1.0 / (1.0 + math.exp(-value))

    def _sigmoid_derivative_from_output(self, sigmoid_output):
        return sigmoid_output * (1.0 - sigmoid_output)

    def forward(self, input_data):
        """Rechnet den Vorwärtsschritt fuer diese Layer."""
        if len(input_data) != self.input_size:
            raise ValueError("input_data has wrong size for this layer")

        self.last_input = input_data

        self.last_pre_activation_values = []
        for output_neuron_index in range(self.output_size):
            weighted_sum = self.biases[output_neuron_index]
            for input_neuron_index in range(self.input_size):
                input_value = input_data[input_neuron_index]
                weight_value = self.weights[input_neuron_index][output_neuron_index]
                weighted_sum += input_value * weight_value
            self.last_pre_activation_values.append(weighted_sum)

        self.last_output = []
        for output_neuron_index in range(self.output_size):
            pre_activation_value = self.last_pre_activation_values[output_neuron_index]
            activated_output_value = self._sigmoid(pre_activation_value)
            self.last_output.append(activated_output_value)

        return self.last_output

    def backward(self, gradient_from_next_layer):
        """Rechnet den Rueckwärtsschritt und aktualisiert Gewichte/Biases."""
        if len(gradient_from_next_layer) != self.output_size:
            raise ValueError("gradient_from_next_layer has wrong size for this layer")
        if self.last_input is None or self.last_output is None:
            raise ValueError("Call forward before backward")

        self.last_gradient_from_next_layer = gradient_from_next_layer

        local_gradient_per_output = []
        for output_neuron_index in range(self.output_size):
            output_value = self.last_output[output_neuron_index]
            sigmoid_derivative = self._sigmoid_derivative_from_output(output_value)
            gradient_value = gradient_from_next_layer[output_neuron_index] * sigmoid_derivative
            local_gradient_per_output.append(gradient_value)

        gradient_for_previous_layer = []
        for input_neuron_index in range(self.input_size):
            propagated_gradient_sum = 0.0
            for output_neuron_index in range(self.output_size):
                current_weight = self.weights[input_neuron_index][output_neuron_index]
                propagated_gradient_sum += local_gradient_per_output[output_neuron_index] * current_weight
            gradient_for_previous_layer.append(propagated_gradient_sum)

        for output_neuron_index in range(self.output_size):
            self.biases[output_neuron_index] -= self.learning_rate * local_gradient_per_output[output_neuron_index]

        for input_neuron_index in range(self.input_size):
            input_value = self.last_input[input_neuron_index]
            for output_neuron_index in range(self.output_size):
                weight_gradient = input_value * local_gradient_per_output[output_neuron_index]
                self.weights[input_neuron_index][output_neuron_index] -= self.learning_rate * weight_gradient

        return gradient_for_previous_layer


class NeuralNetwork:
    def __init__(self, structure: list[int], learning_rate: float = 0.1):
        """Erstellt ein Netzwerk aus Layer-Groessen, z.B. [2, 2, 1]."""
        if len(structure) < 2:
            raise ValueError("structure needs at least input and output size")

        self.layers = []
        for layer_position in range(len(structure) - 1):
            layer_input_size = structure[layer_position]
            layer_output_size = structure[layer_position + 1]
            layer = Layer(layer_input_size, layer_output_size, learning_rate)
            self.layers.append(layer)

        self.last_output = None
        self.learning_rate = learning_rate

    def forward(self, input_data):
        """Leitet Eingabe durch alle Layer."""
        current_layer_input = input_data
        for layer_index in range(len(self.layers)):
            current_layer_input = self.layers[layer_index].forward(current_layer_input)

        self.last_output = current_layer_input
        return current_layer_input

    def backward(self, target_output):
        """Berechnet Gradienten rückwärts ab der Ausgabeschicht."""
        if self.last_output is None:
            raise ValueError("Call forward before backward")
        if len(target_output) != len(self.last_output):
            raise ValueError("target_output has wrong size")

        current_gradient_vector = []
        for output_value_index in range(len(self.last_output)):
            predicted_output_value = self.last_output[output_value_index]
            expected_output_value = target_output[output_value_index]
            output_gradient_value = predicted_output_value - expected_output_value
            current_gradient_vector.append(output_gradient_value)

        for layer_index in range(len(self.layers) - 1, -1, -1):
            current_gradient_vector = self.layers[layer_index].backward(current_gradient_vector)

        return current_gradient_vector

    def calculate_sample_loss(self, target_output):
        """MSE pro Sample: 1/2 * sum((prediction - target)^2)."""
        if self.last_output is None:
            raise ValueError("Call forward before calculate_sample_loss")
        if len(target_output) != len(self.last_output):
            raise ValueError("target_output has wrong size")

        total_loss = 0.0
        for output_index in range(len(self.last_output)):
            prediction_value = self.last_output[output_index]
            expected_value = target_output[output_index]
            error_value = prediction_value - expected_value
            total_loss += 0.5 * error_value * error_value
        return total_loss

    def train(
        self,
        training_inputs,
        training_targets,
        target_loss,
        fail_safe_max_epochs=100000,
        report_every_epochs=0,
    ):
        """Trainiert bis target_loss erreicht ist oder fail-safe greift."""
        if len(training_inputs) != len(training_targets):
            raise ValueError("training_inputs and training_targets must have same length")
        if fail_safe_max_epochs <= 0:
            raise ValueError("fail_safe_max_epochs must be > 0")
        if target_loss < 0.0:
            raise ValueError("target_loss must be >= 0")
        if report_every_epochs < 0:
            raise ValueError("report_every_epochs must be >= 0")

        epoch_losses = []
        for epoch_number in range(1, fail_safe_max_epochs + 1):
            epoch_loss_sum = 0.0
            for sample_index in range(len(training_inputs)):
                sample_input = training_inputs[sample_index]
                sample_target_output = training_targets[sample_index]
                self.forward(sample_input)
                epoch_loss_sum += self.calculate_sample_loss(sample_target_output)
                self.backward(sample_target_output)

            average_epoch_loss = epoch_loss_sum / len(training_inputs)
            epoch_losses.append(average_epoch_loss)

            if report_every_epochs > 0:
                if epoch_number == 1 or epoch_number % report_every_epochs == 0:
                    print("epoch", epoch_number, "loss", round(average_epoch_loss, 6))

            if average_epoch_loss <= target_loss:
                print("target_loss reached at epoch", epoch_number)
                return epoch_losses

        raise RuntimeError(
            "target_loss was not reached within fail-safe limit of "
            + str(fail_safe_max_epochs)
            + " epochs"
        )