
import math

from matrix import create_weight_matrix, create_bias_vector

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

        self.last_input: list[float] | None = None
        self.last_pre_activation_values: list[float] | None = None
        self.last_output: list[float] | None = None
        self.last_gradient_from_next_layer: list[float] | None = None

    def _sigmoid(self, value):
        return 1.0 / (1.0 + math.exp(-value))

    def _sigmoid_derivative_from_output(self, sigmoid_output):
        return sigmoid_output * (1.0 - sigmoid_output)

    def forward(self, input_data):
        """Rechnet den Vorwärtsschritt für diesen Layer."""
        self.last_input = input_data

        # speicher nur für debugging
        self.last_pre_activation_values = []
        self.last_output = []
        for output_neuron_index in range(self.output_size):
            # erst summe dann aktivierung
            pre_activation_value = self.biases[output_neuron_index]
            for input_neuron_index in range(self.input_size):
                input_value = input_data[input_neuron_index]
                weight_value = self.weights[input_neuron_index][output_neuron_index]
                pre_activation_value += input_value * weight_value
            
            activated_output_value = self._sigmoid(pre_activation_value)

            self.last_pre_activation_values.append(pre_activation_value)
            self.last_output.append(activated_output_value)

        return self.last_output

    def backward(self, gradient_from_next_layer):
        """Rechnet den Rückwärtsschritt und aktualisiert Gewichte/Biases."""
        self.last_gradient_from_next_layer = gradient_from_next_layer
        assert self.last_output is not None
        assert self.last_input is not None

        # fehler für jeden ausgang
        local_gradient_per_output = []
        for output_neuron_index in range(self.output_size):
            output_value = self.last_output[output_neuron_index]
            sigmoid_derivative = self._sigmoid_derivative_from_output(output_value)
            gradient_value = gradient_from_next_layer[output_neuron_index] * sigmoid_derivative
            local_gradient_per_output.append(gradient_value)

        # gradient für das vorherige layer
        gradient_for_previous_layer = []
        for input_neuron_index in range(self.input_size):
            propagated_gradient_sum = 0.0
            for output_neuron_index in range(self.output_size):
                current_weight = self.weights[input_neuron_index][output_neuron_index]
                propagated_gradient_sum += local_gradient_per_output[output_neuron_index] * current_weight
            gradient_for_previous_layer.append(propagated_gradient_sum)

        # bias anpassen
        for output_neuron_index in range(self.output_size):
            self.biases[output_neuron_index] -= self.learning_rate * local_gradient_per_output[output_neuron_index]

        # gewichte anpassen
        for input_neuron_index in range(self.input_size):
            input_value = self.last_input[input_neuron_index]
            for output_neuron_index in range(self.output_size):
                weight_gradient = input_value * local_gradient_per_output[output_neuron_index]
                self.weights[input_neuron_index][output_neuron_index] -= self.learning_rate * weight_gradient

        return gradient_for_previous_layer


class NeuralNetwork:
    def __init__(self, structure: list[int], learning_rate: float = 0.1):
        """Erstellt ein Netzwerk aus Layer-Größen, z.B. [2, 2, 1]."""
        self.layers = []
        for layer_position in range(len(structure) - 1):
            layer_input_size = structure[layer_position]
            layer_output_size = structure[layer_position + 1]
            layer = Layer(layer_input_size, layer_output_size, learning_rate)
            self.layers.append(layer)

        self.last_output: list[float] | None = None
        self.learning_rate = learning_rate

    def forward(self, input_data):
        """Leitet Eingabe durch alle Layer."""
        current_layer_input = input_data
        # layer nach layer durchrechnen
        for layer in self.layers:
            current_layer_input = layer.forward(current_layer_input)

        self.last_output = current_layer_input
        return current_layer_input

    def backward(self, target_output):
        """Berechnet Gradienten rückwärts ab der Ausgabeschicht."""
        assert self.last_output is not None
        last_output = self.last_output
        current_gradient_vector = []
        # fehler vom output sammeln
        for output_value_index in range(len(last_output)):
            predicted_output_value = last_output[output_value_index]
            expected_output_value = target_output[output_value_index]
            output_gradient_value = predicted_output_value - expected_output_value
            current_gradient_vector.append(output_gradient_value)

        # rueckwärts durch alle layer
        for layer in reversed(self.layers):
            current_gradient_vector = layer.backward(current_gradient_vector)

        return current_gradient_vector

    def calculate_sample_loss(self, target_output):
        """MSE pro Sample: 1/2 * sum((prediction - target)^2)."""
        assert self.last_output is not None
        last_output = self.last_output
        total_loss = 0.0
        # fehler pro output aufsummieren
        for output_index in range(len(last_output)):
            prediction_value = last_output[output_index]
            expected_value = target_output[output_index]
            error_value = prediction_value - expected_value
            total_loss += 0.5 * (error_value ** 2)
        return total_loss

    def train(
        self,
        training_inputs,
        training_targets,
        target_loss,
        fail_safe_max_epochs=100000,
        report_every_epochs=0,
    ):
        """Trainiert bis target_loss erreicht ist oder fail-safe greift.

        Gibt die durchschnittlichen Verluste pro Epoche zurueck. Wenn der
        fail-safe zuerst greift, endet das Training einfach nach
        fail_safe_max_epochs.
        """
        epoch_losses = []
        # jede epoche alle trainingsdaten durchgehen
        for epoch_number in range(1, fail_safe_max_epochs + 1):
            epoch_loss_sum = 0.0
            # immer ein sample nach dem anderen
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
                return epoch_losses

        return epoch_losses
