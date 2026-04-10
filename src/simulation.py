import math

try:
	from neuralnet import NeuralNetwork
except ModuleNotFoundError:
	from src.neuralnet import NeuralNetwork

def run_xor_simulation():
	print("\n=== XOR simulation ===")

	xor_training_inputs = [
		[0.0, 0.0],
		[0.0, 1.0],
		[1.0, 0.0],
		[1.0, 1.0],
	]
	xor_training_targets = [
		[0.0],
		[1.0],
		[1.0],
		[0.0],
	]

	xor_network = NeuralNetwork([2, 2, 1], learning_rate=1)
	xor_target_loss = 0.0001
	xor_report_every_epochs = 500
	xor_epoch_losses = xor_network.train(
		xor_training_inputs,
		xor_training_targets,
		xor_target_loss,
		report_every_epochs=xor_report_every_epochs,
	)

	print("start_loss", round(xor_epoch_losses[0], 6))
	print("end_loss", round(xor_epoch_losses[-1], 6))

	print("predictions:")
	for sample_index in range(len(xor_training_inputs)):
		current_input = xor_training_inputs[sample_index]
		prediction_value = xor_network.forward(current_input)[0]
		print(current_input, "->", round(prediction_value, 6))


def build_sine_training_data():
	"""Erzeugt Sinus-Daten fuer x von 0 bis 7 mit 100 Zwischenschritten pro 1.0."""
	input_min_value = 0.0
	input_max_value = 7.0
	steps_between_integers = 100
	input_range = input_max_value - input_min_value
	total_step_count = int(input_range * steps_between_integers)

	sine_training_inputs = []
	sine_training_targets = []

	for step_index in range(total_step_count + 1):
		raw_input_value = input_min_value + (step_index / steps_between_integers)

		# Input auf [0, 1] skalieren.
		normalized_input_value = (raw_input_value - input_min_value) / input_range

		# Wegen Sigmoid-Ausgabe Ziel auch auf [0, 1] skalieren.
		raw_sine_target = math.sin(raw_input_value)
		normalized_sine_target = (raw_sine_target + 1.0) / 2.0

		sine_training_inputs.append([normalized_input_value])
		sine_training_targets.append([normalized_sine_target])

	return sine_training_inputs, sine_training_targets


def run_sine_simulation():
	print("\n=== Sine simulation ===")

	sine_training_inputs, sine_training_targets = build_sine_training_data()

	target_loss = 0.0001
	report_every_epochs = 500

	sine_network = NeuralNetwork([1, 2, 1], learning_rate=0.275)
	sine_epoch_losses = sine_network.train(
		sine_training_inputs,
		sine_training_targets,
		target_loss,
		report_every_epochs=report_every_epochs,
	)

	print("start_loss", round(sine_epoch_losses[0], 6))
	print("end_loss", round(sine_epoch_losses[-1], 6))

	print("sample predictions (x = 0..7):")
	for integer_input in range(8):
		normalized_input_value = integer_input / 7.0
		predicted_output_normalized = sine_network.forward([normalized_input_value])[0]
		predicted_output_sine = (predicted_output_normalized * 2.0) - 1.0
		expected_output_sine = math.sin(float(integer_input))

		print(
			"x=",
			integer_input,
			"pred=",
			round(predicted_output_sine, 6),
			"target=",
			round(expected_output_sine, 6),
		)


def main():
	run_xor_simulation()
	run_sine_simulation()


if __name__ == "__main__":
	main()
