import numpy as np

class NeuralNetwork:
    def __init__(self, n_neurons_input_layer, n_neurons_hidden_layer, n_neurons_output_layer):
        self.input_hidden_weights = np.random.rand(n_neurons_input_layer, n_neurons_hidden_layer)
        self.hidden_output_weights = np.random.rand(n_neurons_hidden_layer, n_neurons_output_layer)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def think(self, inputs):
        output_from_hidden_layer = self.sigmoid(np.dot(inputs, self.input_hidden_weights))
        output_from_output_layer = self.sigmoid(np.dot(output_from_hidden_layer, self.hidden_output_weights))
        return output_from_hidden_layer, output_from_output_layer

    def train(self, training_inputs, training_outputs, epochs):
        for epoch in range(epochs):
            output_from_hidden_layer, output_from_output_layer = self.think(training_inputs)

            output_layer_error = training_outputs - output_from_output_layer
            output_layer_delta = output_layer_error * self.sigmoid_derivative(output_from_output_layer)

            hidden_layer_error = output_layer_delta.dot(self.hidden_output_weights.T)
            hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(output_from_hidden_layer)

            self.input_hidden_weights += training_inputs.T.dot(hidden_layer_delta)
            self.hidden_output_weights += output_from_hidden_layer.T.dot(output_layer_delta)

training_inputs = np.array([
    [0.135, 0.0341, 0.050, 0.885],
    [0.145, 0.0330, 0.020, 0.592],
    [0.115, 0.0363, 0.500, 0.532],
    [0.100, 0.0310, 0.507, 0.645],
    [0.100, 0.0310, 0.422, 0.712],
    [0.085, 0.0321, 0.420, 0.890],
    [0.085, 0.0331, 0.420, 0.585],
    [0.140, 0.0368, 0.033, 0.550],
    [0.140, 0.0332, 0.036, 0.610],
    [0.140, 0.0351, 0.032, 0.598]
])

training_outputs = np.array([
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [0],
    [0],
    [0]
])

neural_network = NeuralNetwork(4, 4, 1)

print(f'Pesos entre a camada input e hidden antes do treino:\n{neural_network.input_hidden_weights}\n')
print(f'Pesos entre a camada hidden e output antes do treino:\n{neural_network.hidden_output_weights}\n')

neural_network.train(training_inputs, training_outputs, 60000)

print(f'Pesos entre a camada input e hidden depois do treino:\n{neural_network.input_hidden_weights}\n')
print(f'Pesos entre a camada hidden e output depois do treino:\n{neural_network.hidden_output_weights}\n')

test_input_index = 0
print(f'Resultado: {neural_network.think(training_inputs[test_input_index])}.')
print(f'Esperado: {training_outputs[test_input_index]}.')
