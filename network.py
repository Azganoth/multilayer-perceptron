'''
Classificação de vinho com rede neural

Autores: Ademir J. Ferreira Júnior (@Azganoth), José C. Pereira (@JPereira1330)

Amostras (Alcool, Ph, Açucar, Acidez)
Amostra 01 (SECO): 13.5%; 3.41; 5g/l; 8.85g/l.
Amostra 02 (SECO): 14.5%; 3.3; 2g/l; 5.92g/l.
Amostra 03 (DOCE): 11.5%; 3.63; 50g/l; 5.32g/l.
Amostra 04 (DOCE): 10%; 3.1; 50.7g/l; 6.45g/l.
Amostra 05 (DOCE): 10%; 3.1; 42,2g/l; 7.12g/l.
Amostra 06 (DOCE): 8.5%; 3.21; 42g/l; 8.9g/l.
Amostra 07 (DOCE): 8.5%; 3.31; 42g/l; 5.85g/l.
Amostra 08 (SECO): 14%; 3.68; 3.3g/l; 5.5g/l.
Amostra 09 (SECO): 14%; 3.32; 3.6g/l; 6.1g/l.
Amostra 10 (SECO): 14%; 3.51; 3.2g/l; 5.98g/l.

Amostra de Teste 01 (SECO): 13%; 3.45; 1.75g/l; 5.1g/l.
Amostra de Teste 02 (DOCE): 8.5%; 3.31; 42g/l; 5.85g/l.

O resultado varia entre 0 e 1,
sendo 0 o mais próximo de um vinho seco e 1 o mais próximo de um vinho doce.

Dados: http://www.casavalduga.com.br/produtos/vinhos/
Artigo: https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
'''
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
    [0.135, 0.341, 0.05, 0.885],
    [0.145, 0.33, 0.02, 0.592],
    [0.115, 0.363, 0.5, 0.532],
    [0.1, 0.31, 0.507, 0.645],
    [0.1, 0.31, 0.422, 0.712],
    [0.085, 0.321, 0.42, 0.89],
    [0.085, 0.331, 0.42, 0.585],
    [0.14, 0.368, 0.033, 0.55],
    [0.14, 0.332, 0.036, 0.61],
    [0.14, 0.351, 0.032, 0.598]
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

print('Teste 1: Vinho seco = 13%; 3.45; 1.75g/l; 5.1g/l')
print(f'Resultado: {neural_network.think(np.array([0.13, 0.345, 0.0175, 0.051]))[1]}')
print(f'Esperado: {[0]}')

print('\nTeste 2: Vinho doce = 8.5%; 3.31; 42g/l; 5.85g/l')
print(f'Resultado: {neural_network.think(np.array([0.085, 0.331, 0.42, 0.0585]))[1]}')
print(f'Esperado: {[1]}')
