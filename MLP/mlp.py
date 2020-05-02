# MLP "from scratch" feita por:
# Gabriel Furtado Lins Melo
# Universidade Estadual do Ceará (UECE)
# Laboratório de Matemática Computacional

import matplotlib.pyplot as plt
import numpy as np
import random
import os

class MLP:
    # Classe de Multi-Layer Perceptron
    # Parâmentros:
    #     input_dim: (int) tamanho da entrada da rede
    #     hidden_layers: (list) lista onde cada elemento representa a quantidade de neurônios de cada camada
    #     output_dim: (int) tamanho da saída da rede
    def __init__(
                self, 
                input_dim, 
                hidden_layers, 
                output_dim, 
                learning_rate=0.1, 
                momentum=0.5, 
                activation_function="sigmoid", 
                error_function="mse", 
                network_name="net_1"
                ):
        # Network architecture and weights:
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.layers = [self.input_dim] + self.hidden_layers + [self.output_dim]
        self.weights = []
        # Network parameters:
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.activation_function_name = activation_function
        self.activation_derivative = ""
        self.error_function = error_function
        self.error_function_name = error_function
        self.vec_error_function = ""
        # Network log for backpropagation:
        self.activations = []
        self.old_deltas = [] # Para o cálculo do momento
        self.deltas = []
        # Saving / loading configs and training results
        self.network_name = network_name
        self.save_path = "weights/"
        self.results_path = "results/"
        self.last_results = []
        self.last_accuracy = 0
        self.last_error = 0
        # Initialization functions
        self.initialize_weights()
        self.initialize_functions()

    def __call__(self, x):
        # Ao chamar o objeto de MLP criado como se fosse um método, o método forward
        # é aplicado. Ex.:
        # mlp = MLP(2,3,2)
        # saida_da_rede = mlp(amostras)
        return self.forward(x)

    # ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~
    # Funções de ativação:

    # ReLU para vetores
    def relu(self, X): return np.maximum(0, X).tolist()

    # Softmax para vetores
    def softmax(self, X):
        expo = np.exp(X)
        expo_sum = np.sum(expo)
        return (expo/expo_sum).tolist()

    # Sigmóide para números
    def _sigmoid(self, x): return 1/(1+np.exp(-x))
    def _sigmoid_derivative(self, x):
        s = self._sigmoid(x)
        return s*(1-s)
    # Sigmóide para vetores
    def sigmoid(self, X): return [self._sigmoid(x) for x in X]
    # Derivada da sigmóide
    def sigmoid_derivative(self, X): return [self._sigmoid_derivative(x) for x in X]

    # ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~
    # Funções de erro:

    # Mean Square Error entre 2 elementos
    def mse(self, target, prediction): return 0.5*(target-prediction)**2
    # Mean Square Error com n elementos
    def vec_mse(self, target, prediction): return np.square(np.subtract(target, prediction)).mean()

    # ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~
    # Métodos de inicialização

    def initialize_weights(self):
        # (RE)Inicializa aleatoriamente os pesos e os logs da rede. (Pode ser usado para resetar os pesos)
        # Uma rede com n cadamadas (incluindo as de entrada e saída) possui n-1 camadas de pesos
        # uma vez que, para cada par de camadas adjacentes, há 1 camada de peso.
        # Na criação dos pesos, para cada neurônio da próxima camada, é adicionado um valor de peso extra
        # que representa o bias (ou viés) do neurônio.
        self.weights = []
        self.activations = []
        self.deltas = []
        self.old_deltas = []
        for i in range(len(self.layers)-1):
            self.weights.append([[random.random() for j in range(self.layers[i] + 1)] for k in range(self.layers[i+1])])
        self.activations = [np.zeros(self.layers[i]).tolist() for i in range(len(self.layers))]
        self.deltas = [np.zeros(self.layers[i]).tolist() for i in range(len(self.layers))]
        for i in range(len(self.layers)-1):
            self.old_deltas.append([[0.0 for j in range(self.layers[i] + 1)] for k in range(self.layers[i+1])])

    def initialize_functions(self):
        if(self.activation_function=="sigmoid"):
            self.activation_function = self.sigmoid
            self.activation_derivative = self._sigmoid_derivative
        if(self.error_function=="mse"):
            self.error_function = self.mse
            self.vec_error_function = self.vec_mse

    # ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~
    # Métodos de forward e backward:

    def activate_neuron(self, weights, x):
        # Saída de determinado neurônio
        # Parâmetros:
        #     weights: (list ou np.ndarray) pesos + bias do neurônio a ser ativado
        #     x: (list ou np.ndarray) entrada (ou saída da camada anterior) 
        return np.matmul(weights[:-1], x)+weights[-1] # weights[-1]: bias

    def activate_layer(self, layer_weights, x):
        # Saída de determinada camada
        # Parâmetros:
        #     layer_weights: (list ou np.ndarray) lista de lista, contendo os vetores de pesos de cada neurônio da camada
        #     x: (list ou np.ndarray) entrada (ou saída da camada anterior)
        output = [self.activate_neuron(neuron_weights, x) for neuron_weights in layer_weights]
        return self.activation_function(output)

    def forward(self, x):
        # Saída da MLP. Retorna um np.ndarray de dimensão output_dim
        # Parâmetros:
        #     x: (list ou np.ndarray) de dimensão input_dim, contendo a entrada da rede (features da amostra)
        self.activations[0] = x
        for i, layer_weights in enumerate(self.weights):
            x = self.activate_layer(layer_weights, x)
            self.activations[i+1] = x
        return x

    def backpropagate(self, expected):
        # Método de backpropagate da rede
        # Utiliza a fórmula:
        #   dE/dW_(i-1) = dE/da_(i) * da_(i)/dh_(i) * dh_(i)/dW_(i-1)
        # com:
        #   - dE/da: Derivada* da função erro em relação à ativação
        #   - da/dh: Derivada* da ativação (função de ativação) em relação à entrada desta
        #   - dh/dW: Derivada* da da entrada em relação aos pesos
        # *Derivada Parcial, já que se trata de vetores (multivariáveis)
        for i in reversed(range(len(self.layers))):
            errors = []
            if(i != len(self.layers)-1):
                for j in range(self.layers[i]):
                    error = 0.0
                    for k in range(self.layers[i+1]):
                        error += (self.weights[i][k][j] * self.deltas[i+1][k])
                    errors.append(error)
            else:
                for j in range(self.layers[i]):
                    errors.append(expected[j] - self.activations[i][j])
            for j in range(self.layers[i]):
                self.deltas[i][j] = errors[j] * self.activation_derivative(self.activations[i][j])

    def update_weights(self):
        counter = 0
        for i in range(len(self.layers)-1):
            for n in range(self.layers[i+1]):
                for j in range(len(self.activations[i])):
                    delta = self.learning_rate * self.deltas[i+1][n] * self.activations[i][j]
                    self.weights[i][n][j] += delta + self.momentum * self.old_deltas[i][n][j]
                    self.old_deltas[i][n][j] = delta
                self.weights[i][n][-1] += self.learning_rate * self.deltas[i+1][n] # Bias update

    # ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~
    # Métodos de carregamento e armazenamento:

    def set_network_name(self, name):
        # Seta o nome dos pesos (utilizado ao salvar e carregar pesos)
        self.network_name = name

    def set_save_path(self, path):
        # Seta o caminho de diretório que os pesos serão salvos
        self.save_path = path

    def save(self):
        # Salva os pesos da rede em self.savepath+self.network_name
        if(self.network_name==""):
            print("Please set your weights name with instance.set_network_name(\"name\")")
            return
        if(not os.path.isdir(self.save_path)): os.mkdir(self.save_path)
        full_path = self.save_path + self.network_name + ".npy"
        np.save(full_path, [[self.network_name], self.weights], allow_pickle=True, fix_imports=True)
        print("Weights saved in: \"{}\"\n".format(full_path))

    def load_from_path(self, path):
        # Carrega, do caminho "path", pesos de uma rede
        loaded = np.load(path, allow_pickle=True)
        self.network_name, self.weights = loaded[0][0], np.array(loaded[1])
    
    def load(self, name):
        # Carrega, de self.save_path + name, os pesos de uma rede
        self.load_from_path(self.save_path + name + ".npy")

    # ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~
    # Métodos de treino e validação:

    # Função auxiliar para printar uma linha:
    def printline(self):
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    def index_of_max_value(self, vec):
        # Função auxiliar que retorna o índice com o maior valor de um array
        return [i for i, j in enumerate(vec) if j==max(vec)][0]

    def train(self, data_X, data_Y, epochs=100, save_results=True, save_name="results", save_weights=True):
        # Treinamento da rede
        # Parâmetros:
        #    data_X (list): lista contendo, em cada posição, as features das amostras
        #    data_Y (list): lista contendo, em cada posição, os rótulos (labels) das respectivas amostras
        #    epochs (int): Número de épocas de treinamento
        #    save_results (bool): Informa se salva, ou não, os resultados do treinamento
        #    save_name (str): O nome dos resultados a serem salvos
        #    save_weights (bool): Se deve salvar, ou não, os pesos     
        error = []
        accuracy = []
        trainset_size = len(data_X)

        self.printline()
        print("Train:\n")
        for epoch in range(1,epochs+1):
            err = 0.0
            acc = 0.0
            for i, x in enumerate(data_X):
                pred = self.forward(x)
                target = data_Y[i]
                err += self.vec_error_function(target, pred)
                if(self.index_of_max_value(pred) == self.index_of_max_value(target)): acc += 1
                self.backpropagate(target)
                self.update_weights()
            accuracy.append(float(acc)/trainset_size)
            err /= trainset_size
            error.append(err)
            if(epoch<epochs):
                print("Epoch: {}; Hits: {} of {}; Accuracy: {:.1f}%; Error: {:.5f}".format(epoch, acc, trainset_size, 100*accuracy[-1], err), end='\r')
            else: 
                print("Epoch: {}; Hits: {} of {}; Accuracy: {:.1f}%; Error: {:.5f}".format(epoch, acc, trainset_size, 100*accuracy[-1], err))
                self.printline()

        self.make_results(epochs, accuracy, error, trainset_size, len(data_X[0]), save_name)
        if(save_results): self.save_results(self.last_results, save_name)
        if(save_weights): self.save()

    def test(self, data_X, data_Y, _print=True):
        error = 0
        accuracy = 0
        testset_size = len(data_X)
        for i, x in enumerate(data_X):
            pred = self.forward(x)
            target = data_Y[i]
            error += self.vec_error_function(target, pred)
            if(self.index_of_max_value(pred) == self.index_of_max_value(target)): accuracy += 1
        error /= testset_size
        acc = accuracy/float(testset_size)
        if(_print):
            print("Test results:\n")
            print("- Hits: {} of {}".format(accuracy, testset_size))
            print("- Accuracy: {:.1f}%".format(100*acc))
            print("- Error: {:.5f}".format(error))
            self.printline()
        return accuracy, error

    # ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~
    # Manipulação dos resultados de treino:

    def make_results(self, epochs, accuracy, error, amount_of_data, data_size, save_name):
        # Formata os resultados
        results = []
        results.append("Network informations:")
        results.append(" - Network name: " + self.network_name)
        results.append(" - Network architecture: " + str(self.layers))
        results.append("")
        results.append("Trainset informations:")
        results.append(" - Amount of training samples: " + str(amount_of_data))
        results.append(" - Sample size: " + str(data_size))
        results.append("")
        results.append("Train parameters:")
        results.append(" - Epochs: " + str(epochs))
        results.append(" - Learning rate: " + str(self.learning_rate))
        results.append(" - Momentum: " + str(self.momentum))
        results.append(" - Activation function: " + self.activation_function_name)
        results.append(" - Error function: " + self.error_function_name)
        results.append("")
        results.append("Results:")
        results.append(" - Final accuracy: " + str(accuracy[-1]))
        results.append(" - Final error: " + str(error[-1]))
        results.append(accuracy)
        results.append(error)
        results.append(epochs)
        self.last_results = results
        self.last_accuracy = accuracy[-1]
        self.last_error = error[-1]

    def show_results(self, summarize=False):
        print("Train results:\n")
        if(summarize): 
            for line in self.last_results[-5:-3]: 
                print(line)
        else: 
            for line in self.last_results[:-3]: 
                print(line)
        self.printline()

    def plot_acc_err(self):
        plt.subplot(1,2,1)
        plt.plot(list(range(1,self.last_results[-1]+1)), self.last_results[-2], label="Erro")
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.subplot(1,2,2)
        plt.plot(list(range(1,self.last_results[-1]+1)), self.last_results[-3], label="Acurácia")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.subplots_adjust(wspace=0.4)

        plt.show()

    def plot_accuracy(self):
        self.plot_results(self.last_results[-1], self.last_results[-3], "Accuracy")

    def plot_error(self):
        self.plot_results(self.last_results[-1], self.last_results[-2], "Error")           

    def plot_results(self, epochs, y, label):
        # Função auxiliar para plotar resultados
        plt.plot(list(range(1,epochs+1)), y, label=label)
        plt.xlabel("Epochs")
        plt.ylabel(label)
        plt.legend()
        plt.show()

    def save_results(self, results, name):
        if(not os.path.isdir(self.results_path)): os.mkdir(self.results_path)
        path = self.results_path + self.network_name + "_" + name + ".npy"
        np.save(path, results, allow_pickle=True, fix_imports=True)
        print("Results saved in: \"{}\"".format(path))

    def load_results(self, file_name):
        path = self.results_path + file_name
        self.last_results = np.load(path, allow_pickle=True)

    # ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~
