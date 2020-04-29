import dataset as ds
import mlp
import numpy as np

# ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~
# Parâmetros de Dataset:
file_path = "dados/conservante_bebidas.csv" # Caminho do dataset
skip_rows = 1 # Número de linhas a pular para carregar os dados brutos do arquivo com os dados (ex: pular a linha que contém o título)
split_percentual = 0.8 # Divisão treino/teste
num_classes = 3 # Quantidade de classes (rótulos) do dataset
load_titles = True # Informa se quer carregar os títulos dos dados (para um print melhor das amostras e rótulos)

# Parâmetros da arquitetura da rede:
hidden_layers = [15] # A quantidade de números no array é a qtd de camadas escondidas, cada número representa a qtd de neurônios
learning_rate = 0.1 # Taxa de aprendizagem
momentum = 0.5 # Momento
activation_function = "sigmoid" # Função de ativação
error_function = "mse" # Função de erro
network_name = "net_1" # Nome da rede (para salvar nos resultados)

# Parâmetros de treinamento:
epochs = 100 # Quantidade de épocas
save_results = True # Se deseja salvar os resultados
save_weights = True # Se deseja salvar os pesos
summarize = False # Resumir os resultados de treinamento
plot_results = True # Mostrar os gráficos dos resultados de treino

# ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~
# Carregando os dados:
dataset = ds.Dataset()
dataset.load_data(
                file_path, # Caminho para o dataset
                skip_rows=skip_rows, # Número de linhas a pular quando for carregar as amostras
                num_labels=num_classes, # Quantidade de rótulos (para já separar X e Y)
                load_titles=load_titles # Para carregar os títulos das features e labels (1a linha do arquivo com os dados)
                )
dataset.shuffle() # Embaralhando os dados
dataset.split(split_percentual) # Fazendo a divisão
data_X = dataset.train_set_X # Conjunto de amostras de treino
data_Y = dataset.train_set_Y # Conjunto de rótulos de treino
input_dim = len(data_X[0]) # Pega uma amostra e seta o tamanho de entrada para a rede

# ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~
# Criando a rede:
net = mlp.MLP(
	input_dim=input_dim, 
	hidden_layers=hidden_layers, 
	output_dim=num_classes, 
	learning_rate=learning_rate, 
	momentum=momentum, 
	activation_function=activation_function, 
	error_function=error_function, 
	network_name=network_name
	)

# ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~
# Treinamento e teste:
net.train(data_X, data_Y, epochs, save_results=save_results, save_weights=save_weights)
net.show_results(summarize)
net.test(dataset.test_set_X, dataset.test_set_Y)
if(plot_results): net.plot_acc_err()

# ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~ x ~~~~~~~~~~~~~~~