from format import *

class Dataset:
    def __init__(self):
        self.x_titles = [] # Títulos das features
        self.y_titles = [] # Títulos dos rótulos
        self.train_set_X = [] # Conjunto de treino
        self.train_set_Y = [] # Rótulos do conjunto de treino
        self.test_set_X = [] # Conjunto de teste
        self.test_set_Y = [] # Rótulos do conjunto de teste
        self.train_size = lambda : len(self.train_set_X) # Tamanho do conjunto de dados
        self.test_size = lambda : len(self.test_set_X) # Tamanho do conjunto de dados
    
    def load_titles(self, file_path, num_labels=0):
    # Carrega os títulos das amostras e dos rótulos dos dados.
    # Parâmetros:
    #    file_path: (str ou list de tamanho 2) contém o caminho até o arquivo, ou array com [caminho_dados_X, caminho_dados_Y]
    #    num_labels: (int) contém a quantidade de labels (targets) das amostras. Só deve ser especificado caso seja arquivo único.
    # * Importante: No caso de um arquivo único, é considerado que as "num_labels" últimas features de cada amostra são os rótulos.
        if(type(file_path)==str):
            if(num_labels == 0):
                print("You must specify the number of labels!")
                exit(1)
            format_line = format_func(file_path)
            with open(file_path, 'r') as f: self.x_titles = format_line(f.readlines()[0])
            self.y_titles = self.x_titles[-num_labels:]
            self.x_titles = self.x_titles[:-num_labels]
        elif(type(file_path)==list and len(file_path)==2):
            format_line = format_func(file_path[0])
            with open(file_path[0], 'r') as f: self.x_titles = format_line(f.readlines()[0])
            with open(file_path[1], 'r') as f: self.y_titles = format_line(f.readlines()[0])
        else:
            print("file_path must be str containing the path to the entire dataset")
            print("or a list containing X data in file_path[0] and Y data in file_path[1].")
            exit(1)

    def load_data(self, file_path, data_type=float, skip_rows=0, num_labels=0, load_titles=False, test_set=False):
    # Função que carrega os dados do dataset (X e Y) em train_set_X e train_set_Y. Para dividí-los, utilizar split() ou load_test().
    # Parâmetros:
    #   file_path: (str ou list de tamanho 2) contém o(s) caminho(s) até o(s) arquivo(s) com os dados (juntos ou X e Y).
    #       file_path pode ser "dados/dados_juntos.csv" ou ["dados/amostras.csv", "dados/rotulos.csv"] **Nesta ordem, no caso de 2 arquivos**
    #   data_type: (<class 'type'>) tipo dos dados (para conversão no carregamento)
    #   skip_rows: (int) pula as primeiras "skip_rows" linhas do arquivo na obtenção dos dados.
    #   num_labels: (int) informa a quantidade de rótulos (labels) as amostras podem ter.
    #   load_titles: (bool) informa se deve, também, carregar os títulos das amostras do(s) arquivo(s).
    #   test_set: (bool) informa se está pegando o conjunto de teste.
    # * load_titles Considera que os títulos estão na primeira linha do arquivo.
        if(test_set): destiny_X, destiny_Y = self.test_set_X, self.test_set_Y
        else: destiny_X, destiny_Y = self.train_set_X, self.train_set_Y
        if(load_titles): 
            if(skip_rows==0):
                print("Please specify the parameter \"skip_rows\" for data obtaining!")
                exit(1)
            else: self.load_titles(file_path, num_labels)
        if(type(file_path)==str):
            if(num_labels == 0):
                print("Please specify the number of labels \"num_labels\".")
                print("The labels must be in the \"num_labels\" last rows of your raw data.")
                exit(1)
            format_line = format_func(file_path)
            with open(file_path, 'r') as f: lines = f.readlines()[skip_rows:]
            for line in lines:
                destiny_X.append(format_line(line, data_type)[:-num_labels])
                destiny_Y.append(format_line(line, data_type)[-num_labels:])
        elif(type(file_path)==list and len(file_path)==2):
            format_line = format_func(file_path[0])
            with open(file_path[0], 'r') as f: destiny_X = [format_line(line, data_type) for line in f.readlines()[skip_rows:]]
            with open(file_path[1], 'r') as f: destiny_Y = [format_line(line, data_type) for line in f.readlines()[skip_rows:]]
        else:
            print("file_path must be str containing the path to the entire dataset")
            print("or a list containing X data in file_path[0] and Y data in file_path[1].")
            exit(1)

    def load_test(self, file_path, data_type=float, skip_rows=0, num_labels=0, load_titles=False):
    # Carrega os dados de teste no conjunto de teste.
    # Parâmetros: iguais aos de load_data.
        load_data(self, file_path, data_type=data_type, skip_rows=skip_rows, num_labels=num_labels, load_titles=load_titles, test_set=True)

    def show_samples(self, indexes, train_or_test="train"):
    # Mostra, de maneira organizada, amostras do dataset (indexes é uma 'list' com os índices das amostras)
    # Parâmetros:
    #   indexes: (list) contendo os índices das amostras que deseja visualizar
    #   train_or_test: (str) pode ser "train" ou "teste". Informa de que conjunto pegar as amostras.
        if(train_or_test == "train"): destiny_X, destiny_Y = self.train_set_X, self.train_set_Y
        elif(train_or_test == "test"): destiny_X, destiny_Y = self.test_set_X, self.test_set_Y
        else:
            print("\"train_or_test\" must be \"train\" or \"test\"!")
            return
        if(len(destiny_X)==0):
            print("Your " + train_or_test + " set is empty!")
            return
        if(self.x_titles != [] and self.y_titles != []): matrix = [['sample'] + self.x_titles + self.y_titles]
        else: matrix = []
        for i in indexes: matrix.append([i] + self.train_set_X[i] + self.train_set_Y[i])
        pretty_print_matrix(matrix)

    def split(self, percentage=0.7):
    # Divide o dataset em treino e teste. Pega os dados do train_set
    # Parâmetros:
    #   percentage: (float) valor entre 0 e 1, que define a divisão como: percentage para treino e (1-percentage) para teste
        index = int(self.train_size()*percentage)
        self.test_set_X = self.train_set_X[index:]
        self.test_set_Y = self.train_set_Y[index:]
        self.train_set_X = self.train_set_X[:index]
        self.train_set_Y = self.train_set_Y[:index]

    def shuffle(self):
    # Embaralha o dataset
        if(self.train_size() != 0): self.train_set_X, self.train_set_Y = shuffle_together(self.train_set_X, self.train_set_Y)
        if(self.test_size() != 0): self.test_set_X, self.test_set_Y = shuffle_together(self.test_set_X, self.test_set_Y)