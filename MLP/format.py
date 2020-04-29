import random

def format_csv_line(line, dtype=str): 
# Função para formatar as linhas de um arquivo .csv
    if(dtype == str):
        line = line.replace('\n','').replace('"','').split(',')[:-1]
    else:
        line = format_csv_line(line)
        line = [dtype(i) for i in line]
    return line

def format_func(file_path):
# Retorna a função responsável pela conversão de linhas de um arquivo
    if(file_path.endswith(".csv")):
        return format_csv_line
    else: 
        print("Unknown data format type!")
        exit(1)

def pretty_print_matrix(matrix):
# Mostra uma matriz de maneira organizada no terminal
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print ('\n'.join(table))

def shuffle_together(a, b):
# Embaralha dois conjuntos (a e b) juntos (mantendo a ordem relativa dos elementos igual nos conjuntos)
    c = list(zip(a,b))
    random.shuffle(c)
    a[:], b[:] = zip(*c)
    return a, b

