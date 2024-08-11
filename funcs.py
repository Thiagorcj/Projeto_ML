import numpy as np

class LinearRegression:

    def __init__(self):
        self.w = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        num_samples = X.shape[0]
        
        # Adiciona uma coluna de 1s para o termo de interceptação
        X = np.column_stack((np.ones(num_samples), X))
    
        # Calcula o vetor de pesos w usando a fórmula
        xt = np.transpose(X)
        xtx = np.dot(xt, X)
        xty = np.dot(xt, y)
        self.w = np.dot(np.linalg.inv(xtx), xty)

    def predict(self, X):
        X = np.array(X)
        num_samples = X.shape[0]
        X = np.column_stack((np.ones(num_samples), X))
        return np.dot(X, self.w) #Produto interno do vetor X com w

    def getW(self):
        return self.w

# Exemplo de uso:            
'''
X = _time
y = _data
lr = LinearRegression()
lr.fit(X,y)
ws = lr.getW()
print(f'Percebe-se que nosso w0 (coeficiente linear da reta) é {ws[0]:.3f} e que nosso w1 (coeficiente angular da reta) é {ws[1]:.3f}.\n')
predicts = lr.predict(X)
print(f'A título de curiosidade as previsões seguindo a reta seriam:\n')
prints = [print(f'{predict:.3f}') for predict in predicts]
'''

def constroiListaPCI(X, y, w):
    """
    Esta função constrói a lista de pontos classificados incorretamente.

    Paramêtros:
    - X (list[]): Matriz correspondendo aos dados amostra. Cada elemento de X é uma lista que corresponde
    às coordenadas dos pontos gerados.
    - y (list): Classificação dos pontos da amostra X.
    - w (list): Lista correspondendo aos pesos do perceptron.

    Retorno:
    - l (list): Lista com os pontos classificador incorretamente.
    - new_y (list): Nova classificação de tais pontos.

    """
    l = []
    new_y = []
    for i in range(len(X)):
      xi = X[i][0]
      yi = X[i][1]

      new_yi = np.sign(w[2]*yi + w[1]*xi + w[0])
      if (new_yi != y[i]):
        l.append(X[i])
      new_y.append(new_yi)

    return l, new_y

def PLA(X, y, f):
    """
    Esta função corresponde ao Algoritmo de Aprendizagem do modelo Perceptron.

    Paramêtros:
    - X (list[]): Matriz correspondendo aos dados amostra. Cada elemento de X é uma lista que corresponde
    às coordenadas dos pontos gerados.
    - y (list): Classificação dos pontos da amostra X.
    - f (list): Lista de dois elementos correspondendo, respectivamente, aos coeficientes angular e linear
    da função alvo.
    Retorno:
    - it (int): Quantidade de iterações necessárias para corrigir todos os pontos classificados incorretamente.
    - w (list): Lista de três elementos correspondendo aos pesos do perceptron.
    """

    listaPCI = X
    it = 0
    w = [0,0,0]

    new_y = y
    while (len(listaPCI) > 0):
        numero_aleatorio = random.randint(0, len(listaPCI)-1)
        ponto = listaPCI[numero_aleatorio]


        # Índice do ponto na lista original X
        indice_ponto = np.where((X == ponto).all(axis=1))[0][0]

        # Atualiza os pesos
        w[0] += y[indice_ponto]
        w[1] += ponto[0] * y[indice_ponto]
        w[2] += ponto[1] * y[indice_ponto]



        listaPCI, _ = constroiListaPCI(X, new_y, w)

        it+=1

    return it, w
    
#Exemplo de uso
'''
it,w = PLA(pontos, labels, [m, b])
print("Quantidade de iterações: ", it)
print("Pesos: ", w)
'''
