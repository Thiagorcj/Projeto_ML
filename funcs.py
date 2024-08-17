import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
class Perceptron:

  def __init__(self,initial_w=None):
    if initial_w is None:
      self.w = [0,0,0]
    else:
      self.w = initial_w
      
  def constroiListaPCI(self,X, y, w):
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

        new_yi = np.sign(self.w[2]*yi + self.w[1]*xi + self.w[0])
        if (new_yi != y[i]):
          l.append(X[i])
        new_y.append(new_yi)

      return l, new_y

  def fit(self, X, y):
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

      listaPCI,_ = self.constroiListaPCI(X, y, self.w)
      it = 0

      new_y = y
      while (len(listaPCI) > 0):
          numero_aleatorio = random.randint(0, len(listaPCI)-1)
          ponto = listaPCI[numero_aleatorio]


          # Índice do ponto na lista original X
          indice_ponto = np.where((X == ponto).all(axis=1))[0][0]

          # Atualiza os pesos
          self.w[0] += y[indice_ponto]
          self.w[1] += ponto[0] * y[indice_ponto]
          self.w[2] += ponto[1] * y[indice_ponto]



          listaPCI, _ = self.constroiListaPCI(X, new_y, self.w)

          # Após atualizar os pesos para correção do ponto escolhido, você irá chamar a função plotGrafico()
          # plot_grafico(X, y, w, f)
          it+=1

      return it, self.w

  def predict(self,X):
    results = []
    for i in range(len(X)):
        results.append(np.sign(self.w[2]*X[i][1] + self.w[1]*X[i][0] + self.w[0]))
        
    return results

  def plot(self, X, y,label1,label2, savepath=None, title='Perceptron - Separação de Classes',scale=1.0):
      """
      Plota os pontos e a linha separadora.

      Parâmetros:
      - X (np.ndarray): Dados de entrada.
      - y (np.ndarray): Rótulos de classe (1 ou -1).
      - savepath (str, opcional): Caminho para salvar o arquivo da figura.
      - title (str, opcional): Título do gráfico.
      - scale (float, opcional): Fator de escala para ajustar o tamanho da imagem.
      """
      X = np.array(X)
      y = np.array(y)
      
      X_pos = X[y == 1]
      X_neg = X[y == -1]
      
      # Definir a paleta de cores 'husl'
      sns.set_palette('husl')
      
      plt.figure(figsize=(8*scale, 6*scale))  # Ajusta o tamanho da figura
      
      plt.scatter(X_pos[:, 0], X_pos[:, 1], color=sns.color_palette()[0], label=label1, alpha=0.7)
      plt.scatter(X_neg[:, 0], X_neg[:, 1], color=sns.color_palette()[1], label=label2, alpha=0.7)
      
      # Coeficientes da reta
      if self.w is not None and len(self.w) > 1:
          x_values = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
          y_values = (-self.w[0] - self.w[1] * x_values) / self.w[2]
          
          plt.plot(x_values, y_values, color='black', label='Reta')
      
      plt.xlabel('Intensidade')
      plt.ylabel('Simetria')
      plt.title(title)
      plt.legend()
      plt.grid(True)
      
      # Ajustar limites dos eixos com base na escala
      x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
      y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
      
      plt.xlim(x_min - (x_max - x_min) * 0.1 * scale, x_max + (x_max - x_min) * 0.1 * scale)
      plt.ylim(y_min - (y_max - y_min) * 0.1 * scale, y_max + (y_max - y_min) * 0.1 * scale)
      
      # Salvar o gráfico se o caminho for fornecido
      if savepath:
          plt.savefig(savepath)
      else:
          plt.show()

#Exemplo de uso:          
"""
filtered_df = filter_and_transform_df(df_train, 0, 1)
X = filtered_df[['intensidade', 'simetria']].to_numpy()
y = filtered_df['label_to_calculate'].to_numpy()
p = Perceptron()
it,w = p.fit(X,y)

print("Quantidade de iterações: ", it)
print("Pesos: ", w)
"""

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def logistic(self, s):
        return np.exp(s) / (1 + np.exp(s))

    def compute_gradient(self, X, y, y_pred):
        num_samples = X.shape[0]
        dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / num_samples) * np.sum(y_pred - y)
        return dw, db

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.logistic(linear_model)
            dw, db = self.compute_gradient(X, y, y_pred)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self.weights

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.logistic(linear_model)
        return y_pred