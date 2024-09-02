import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.table import Table


def calculate_pixel_sum(row):
    # Como a primeira coluna é a label e as outras 784 colunas são os valores dos pixels
    pixel_values = row[1:]  # Ignorar a primeira coluna (rótulo)
    return pixel_values.sum() / 255

def calculate_vertical_symmetry(row):
    """Calcula a simetria vertical para uma linha de pixels de 784 elementos."""
    # Reformate a linha em uma matriz 28x28
    image = row[1:].values.reshape(28, 28)
    
    # Número de colunas na imagem (28)
    num_cols = image.shape[1]
    
    # Calcular a diferença entre pixels simétricos
    symmetry_sum = 0
    for col in range(num_cols):
        symmetry_sum += np.sum(np.abs(image[:, col] - image[:, num_cols - col - 1]))
    
    return symmetry_sum/255


def calculate_horizontal_symmetry(row):
    """Calcula a simetria horizontal para uma linha de pixels de 784 elementos."""
    # Reformate a linha em uma matriz 28x28
    image = row[1:].values.reshape(28, 28)
    
    # Número de linhas na imagem (28)
    num_rows = image.shape[0]
    
    # Calcular a diferença entre pixels simétricos
    symmetry_sum = 0
    for row in range(num_rows):
        symmetry_sum += np.sum(np.abs(image[row, :] - image[num_rows - row - 1, :]))
    
    return symmetry_sum/255

def calculate_total_symmetry(row):
    """Calcula a soma das simetrias vertical e horizontal para uma linha de pixels de 784 elementos."""
    vertical_symmetry = calculate_vertical_symmetry(row)
    horizontal_symmetry = calculate_horizontal_symmetry(row)
    
    return vertical_symmetry + horizontal_symmetry

def plot(row, title=False):
    """Plota a imagem a partir de uma linha de pixels de 784 elementos."""
    # Reformatando a linha em uma matriz 28x28
    image = row[1:].values.reshape(28, 28)
    
    # Plotando a imagem
    plt.imshow(image, cmap='gray_r', vmin=0, vmax=255)
    if title:   
        plt.title(f'{title}')
    plt.axis('off')  # Desliga os eixos
    plt.show()


def plot_intensity_vs_symmetry(df, weights=None, save_path=False,title=False):
    """
    Plota um gráfico de dispersão de Intensidade vs. Simetria, colorido pelos labels,
    e salva o gráfico em um arquivo PNG. Opcionalmente, plota a reta de separação
    se os pesos forem fornecidos.

    Parâmetros:
    df (pd.DataFrame): O DataFrame que contém as colunas 'intensidade', 'simetria' e 'label'.
    weights (list ou np.array): Vetor de pesos do modelo de regressão linear [intercepto, coef_intensidade, coef_simetria].
    save_path (str): O caminho onde o gráfico será salvo como um arquivo PNG.
    """
    # Criando uma paleta de cores discreta para os labels
    unique_labels = df['label'].unique()
    palette = sns.color_palette("husl", len(unique_labels))  # Usa a paleta de cores Husl

    # Criando o gráfico usando seaborn
    plt.figure(figsize=(10, 6))

    # Plotando Intensidade vs. Simetria
    sns.scatterplot(x='intensidade', y='simetria', hue='label', data=df, palette=palette, edgecolor='k')

    if weights is not None:
        # Adicionando a reta de separação
        intercept = weights[0]
        coef_intensidade = weights[1]
        coef_simetria = weights[2]

        x_values = np.linspace(df['intensidade'].min(), df['intensidade'].max(), 100)
        y_values = - (coef_intensidade / coef_simetria) * x_values - (intercept / coef_simetria)

        plt.plot(x_values, y_values, color='black', linestyle='--', label='Reta')

    # Adicionando título e rótulos
    if title:   plt.title(f'{title}')
    plt.xlabel('Intensidade')
    plt.ylabel('Simetria')
    plt.legend(title='Label')

    # Salvando o gráfico como um arquivo PNG
    if save_path:
        plt.savefig(save_path)

    # Exibindo o gráfico
    plt.show()


def filter_and_transform_df(df, label1, label2):
    """
    Filtra um DataFrame para incluir apenas linhas com os labels fornecidos e
    substitui esses labels por 1 e -1, respectivamente.

    Parâmetros:
    df (pd.DataFrame): O DataFrame original com as colunas 'label', 'intensidade', 'simetria'.
    label1 (int): O primeiro label para manter e substituir por 1.
    label2 (int): O segundo label para manter e substituir por -1.

    Retorna:
    pd.DataFrame: Um novo DataFrame filtrado e com os labels transformados.
    """
    # Filtrar o DataFrame para incluir apenas os labels fornecidos
    filtered_df = df[df['label'].isin([label1, label2])].copy()

    # Substituir os labels
    filtered_df['label_to_calculate'] = filtered_df['label'].replace({label1: 1, label2: -1})

    return filtered_df

def filter_and_transform_df2(df, label1, label_list):
    """
    Filtra um DataFrame para incluir apenas linhas com os labels fornecidos e
    substitui esses labels por 1 para o label principal e -1 para os outros labels.

    Parâmetros:
    df (pd.DataFrame): O DataFrame original com as colunas 'label', 'intensidade', 'simetria'.
    label1 (int): O label principal para manter e substituir por 1.
    label_list (list): A lista de labels para substituir por -1.

    Retorna:
    pd.DataFrame: Um novo DataFrame filtrado e com os labels transformados.
    """
    # Filtrar o DataFrame para incluir apenas os labels fornecidos
    labels_to_keep = [label1] + label_list
    filtered_df = df[df['label'].isin(labels_to_keep)].copy()

    # Substituir os labels
    label_mapping = {label1: 1}
    label_mapping.update({label: -1 for label in label_list})
    filtered_df['label_to_calculate'] = filtered_df['label'].replace(label_mapping)

    return filtered_df


def acuracia(y1,y2):
    sum = 0
    for i in range(len(y2)):
        if y1[i]==y2[i]:
            sum+=1

    return sum/len(y2)*100

def plot_confusion_matrix(y_true, y_pred,cor, labels,true_labels, save_path=None):
    """
    Plota a matriz de confusão.

    Parâmetros:
    - y_true (np.ndarray): Rótulos verdadeiros.
    - y_pred (np.ndarray): Rótulos previstos.
    - labels (list): Lista de rótulos para exibição na matriz de confusão.
    - save_path (str, opcional): Caminho para salvar o arquivo da figura.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cor, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão - labels {true_labels[0]} e {true_labels[1]}')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Função para gerar tabela estilizada
def plot_accuracy_table(accuracies, labels, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_axis_off()
    table = Table(ax, bbox=[0, 0, 1, 1])

    # Configuração das colunas e linhas
    n_rows, n_cols = len(labels), 2
    width, height = 1.0 / n_cols, 1.0 / n_rows

    # Função para definir a cor com base na acurácia
    def get_color(accuracy):
        if accuracy >= 95:
            return 'lightgreen'
        elif accuracy >= 75:
            return 'lightyellow'
        else:
            return 'lightcoral'

    # Adicionando as células
    for i in range(n_rows):
        table.add_cell(i, 0, width, height, text=f'{labels[i][0]} vs {labels[i][1]}', loc='center', facecolor='lightblue')
        table.add_cell(i, 1, width, height, text=f'{accuracies[i]:.2f}', loc='center', facecolor=get_color(accuracies[i]))

    # Adicionando cabeçalhos
    table.add_cell(-1, 0, width, height, text='Labels', loc='center', facecolor='lightgray')
    table.add_cell(-1, 1, width, height, text='Acuracia', loc='center', facecolor='lightgray')

    ax.add_table(table)
    
    # Salvar o gráfico se um caminho de salvamento for fornecido
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

def plot_all_decision_boundaries(df, weights_list=None, title='', save_path=None):
    plt.figure(figsize=(8, 6))
    
    # Normalizar os dados de intensidade e simetria
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(df[['intensidade', 'simetria']])
    
    # Substituir os valores de intensidade e simetria com os valores normalizados
    df['intensidade_norm'] = X_normalized[:, 0]
    df['simetria_norm'] = X_normalized[:, 1]

    # Plotar os dados normalizados
    plt.scatter(df['intensidade_norm'], df['simetria_norm'], c=df['label'], cmap='viridis', edgecolors='k')

    if weights_list is not None:
        # Para cada conjunto de pesos, plotar a linha de decisão correspondente
        x_vals = np.linspace(df['intensidade_norm'].min(), df['intensidade_norm'].max(), 100)
        for i in range(len(weights_list)):
            weights = weights_list[i] 
            y_vals = -(weights[0] + weights[1] * x_vals) / weights[2]
            plt.plot(x_vals, y_vals, label=f'Passo {i+1}', linestyle='--')
    
    plt.title(title)
    plt.xlabel('Intensidade (Normalizada)')
    plt.ylabel('Simetria (Normalizada)')
    plt.xlim(df['intensidade_norm'].min() - 1, df['intensidade_norm'].max() + 1)
    plt.ylim(df['simetria_norm'].min() - 1, df['simetria_norm'].max() + 1)

    plt.legend()  # Adiciona uma legenda para identificar as retas de decisão
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_logistic_regression(X, y, label1, label2, model, filename=None):
    """
    Plota a fronteira de decisão de um modelo de regressão logística.

    Parâmetros:
    - X: np.ndarray, matriz de características.
    - y: np.ndarray, vetor de rótulos.
    - label1: int, rótulo da primeira classe.
    - label2: int, rótulo da segunda classe.
    - model: LogisticRegression, modelo treinado de regressão logística.
    - filename: str ou None, caminho para salvar o gráfico (opcional).
    """
    plt.figure(figsize=(8, 6))

    # Plotar os pontos de dados
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label=f'Label = {label1}', edgecolor='k')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label=f'Label = {label2}', edgecolor='k')

    # Criar uma grade de pontos para a decisão boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Previsões para cada ponto na grade
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z >= 0.5).astype(int)  # Convertendo para 0 ou 1
    Z = Z.reshape(xx.shape)

    # Contour plot da decisão boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr', levels=[-0.5, 0.5])
    plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=1.5)

    plt.title('Regressão Logística - Fronteira de Decisão')
    plt.xlabel('Intensidade')
    plt.ylabel('Simetria')
    plt.legend()

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        
def filter_and_transform_df2(df, label1, label_list):
    """
    Filtra um DataFrame para incluir apenas linhas com os labels fornecidos e
    substitui esses labels por 1 para o label principal e -1 para os outros labels.

    Parâmetros:
    df (pd.DataFrame): O DataFrame original com as colunas 'label', 'intensidade', 'simetria'.
    label1 (int): O label principal para manter e substituir por 1.
    label_list (list): A lista de labels para substituir por -1.

    Retorna:
    pd.DataFrame: Um novo DataFrame filtrado e com os labels transformados.
    """
    # Filtrar o DataFrame para incluir apenas os labels fornecidos
    labels_to_keep = [label1] + label_list
    filtered_df = df[df['label'].isin(labels_to_keep)].copy()

    # Substituir os labels
    label_mapping = {label1: 1}
    label_mapping.update({label: -1 for label in label_list})
    filtered_df['label_to_calculate'] = filtered_df['label'].replace(label_mapping)

    return filtered_df
