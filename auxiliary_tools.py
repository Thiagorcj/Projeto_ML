import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix

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