import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np
from sklearn.utils import shuffle   
from sklearn.metrics.pairwise import euclidean_distances

def display_general_info(df):
    """
    Exibe informações gerais e estatísticas básicas do DataFrame.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.

    Retorna:
    None
    """
    print("Informações gerais:")
    print(df.info())
    print("\nEstatísticas descritivas:")
    print(df.describe())

def check_missing_data(df):
    """
    Verifica e visualiza dados ausentes no DataFrame.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.

    Retorna:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Mapa de Dados Ausentes')
    plt.show()

def plot_variable_distributions(df, columns):
    """
    Plota as distribuições das variáveis espectrais.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.
    columns (list): Lista de colunas para plotar as distribuições.

    Retorna:
    None
    """
    df[columns].hist(bins=30, figsize=(30, 20))
    plt.suptitle('Distribuições das Variáveis Espectrais')
    plt.show()

def plot_correlations(df):
    """
    Plota a matriz de correlações entre as variáveis espectrais.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.

    Retorna:
    None
    """
    plt.figure(figsize=(30, 20))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=False, fmt='.2f', cmap='coolwarm')
    plt.title('Matriz de Correlações')
    plt.show()

def plot_spectral_data(df, wavelength_columns, title='Espectros Médios e Variabilidade'):
    """
    Plota os espectros médios e variabilidade.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.
    wavelength_columns (list): Lista de colunas que representam as variáveis espectrais (comprimentos de onda).

    Retorna:
    None
    """
    mean_spectrum = df[wavelength_columns].mean()
    std_spectrum = df[wavelength_columns].std()

    plt.figure(figsize=(14, 7))
    plt.plot(wavelength_columns, mean_spectrum, label='Espectro Médio', color='blue')
    plt.fill_between(wavelength_columns, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, color='blue', alpha=0.2, label='Variabilidade (±1 std)')
    plt.title(title)
    plt.xlabel('Comprimento de Onda')

    plt.xticks(np.arange(0, len(wavelength_columns)+1, 5), rotation=45, ha='right')
    plt.ylabel('Intensidade')
    plt.legend()
    plt.show()

def plot_individual_spectra(df, wavelength_columns):
    """
    Plota os espectros individuais para todas as linhas no DataFrame.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.
    wavelength_columns (list): Lista de colunas que representam as variáveis espectrais (comprimentos de onda).

    Retorna:
    None
    """
    plt.figure(figsize=(14, 7))

    for idx in df.index:
        plt.plot(wavelength_columns, df.loc[idx, wavelength_columns], label=f'Espectro {idx}')

    plt.title('Espectros Individuais')
    plt.xlabel('Comprimento de Onda')
    plt.ylabel('Intensidade')
    #plt.legend()
    plt.show()

def kennard_stone_split(X=None, y=None, sample_id=None, test_size=0.3, random_state=None, stratify=None):
    if stratify is not None:
        unique_classes, y_indices = np.unique(stratify, return_inverse=True)
        train_indices, test_indices = [], []

        for cls in unique_classes:
            class_indices = np.where(y_indices == cls)[0]
            X_class = X[class_indices]

            #if y is not None:
            #    y_class = y[class_indices]
            #else:
            #    y_class = None

            ks_train_indices, ks_test_indices = kennard_stone_indices(X_class, test_size)

            train_indices.extend(class_indices[ks_train_indices])
            test_indices.extend(class_indices[ks_test_indices])
    else:
        train_indices, test_indices = kennard_stone_indices(X, test_size)

    # Shuffle the indices
    if random_state is not None:
        np.random.seed(random_state)

    train_indices = shuffle(train_indices, random_state=random_state)
    test_indices = shuffle(test_indices, random_state=random_state)

    # Create the train-test split
    X_train, X_test = X[train_indices], X[test_indices]
    train_label_id, test_label_id = sample_id[train_indices], sample_id[test_indices]

    if y is not None:
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test, train_label_id, test_label_id
    else:
        return X_train, X_test, train_label_id, test_label_id


def insert_results_subpath(file_path):
    """
    Modify a file path by inserting "results" as a subdirectory before the file name.

    Parameters:
        file_path (str): The original file path.

    Returns:
        str: The modified file path with "results" added as a subdirectory.
    """
    # Get the directory and file name from the original path
    dir_name, file_name = os.path.split(file_path)

    # Create the new path with the "results" subdirectory
    new_path = os.path.join(dir_name, "results", file_name.split(".")[0])

    return new_path

def kennard_stone_indices(X, test_size=0.3):
    n_samples = X.shape[0]

    if isinstance(test_size, float):
        n_test = int(np.ceil(test_size * n_samples))
    else:
        n_test = test_size

    n_train = n_samples - n_test

    # Step 1: Compute pairwise distances
    distances = euclidean_distances(X, X)

    # Step 2: Select the first point which is the one farthest from the center (mean)
    mean_point = np.mean(X, axis=0)
    initial_index = np.argmax(np.linalg.norm(X - mean_point, axis=1))

    train_indices = [initial_index]
    remaining_indices = list(range(n_samples))
    remaining_indices.remove(initial_index)

    # Step 3: Iteratively select points that are farthest from the already selected points
    for _ in range(1, n_train):
        dist_to_selected = np.min(distances[train_indices, :], axis=0)
        next_index = remaining_indices[np.argmax(dist_to_selected[remaining_indices])]
        train_indices.append(next_index)
        remaining_indices.remove(next_index)

    # The remaining points are used as the test set
    test_indices = remaining_indices

    return train_indices, test_indices

def load_data(file_path):
    """
    Carrega a base de dados espectrais a partir de um arquivo CSV.

    Parâmetros:
    file_path (str): Caminho para o arquivo CSV.

    Retorna:
    DataFrame: DataFrame contendo os dados carregados.
    """
    df = pd.read_csv(file_path)
    df = df.fillna(0)
    return df

def plot_performance_comparison(data):
    """
    Plots a bar chart to compare the performance of pipelines using the original columns.

    The plot combines "LV" and "Outliers (c)" to create a more descriptive x-axis for the pipelines.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the columns ["LV", "Outliers (c)", "Pipeline", "Performance"].
    """
    # Ensure the required columns are in the DataFrame
    required_columns = ["LV", "Outliers (c)", "Pipeline", "Performance"]
    if not set(required_columns).issubset(data.columns):
        raise ValueError(f"The DataFrame must contain the following columns: {required_columns}")

    # Create a new column for combined LV and Outliers description
    data['Combination'] = "Pipeline: " + data['Pipeline'].astype(str) + " | LV: " + data['LV'].astype(str) + " | Outliers: " + data['Outliers (c)'].astype(str)

    # Create the bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=data,
        x="Pipeline",
        y="Performance",
        hue="Combination",
    )

    # Customize plot appearance
    plt.title("Performance Comparison of Pipelines", fontsize=16)
    plt.xlabel("Pipeline (Combination of LV and Outliers)", fontsize=14)
    plt.ylabel("Performance", fontsize=14)
    #plt.legend(title="Combination", title_fontsize=12, fontsize=10)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.show()
