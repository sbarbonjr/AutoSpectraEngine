import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split, cross_val_score

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, confusion_matrix
from sklearn.cross_decomposition import PLSRegression

import seaborn as sns

def plot_PCA(X, Y, combinacao):
    # Perform PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    # Get explained variance ratios
    explained_variance = pca.explained_variance_ratio_

    # Plot using matplotlib
    plt.figure(figsize=(10, 8))

    # Check if Y is numeric or categorical
    if np.issubdtype(Y.dtype, np.number):
        # Y is numeric (continuous values)
        scatter = plt.scatter(components[:, 0], components[:, 1], c=Y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Target Variable (Numeric)')
    else:
        # Y is categorical (labels)
        categories = np.unique(Y)
        colors = plt.cm.get_cmap('viridis', len(categories))
        for i, category in enumerate(categories):
            plt.scatter(components[Y == category, 0], components[Y == category, 1], color=colors(i), label=str(category))
        plt.legend(title='Target Variable (Categorical)')

    file_name_no_ext = os.path.splitext(file)[0]

    # Set labels and title with variance explained
    plt.xlabel(f'PC1 Variance Explained: {explained_variance[0]*100:.2f}%')
    plt.ylabel(f'PC2 Variance Explained: {explained_variance[1]*100:.2f}%')
    plt.title('PCA Plot:'+combinacao)
    plt.savefig(os.path.join(PATH, f'PCA {file_name_no_ext}_{pre_processamento}'), format='png')
    plt.show()
    plt.close()

def get_plsda_performance(X, y, preprocess_name, test_size=0.30, max_components=20, n_splits=10, label_espectro=[], plotar_plsda=False, x_scale=20, file_name_no_ext="experimento_"):

    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        nan_indices_X = X.isnull().any(axis=1)  # Identifica linhas com qualquer NaN em X
    else:
        nan_indices_X = np.any(np.isnan(X), axis=1)

    # Considere '0', strings vazias e NaN como rótulos inválidos
    nan_indices_y = pd.isnull(y) | (y == 0) | (y == "")  # Exclui valores 0, strings vazias e NaNs

    # Identificar amostras que contêm NaN ou valores inválidos tanto em X quanto em y
    nan_indices = np.logical_or(nan_indices_X, nan_indices_y)

    # Remover amostras com NaN ou valores inválidos em X ou em y
    X_clean = X[~nan_indices]
    y_clean = y[~nan_indices]

    # Verificar se ainda existem dados após remover as amostras inválidas
    if len(X_clean) == 0 or len(y_clean) == 0:
        print("Nenhuma amostra válida após remover as linhas com NaN ou valores inválidos.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Garantir que não exista nenhum valor que não deveria estar no vetor de rótulos
    y_clean = [label for label in y_clean if pd.notnull(label) and label != 0 and label != ""]  # Remover qualquer valor indesejado

    # Certifique-se de que os valores de y_clean são strings ou números
    y_clean = [str(label) for label in y_clean]  # Converter todos os rótulos para strings

    # Encode labels using LabelEncoder
    le = LabelEncoder()
    y_clean = le.fit_transform(y_clean)  # Codificar os rótulos restantes

    # Verificar e classificar os rótulos originais
    try:
      sorted_labels = sorted(le.classes_, key=lambda x: float(x))
    except ValueError:
      sorted_labels = sorted(le.classes_)
    sorted_indices = np.argsort([le.transform([label])[0] for label in sorted_labels])


    #sorted_indices = np.argsort(le.transform(le.classes_))
    #print('le.classes_[sorted_indices]', le.classes_[sorted_indices])
    #sorted_labels = le.classes_[sorted_indices]  # Mantém os rótulos originais em ordem


    #print(f"Rótulos labels: {sorted_labels}")
    #print(f"Rótulos index: {sorted_indices}")

    # Divida os dados em conjuntos de treino e teste
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=test_size, random_state=42, stratify=y_clean)
    except ValueError as e:
        print(f"Erro durante a separação treino-teste: {e}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    ohe = OneHotEncoder(sparse_output=False)  # Corrigido aqui
    y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))

    accuracy = []
    rmsecv = []

    for n_components in range(1, max_components + 1):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_rmse = []

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train_ohe[train_index], y_train[val_index]

            plsda = PLSRegression(n_components=n_components)
            plsda.fit(X_train_fold, y_train_fold)
            y_pred = plsda.predict(X_val_fold)
            y_pred_class = np.argmax(y_pred, axis=1)

            fold_rmse.append(np.sqrt(mean_squared_error(y_val_fold, y_pred_class)))

        rmsecv.append(np.mean(fold_rmse))


    best_n_components = np.argmin(rmsecv) + 1

    plsda_best = PLSRegression(n_components=best_n_components, scale=False)
    plsda_best.fit(X_train, y_train_ohe)
    y_pred_test = plsda_best.predict(X_test)
    y_pred_test_class = np.argmax(y_pred_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred_test_class)

    if plotar_plsda:
        # Gráfico de RMSECV por LV
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_components + 1), rmsecv, marker='o', linestyle='-', color='blue', label='RMSECV')
        plt.axvline(x=best_n_components, color='red', linestyle='--', label=f'Best LV: {best_n_components}')
        plt.xlabel('Number of Components', fontsize=15)
        plt.ylabel('RMSECV', fontsize=15)
        plt.title(f'{file_name_no_ext}-{y.name} - RMSECV by LV - {preprocess_name}', fontsize=16)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f'PLSDA{file_name_no_ext}-{y.name} - RMSECV by LV - {preprocess_name}', format='png')
        plt.show()
        plt.close()

        # Gráfico Confusion matriz
        cm = confusion_matrix(y_test, y_pred_test_class)
        report = classification_report(y_test, y_pred_test_class, target_names=sorted_labels, output_dict=True)

        print(accuracy)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'PLSDA{file_name_no_ext}_{y.name}_report_df.csv')
        print(report_df.to_string())

        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels, annot_kws={"size": 14})

        colorbar = heatmap.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=14)

        plt.title(f'{file_name_no_ext} - {y.name} - Accuracy={accuracy:.2f}', fontsize=16)
        plt.xlabel('Predicted', fontsize=15, labelpad=10)
        plt.ylabel('True', fontsize=15, labelpad=10)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f'PLSDA{file_name_no_ext}_{y.name}_Accuracy={accuracy:.2f}', format='png')
        plt.show()
        plt.close()

        # Calcula VIP scores para o modelo PLSDA por classe
        T = plsda_best.x_scores_
        W = plsda_best.x_weights_
        Q = plsda_best.y_loadings_

        # VIP para cada classe
        p, h = X_train.shape[1], best_n_components
        VIP_scores = np.zeros((p, len(sorted_labels)))  # Matriz de VIP para cada classe

        for class_idx in range(len(sorted_labels)):
            for i in range(p):
                s = np.sum([T[:, comp]**2 * W[i, comp]**2 * Q[class_idx, comp]**2 for comp in range(h)], axis=0)
                VIP_scores[i, class_idx] = np.sqrt(p * np.sum(s) / np.sum(T**2))

        # Define os ticks e os labels em escala 10^3 e ordem decrescente
        x_ticks = np.arange(0, len(label_espectro), x_scale)  # X_scale é o intervalo de X
        #x_ticklabels = [round((label_espectro[i]) / 1000, 1) for i in x_ticks]  # Divide cada valor por 1000 para a escala
        x_ticklabels=[label_espectro[i] for i in x_ticks] #caso seja um roto sem numero

        # Ordena os labels em ordem decrescente
        #x_ticks_sorted = sorted(x_ticks, reverse=True)
        #x_ticklabels_sorted = sorted(x_ticklabels, reverse=True)

        #print(f"x_ticklabels_sorted:{x_ticklabels_sorted}")
        #print(f"x_ticklabels:{x_ticklabels}")
        #print(f"x_ticks_sorted:{x_ticks_sorted}")
        #print(f"x_ticks:{x_ticks}")
        #print(label_espectro)



        # Gerar gráficos de VIP score para cada classe
        for class_idx, class_name in enumerate(sorted_labels):
            plt.figure(figsize=(12, 12))
            plt.bar(range(1, VIP_scores.shape[0] + 1), VIP_scores[:, class_idx], color='blue')
            plt.xlabel('Wavenumbers (cm-1)', fontsize=15)
            plt.ylabel('VIP Score', fontsize=20)
            plt.title(f'VIP Scores for {class_name}', fontsize=20)

            # Usa os labels em ordem decrescente e escala 10^3
            plt.xticks(x_ticks, x_ticklabels, rotation=45, fontsize=18)
            plt.yticks(fontsize=20)

            # Salva a figura
            plt.savefig(f'PLSDA_VIP_{file_name_no_ext}_{class_name}.png', format='png')
            plt.show()
            plt.close()

    return best_n_components, accuracy

#quando começa do menor para o maior
def get_plsda_performance_inverse(X, y, preprocess_name, test_size=0.30, max_components=20, n_splits=10, feature_names=None, plotar_plsda=False, x_scale=20):


    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        nan_indices_X = X.isnull().any(axis=1)  # Identifica linhas com qualquer NaN em X
    else:
        nan_indices_X = np.any(np.isnan(X), axis=1)

    # Considere '0', strings vazias e NaN como rótulos inválidos
    nan_indices_y = pd.isnull(y) | (y == 0) | (y == "")  # Exclui valores 0, strings vazias e NaNs

    # Identificar amostras que contêm NaN ou valores inválidos tanto em X quanto em y
    nan_indices = np.logical_or(nan_indices_X, nan_indices_y)

    # Remover amostras com NaN ou valores inválidos em X ou em y
    X_clean = X[~nan_indices]
    y_clean = y[~nan_indices]

    # Verificar se ainda existem dados após remover as amostras inválidas
    if len(X_clean) == 0 or len(y_clean) == 0:
        print("Nenhuma amostra válida após remover as linhas com NaN ou valores inválidos.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Verificação extra: garantir que y_clean só contenha valores válidos
    unique_labels_before = set(y_clean)
    #print(f"Rótulos únicos antes da codificação: {unique_labels_before}")

    # Garantir que não exista nenhum valor que não deveria estar no vetor de rótulos
    y_clean = [label for label in y_clean if pd.notnull(label) and label != 0 and label != ""]  # Remover qualquer valor indesejado

    # Certifique-se de que os valores de y_clean são strings ou números
    y_clean = [str(label) for label in y_clean]  # Converter todos os rótulos para strings

    # Verificar se após essa limpeza ainda existem problemas
    unique_labels_after = set(y_clean)
    #print(f"Rótulos únicos após a conversão para strings: {unique_labels_after}")

    # Encode labels using LabelEncoder
    le = LabelEncoder()
    y_clean = le.fit_transform(y_clean)  # Codificar os rótulos restantes

    # Verificar e classificar os rótulos originais
    try:
      sorted_labels = sorted(le.classes_, key=lambda x: float(x))
    except ValueError:
      sorted_labels = sorted(le.classes_)
    sorted_indices = np.argsort([le.transform([label])[0] for label in sorted_labels])


    #sorted_indices = np.argsort(le.transform(le.classes_))
    #print('le.classes_[sorted_indices]', le.classes_[sorted_indices])
    #sorted_labels = le.classes_[sorted_indices]  # Mantém os rótulos originais em ordem


    #print(f"Rótulos labels: {sorted_labels}")
    #print(f"Rótulos index: {sorted_indices}")

    # Divida os dados em conjuntos de treino e teste
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=test_size, random_state=42, stratify=y_clean)
    except ValueError as e:
        print(f"Erro durante a separação treino-teste: {e}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    ohe = OneHotEncoder(sparse_output=False)  # Corrigido aqui
    y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))

    accuracy = []
    rmsecv = []

    for n_components in range(1, max_components + 1):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_rmse = []

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train_ohe[train_index], y_train[val_index]

            plsda = PLSRegression(n_components=n_components)
            plsda.fit(X_train_fold, y_train_fold)
            y_pred = plsda.predict(X_val_fold)
            y_pred_class = np.argmax(y_pred, axis=1)

            fold_rmse.append(np.sqrt(mean_squared_error(y_val_fold, y_pred_class)))

        rmsecv.append(np.mean(fold_rmse))


    best_n_components = np.argmin(rmsecv) + 1

    plsda_best = PLSRegression(n_components=best_n_components, scale=False)
    plsda_best.fit(X_train, y_train_ohe)
    y_pred_test = plsda_best.predict(X_test)
    y_pred_test_class = np.argmax(y_pred_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred_test_class)

    # Extrai o nome do arquivo sem a extensão
    file_name_no_ext = os.path.splitext(file)[0]

    if plotar_plsda:
        # Gráfico de RMSECV por LV
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_components + 1), rmsecv, marker='o', linestyle='-', color='blue', label='RMSECV')
        plt.axvline(x=best_n_components, color='red', linestyle='--', label=f'Best LV: {best_n_components}')
        plt.xlabel('Number of Components', fontsize=15)
        plt.ylabel('RMSECV', fontsize=15)
        plt.title(f'{file_name_no_ext}-{y.name} - RMSECV by LV - {preprocess_name}', fontsize=16)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(os.path.join(PATH, f'PLSDA{file_name_no_ext}-{y.name} - RMSECV by LV - {preprocess_name}'), format='png')
        plt.show()
        plt.close()

        # Gráfico Confusion matriz
        cm = confusion_matrix(y_test, y_pred_test_class)
        report = classification_report(y_test, y_pred_test_class, target_names=sorted_labels, output_dict=True)

        print(accuracy)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(PATH, f'PLSDA{file_name_no_ext}_{y.name}_report_df.csv'))
        print(report_df.to_string())

        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels, annot_kws={"size": 14})

        colorbar = heatmap.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=14)

        plt.title(f'{file_name_no_ext} - {y.name} - Accuracy={accuracy:.2f}', fontsize=16)
        plt.xlabel('Predicted', fontsize=15, labelpad=10)
        plt.ylabel('True', fontsize=15, labelpad=10)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(os.path.join(PATH, f'PLSDA{file_name_no_ext}_{y.name}_Accuracy={accuracy:.2f}'), format='png')
        plt.show()
        plt.close()


        # Calcula os labels dos espectros
        label_espectro = data.columns.str.split('.').str[0]

        # Calcula VIP scores para o modelo PLSDA por classe
        T = plsda_best.x_scores_
        W = plsda_best.x_weights_
        Q = plsda_best.y_loadings_

        # VIP para cada classe
        p, h = X_train.shape[1], best_n_components
        VIP_scores = np.zeros((p, len(sorted_labels)))  # Matriz de VIP para cada classe

        for class_idx in range(len(sorted_labels)):
            for i in range(p):
                s = np.sum([T[:, comp]**2 * W[i, comp]**2 * Q[class_idx, comp]**2 for comp in range(h)], axis=0)
                VIP_scores[i, class_idx] = np.sqrt(p * np.sum(s) / np.sum(T**2))

        # Define os ticks e os labels em escala 10^3 e ordem decrescente
        x_ticks = np.arange(0, len(label_espectro), x_scale)  # X_scale é o intervalo de X
        x_ticklabels = [round(int(label_espectro[i]) / 100000, 1) for i in x_ticks]  # Divide cada valor por 1000 para a escala

        x_ticklabels_sorted = x_ticklabels[::-1]

        # Gerar gráficos de VIP score para cada classe
        for class_idx, class_name in enumerate(sorted_labels):
            total_variaveis = VIP_scores.shape[0]
            plt.figure(figsize=(8, 6))
            plt.bar(range(total_variaveis, 0, -1), VIP_scores[:, class_idx], color='blue')
            plt.xlabel('Wavenumbers (cm-1)', fontsize=15)
            plt.ylabel('VIP Score', fontsize=15)
            plt.title(f'VIP Scores for {class_name}', fontsize=16)

            # Usa os labels em ordem decrescente e escala 10^3
            plt.xticks(x_ticks, x_ticklabels_sorted, rotation=45, fontsize=14)
            plt.yticks(fontsize=14)

            # Salva a figura
            plt.savefig(os.path.join(PATH, f'PLSDA_VIP_{file_name_no_ext}_{class_name}.png'), format='png')
            plt.show()
            plt.close()

    return best_n_components, accuracy

#TODO: Importar OCPLS
#TODO: Importar DD-SIMCA