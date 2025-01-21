import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, confusion_matrix
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

import seaborn as sns

def plot_PCA(X, Y, combinacao, file_name_no_ext):
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
        categories = Y.unique()
        colors = plt.cm.get_cmap('viridis', len(categories))
        for i, category in enumerate(categories):
            plt.scatter(components[Y == category, 0], components[Y == category, 1], color=colors(i), label=str(category))
        plt.legend(title='Target Variable (Categorical)')

    # Set labels and title with variance explained
    plt.xlabel(f'PC1 Variance Explained: {explained_variance[0]*100:.2f}%')
    plt.ylabel(f'PC2 Variance Explained: {explained_variance[1]*100:.2f}%')
    plt.title('PCA Plot:'+combinacao)
    plt.savefig(f'{file_name_no_ext}_{combinacao}_PCA.png', format='png')
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
        plt.savefig(f'PLSDA{file_name_no_ext}-{y.name} - RMSECV by LV - {preprocess_name}.png', format='png')
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
        plt.savefig(f'PLSDA{file_name_no_ext}_{y.name}_Accuracy={accuracy:.2f}.png', format='png')
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
        plt.savefig(f'{file_name_no_ext}-{y.name} - RMSECV by LV - {preprocess_name}_PLSDA.png', format='png')
        plt.show()
        plt.close()

        # Gráfico Confusion matriz
        cm = confusion_matrix(y_test, y_pred_test_class)
        report = classification_report(y_test, y_pred_test_class, target_names=sorted_labels, output_dict=True)

        print(accuracy)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'{file_name_no_ext}_{y.name}_report_df_PLSDA.csv')
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


class OneClassPLS:
    def __init__(self, n_components, inlier_class, n_splits=20, plotar=False, file_name_no_ext="FilePLS", coluna_y_nome="Target"):
        self.n_components = n_components
        self.inlier_class = inlier_class
        self.pls = PLSRegression(n_components=n_components)
        self.scaler = StandardScaler()
        self.threshold = None
        self.n_splits = n_splits
        self.plotar = plotar
        self.file_name_no_ext = file_name_no_ext
        self.coluna_y_nome = coluna_y_nome

    def fit(self, X, y):
        y = np.array([item.strip() for item in y.tolist()])
        X_inliers = X[y == self.inlier_class] # Seleciona apenas os dados que pertencem à classe "inlier"
        X_scaled = self.scaler.fit_transform(X_inliers) # Standardiza os dados
        self.pls.fit(X_scaled, X_scaled) # Ajusta o modelo final usando todos os dados inliers
        X_scores = self.pls.transform(X_scaled) # Projeta os dados no espaço latente
        distances = np.linalg.norm(X_scores, axis=1) # Calcula a distância para o centro no espaço latente
        self.threshold = np.mean(distances) + 2 * np.std(distances) # Calcula o threshold (média + 2 desvios padrão)

    def predict(self, X):
        X_scaled = self.scaler.transform(X) # Padronizando a matriz
        X_scores = self.pls.transform(X_scaled) # Projetando a data no PLS latent space
        distances = np.linalg.norm(X_scores, axis=1) # Computando a distanca da media no latent space
        return np.where(distances <= self.threshold, 1, -1) # Classificando como inlier (1) se T² e Q aestao abaixo do thresholds, ou denomina outlier (-1)

    def fit_and_evaluate_full_pipeline(self, df_pp, sub_Ys, coluna_predicao, plotar):
        # Preparação dos dados
        sub_Ys = sub_Ys.reset_index(drop=True)
        df_pp = df_pp.reset_index(drop=True)

        pure_mask = sub_Ys.loc[:, coluna_predicao].str.replace(" ", "") == self.inlier_class.replace(" ", "")
        non_pure_mask = ~pure_mask

        # Dados inliers
        X_inliers = df_pp[pure_mask].values
        y_inliers = sub_Ys.loc[pure_mask, coluna_predicao].values

        # Dados outliers
        X_outliers = df_pp[non_pure_mask].values
        y_outliers = sub_Ys.loc[non_pure_mask, coluna_predicao].values

        # Dividir inliers em 80% para treino e 20% para teste
        X_train_pure, X_test_pure, y_train_pure, y_test_pure = train_test_split(
            X_inliers, y_inliers, test_size=0.2, random_state=42, shuffle=True
        )

        # Conjunto de teste: 20% dos inliers + todos os outliers
        X_test = np.vstack([X_test_pure, X_outliers])
        y_test = np.hstack([y_test_pure, y_outliers])

        best_sensitivity = 0
        best_n_components = 2
        best_accuracy = 0
        best_specificity = 0

        # Testa diferentes números de componentes e seleciona com base na melhor sensibilidade
        for n_components in range(2, 11):
            self.n_components = n_components
            self.pls = PLSRegression(n_components=n_components)

            # Ajusta o modelo nos dados de treino (inliers) e avalia no conjunto de teste (inliers e outliers)
            self.fit(X_train_pure, y_train_pure)
            predictions = self.predict(X_test)

            # Verificação de métricas
            true_labels = np.where(y_test == self.inlier_class, 1, -1)
            current_accuracy = np.mean(predictions == true_labels)

            # Calculando Sensibilidade e Especificidade
            cm = confusion_matrix(true_labels, predictions, labels=[1, -1])
            sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] > 0 else 0
            specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if cm[1, 0] + cm[1, 1] > 0 else 0

            # Guarda o número de componentes que resulta na melhor acuracia
            if current_accuracy > best_accuracy:
                best_sensitivity = sensitivity
                best_n_components = n_components
                best_accuracy = current_accuracy
                best_specificity = specificity

        # Usa o melhor número de componentes encontrado
        self.n_components = best_n_components
        self.pls = PLSRegression(n_components=best_n_components)
        self.fit(X_train_pure, y_train_pure)

        if plotar:
          # Criar matriz de confusão
          true_labels = np.where(y_test == self.inlier_class, 1, -1)
          predictions = self.predict(X_test)
          cm = confusion_matrix(true_labels, predictions)


          # Verifique os tamanhos dos arrays
          print(f"True labels shape: {true_labels.shape}, Predictions shape: {predictions.shape}")

          original_labels = ["Non-Pure", "Pure"]  # Ajustar de acordo com a sua classificação

          # Plotar a matriz de confusão
          plt.figure(figsize=(8, 6))
          heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=original_labels, yticklabels=original_labels,
                                annot_kws={"size": 14})

          # Ajustar os parâmetros da barra de cores
          colorbar = heatmap.collections[0].colorbar
          colorbar.ax.tick_params(labelsize=14)

          # Título e rótulos
          plt.title(f'Best Accuracy={best_accuracy:.2f}', fontsize=16)  # Corrigido para título
          plt.xlabel('Predicted', fontsize=15)
          plt.ylabel('True', fontsize=15)
          plt.xticks(fontsize=14)
          plt.yticks(fontsize=14)

          # Salvar o gráfico da matriz de confusão
          plt.savefig(f'{self.file_name_no_ext}_{self.coluna_y_nome}_Best_Accuracy={best_accuracy:.2f}_OneClassPLS.png', format='png')
          plt.show()
          plt.close()

          # Exibir amostras incorretamente classificadas
          incorrect_samples = []
          for idx, (true_label, pred_label) in enumerate(zip(true_labels, predictions)):
            if true_label != pred_label:
                incorrect_sample_info = {
                    'Index': idx,  # Índice da amostra no conjunto original de dados
                    'True Label': 'Pure' if true_label == 1 else 'Non-Pure',
                    'Predicted Label': 'Pure' if pred_label == 1 else 'Non-Pure',
                    'Column 2 Value': y_test[idx]  # Valor da coluna 2
                }
                incorrect_samples.append(incorrect_sample_info)

          if incorrect_samples:
              print("Amostras classificadas incorretamente:")
              for sample in incorrect_samples:
                  print(f"Índice: {sample['Index']}, Rótulo Verdadeiro: {sample['True Label']}, Rótulo Previsto: {sample['Predicted Label']}, Valor da Coluna 2: {sample['Column 2 Value']}")

              column_2_values = [sample['Column 2 Value'] for sample in incorrect_samples]
              plt.figure(figsize=(10, 6))
              plt.hist(column_2_values, bins=10, color='skyblue', edgecolor='black')
              plt.title('Histograma dos Valores da Coluna 2 (Amostras Classificadas Incorretamente)', fontsize=16)
              plt.xlabel('Valor da Coluna 2', fontsize=14)
              plt.ylabel('Frequência', fontsize=14)
              plt.xticks(fontsize=12)
              plt.yticks(fontsize=12)
              plt.show()
              plt.close()
          else:
              print("Nenhuma amostra foi classificada incorretamente.")
        return best_accuracy, best_n_components, best_sensitivity, best_specificity

#TODO: Importar DD-SIMCA