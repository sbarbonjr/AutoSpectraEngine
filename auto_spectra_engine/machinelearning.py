from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import ossaudiodev

def get_RF_performance(X, y, test_size=0.33, n_splits=10, n_runs=10, plotar_RF=False, feature_names=None, x_scale=5, file_name_no_ext=None):
    # Verificação de dados ausentes em X
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

    accuracies = []
    sensitivities = []
    specificities = []
    confusion_matrices = []
    classification_reports = []
    seeds = list(range(n_runs))
    rf_models = []

    # Loop para realizar múltiplas execuções com diferentes seeds
    for seed in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=test_size, random_state=seed)
        rf = RandomForestClassifier(random_state=seed)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        # Avaliação de performance
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred, average='macro')
        specificity = recall_score(y_test, y_pred, average='macro', labels=np.unique(y_test))

        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        confusion_matrices.append(confusion_matrix(y_test, y_pred))
        classification_reports.append(classification_report(y_test, y_pred, output_dict=True))
        rf_models.append(rf)

    # Cálculo de estatísticas de performance
    accuracy_mean = np.mean(accuracies)
    sensitivity_mean = np.mean(sensitivities)
    specificity_mean = np.mean(specificities)

    accuracy_std = np.std(accuracies)
    sensitivity_std = np.std(sensitivities)
    specificity_std = np.std(specificities)

    accuracy_min = np.min(accuracies)
    sensitivity_min = np.min(sensitivities)
    specificity_min = np.min(specificities)

    accuracy_max = np.max(accuracies)
    sensitivity_max = np.max(sensitivities)
    specificity_max = np.max(specificities)

    seed_accuracy_max = seeds[accuracies.index(accuracy_max)]
    confusion_matrix_max = confusion_matrices[accuracies.index(accuracy_max)]
    classification_report_max = classification_reports[accuracies.index(accuracy_max)]
    rf_model_max = rf_models[accuracies.index(accuracy_max)]

    if plotar_RF:
      print(accuracy_max)
      report_df = pd.DataFrame(classification_report_max).transpose()
      report_df.to_csv(f'{file_name_no_ext}_{y.name}_RF_report_df.csv')
      print(report_df.to_string())

      plt.figure(figsize=(8, 6))
      sns.heatmap(confusion_matrix_max, annot=True, fmt='d', cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels)
      plt.title(f'RF {y.name} Accuracy={accuracy_max:.2f}', fontsize=24)
      plt.xlabel('Predicted', fontsize=14)
      plt.ylabel('True', fontsize=14)
      plt.xticks(fontsize=12)
      plt.yticks(fontsize=12)
      plt.savefig(f'{file_name_no_ext}_{y.name}_Accuracy={accuracy_max:.2f}_RF_', format='png')
      plt.show()
      plt.close()

      if feature_names is None:
          feature_names = X_clean.columns if isinstance(X_clean, pd.DataFrame) else range(X_clean.shape[1])

      importances = rf_model_max.feature_importances_

      # Defina os tamanhos de fonte como variáveis
      font_size_labels = 14
      font_size_ticks = 14
      font_size_y_values = 14  # Tamanho da fonte para os valores do eixo y

      fig, ax1 = plt.subplots(figsize=(10, 6))
      x_frame = pd.DataFrame(X_clean).melt()
      sns.lineplot(x='variable', y='value', data=x_frame, ax=ax1, label='Mean spectrum')

      ax2 = ax1.twinx()
      colors = plt.cm.viridis(importances / max(importances))
      x_ticks = np.arange(0, len(feature_names), x_scale)#X_acale é o intervalo de X
      x_ticklabels = [feature_names[i] for i in x_ticks]

      sns.barplot(x=feature_names, y=importances, alpha=0.7, ax=ax2, palette=colors.tolist(), hue=feature_names)

      ax1.set_xlabel('Wavenumbers (cm-1)', fontsize=font_size_labels) # se for cm-1
      #ax1.set_xlabel('Wavelengths (nm)', fontsize=font_size_labels) # se for nm
      ax1.set_ylabel('Absorbance', color='k', fontsize=font_size_labels)
      ax2.set_ylabel('Feature Importance', color='k', fontsize=font_size_labels)
      ax1.set_xticks(x_ticks)
      ax1.set_xticklabels(x_ticklabels, rotation=45, fontsize=font_size_ticks)

      # Ajustar o tamanho da fonte dos valores no eixo y do ax2
      ax1.tick_params(axis='y', labelsize=font_size_y_values)  # Ajustar para Absorbance
      ax2.tick_params(axis='y', labelsize=font_size_y_values)  # Ajustar para Feature Importance


      fig.tight_layout()
      plt.savefig(f'{file_name_no_ext}_{y.name}_RF_feature_importances.png', format='png')
      plt.show()

    return accuracy_mean, accuracy_std, accuracy_min, accuracy_max, seed_accuracy_max, sensitivity_mean, sensitivity_std, sensitivity_min, sensitivity_max, specificity_mean, specificity_std, specificity_min, specificity_max


def iso_forest_outlier_removal(X, Ys, contamination):
     # Convert data to DataFrame if it is not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if contamination<=0:
      return X, Ys

    # Select only numerical columns
    numerical_columns = X.select_dtypes(include='number').columns
    numerical_data = X[numerical_columns]

    # Check if there are numerical columns
    if numerical_data.empty:
        raise ValueError("No numerical columns found in the input data.")

    # Initialize the Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=42)

    # Fit the model and predict outliers
    outliers = iso_forest.fit_predict(numerical_data)

    # Filter out the outliers
    clean_data = X[outliers != -1]
    clean_targets = Ys[outliers != -1]

    #print('cont x',clean_data.shape) #apenas para checar
    #print('cont y', clean_targets.shape) #apenas para checar

    return clean_data, clean_targets