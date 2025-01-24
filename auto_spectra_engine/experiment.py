import os
import pandas as pd

from auto_spectra_engine.util import load_data, insert_results_subpath, plot_performance_comparison
from auto_spectra_engine.preprocessing import mean_centering, autoscaling, smoothing, first_derivative, second_derivative, msc, snv, iso_forest_outlier_removal
from auto_spectra_engine.analysis_regression import plot_PCA, get_plsr_performance 
from auto_spectra_engine.analysis_oneclass import OneClassPLS, DDSIMCA
from auto_spectra_engine.analysis_classification import get_RF_performance, get_plsda_performance

try:
    # Check if running in a Jupyter Notebook or Colab
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        from tqdm.notebook import tqdm  # Use notebook-friendly tqdm
    else:
        from tqdm import tqdm  # Use standard tqdm for console
except Exception:
    from tqdm import tqdm  # Default to console-friendly tqdm

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def run_all_experiments(file, modelo="PLSDA", coluna_predicao="Class", test_contamination=False):
    pipelines = ['mc', 'scal', 'smo', 'd2', 'd1', 'msc', 'snv',
                #'mc + scal', 'mc + smo', 'mc + d2', 'mc + d1', 'mc + msc', 'mc + snv',
                #'scal + smo', 'scal + d2', 'scal + d1', 'scal + msc', 'scal + snv',
                #'smo + d2', 'smo + d1', 'smo + msc', 'smo + snv',
                #'d2 + msc', 'd2 + snv', 'd1 + msc', 'd1 + snv',
                #'mc + scal + smo', 'mc + scal + d2', 'mc + scal + d1', 'mc + scal + msc', 'mc + scal + snv',
                #'mc + smo + d2', 'mc + smo + d1', 'mc + smo + msc', 'mc + smo + snv',
                #'mc + d2 + msc', 'mc + d2 + snv', 'mc + d1 + msc', 'mc + d1 + snv',
                #'scal + smo + d2', 'scal + smo + d1', 'scal + smo + msc', 'scal + smo + snv',
                #'scal + d2 + msc', 'scal + d2 + snv', 'scal + d1 + msc', 'scal + d1 + snv',
                #'smo + d2 + msc', 'smo + d2 + snv', 'smo + d1 + msc', 'smo + d1 + snv',
                #'mc + scal + smo + d2', 'mc + scal + smo + d1', 'mc + scal + smo + msc', 'mc + scal + smo + snv',
                #'mc + scal + d2 + msc', 'mc + scal + d2 + snv', 'mc + scal + d1 + msc', 'mc + scal + d1 + snv',
                #'mc + smo + d2 + msc', 'mc + smo + d2 + snv', 'mc + smo + d1 + msc', 'mc + smo + d1 + snv',
                #'scal + smo + d2 + msc', 'scal + smo + d2 + snv', 'scal + smo + d1 + msc', 'scal + smo + d1 + snv',
                #'mc + scal + smo + d2 + msc', 'mc + scal + smo + d2 + snv', 'mc + scal + smo + d1 + msc', 'mc + scal + smo + d1 + snv'
                ]
    
    c_contamination=[0]
    if test_contamination:
        c_contamination = [0,0.01,0.025,0.05,0.1]
    
    result = []
    for pipeline in tqdm(pipelines, desc='Processing pipelines', leave=False):
        # Exclude combinations with "d2 + d1" and "msc + snv"
        for c in tqdm(c_contamination, desc='Processing contamination', leave=False):
            result.append(run_experiment(file, modelo=modelo, coluna_predicao=coluna_predicao, contamination=c, combinacao=pipeline, plot=False, verbose=False))
    result_df = pd.DataFrame(result, columns=["LV", "Pipeline", "Outliers (c)", "Performance"])
    plot_performance_comparison(result_df)


def run_experiment(file, start_index=0, end_index=None, contamination=0.0, combinacao=None, cortar_extremidades=False, inicio_espectro_col=3, plot=True, modelo="PLSDA", inlier_class="Pure", coluna_predicao=0, verbose=True):
    """
    Função para executar um experimento completo de pré-processamento, remoção de outliers e avaliação de modelos de regressão ou classificação.

    Parâmetros:
    file (str): Caminho do arquivo CSV contendo os dados espectrais.
    start_index (int): Índice da coluna onde inicia o espectro.
    end_index (int): Índice da coluna onde termina o espectro.
    contamination (float): Percentual de outliers na base de dados para filtrar com IsolationForest
    combinacao (str): Combinação de pré-processamentos a serem aplicados.
    cortar_extremidades (bool): Indica se as extremidades do espectro devem ser
    """

    if verbose:
        print("** AutoSpectraEngine (v0.1) ::: Run Experiment **")
        print(f"Processing file:\"{os.path.basename(file)}\"")
        if combinacao is None:
            combinacao = "mc+smo+d1"
            print("Hyperparameter \"combination\" was not used, applying standard value \"mc+smo+d1\"")


    resultados = []
    file_name_no_ext = os.path.splitext(file)[0] # Obtendo o nome das colunas
    data = load_data(file) # Carregando a matriz de dados
    data = data.infer_objects()

    X = data.iloc[:, inicio_espectro_col:] # Filtrando o que é a matriz preditora e o que é resposta ou classe
    Ys = data.iloc[:, :inicio_espectro_col]

    data = X
    if cortar_extremidades: # Caso corte as extremidades
        if verbose:
            print("Removing extremities...")
        perc_corte = int(data.shape[1] * 0.02)
        data = data.iloc[:, perc_corte:]
        data = data.iloc[:, :-perc_corte]
        start_index = perc_corte

    if end_index is None:
        data = data.iloc[:, start_index:]  # Armazenando start_index e end_index
    else:
        data = data.iloc[:, start_index:end_index]  # Armazenando start_index e end_index        
    label_espectro = data.columns.str.split('.').str[0]

    # Pré-processamento
    df_pp = data
    wavelength_columns = df_pp.columns
    if "mc" in combinacao:
        if verbose:
            print("Applying MC...")
        df_pp = pd.DataFrame(mean_centering(df_pp.values), columns=wavelength_columns)

    if "scal"in combinacao:
        if verbose:
            print("Applying AutoScaling...")
        df_pp = pd.DataFrame(autoscaling(df_pp.values), columns=df_pp.columns)

    if "smo" in combinacao:
        if verbose:
            print("Applying SMO...")        
        df_pp = pd.DataFrame(smoothing(df_pp.values), columns=wavelength_columns)

    if "d2" in combinacao:
        if verbose:
            print("Applying Second Derivative...")                
        df_pp = pd.DataFrame(second_derivative(df_pp.values), columns=wavelength_columns)

    if "d1" in combinacao:
        if verbose:
            print("Applying First Derivative...")           
        df_pp = pd.DataFrame(first_derivative(df_pp.values), columns=wavelength_columns)

    if "msc" in combinacao:
        if verbose:
            print("Applying MSC...")                   
        df_pp = pd.DataFrame(msc(df_pp.values), columns=wavelength_columns)

    if "snv" in combinacao:
        if verbose:
            print("Applying SNV...")                   
        df_pp = pd.DataFrame(snv(df_pp.values), columns=wavelength_columns)

    if contamination > 0 :
        if verbose:
            print(f"Outliers removal (Isolation Forest, c ={contamination})...")   
    sub_data, sub_Ys = iso_forest_outlier_removal(df_pp.values, Ys, contamination) # Aplicação do Isolation Forest
           
    if plot:
        if verbose:
            print(f"Plotting PCA...")          
        plot_PCA(df_pp.values, sub_Ys.loc[:, coluna_predicao], combinacao, file_name_no_ext)

    perf_return = []
    if "PLSR" in modelo:
        if verbose:
            print(f"Executing PLSR...")           
        best_n_components, rmse_cal, rmsecv, rmse_test, rpd, rer, r2_cal, r2_cv, r2_test = get_plsr_performance(sub_data.values,sub_Ys.loc[:,coluna_predicao],combinacao,test_size=0.33,max_components=10, n_splits=5, plotar_plsr = plot, file_name_no_ext=file_name_no_ext)
        resultados.append([file, cortar_extremidades, combinacao, round(contamination, 2), round(best_n_components, 2), round(rmse_cal, 2), round(rmsecv, 2), round(rmse_test, 2), round(rpd, 2), round(rer, 2), round(r2_cal, 2), round(r2_cv, 2), round(r2_test, 2), start_index, end_index])
        resultados_df = pd.DataFrame(resultados, columns=[ "file", "cortar_extremidades", "pre_processamento", "contaminacao", "best_n_components", "rmse_cal", "rmsecv", "rmse_test", "RPD", "RER", "r2_cal", "r2_cv", "r2_test", "start_index", "end_index"])
        resultados_df.sort_values(['rmse_test']).to_csv(f"{insert_results_subpath(file_name_no_ext)}_{coluna_predicao}_PLSR_Best.csv", index=False)  # Save to CSV without index
        perf_return = [best_n_components, combinacao, round(contamination, 2), round(rmse_test, 2)]

    if "PLSDA" in modelo:
        if verbose:
            print(f"Executing PLSDA...")           
        best_n_components, accuracy = get_plsda_performance(sub_data.values, sub_Ys.loc[:, coluna_predicao], combinacao, test_size=0.33, max_components=20, n_splits=10, plotar_plsda=plot, label_espectro=label_espectro)
        resultados.append([file, cortar_extremidades, combinacao, round(contamination, 2), round(best_n_components, 2), round(accuracy, 2), start_index, end_index])
        resultados_df = pd.DataFrame(resultados, columns=[ "file", "cortar_extremidades", "pre_processamento", "contaminacao", "best_n_components", "accuracy", "start_index", "end_index"])
        resultados_df.sort_values(['accuracy'], ascending=False).to_csv(f"{insert_results_subpath(file_name_no_ext)}_{coluna_predicao}_PLSDA_Best_.csv", index=False)  # Save to CSV without index
        perf_return = [best_n_components, combinacao, round(contamination, 2), round(accuracy, 2)]

    if "RF" in modelo:
        if verbose:
            print(f"Executing RF Classifier...")          
        accuracy_mean, accuracy_std, accuracy_min, accuracy_max, seed_accuracy_max, sensibility_mean, sensibility_std, sensibility_min, sensibility_max, specificity_mean, specificity_std, specificity_min, specificity_max = get_RF_performance(sub_data.values, sub_Ys.iloc[:, coluna_predicao], test_size=0.33, n_splits=5, plotar_RF=plot, feature_names = label_espectro, file_name_no_ext=file_name_no_ext)
        resultados.append([ file, cortar_extremidades, combinacao, round(contamination, 2), round(accuracy_mean, 2), round(accuracy_std, 2), round(accuracy_min, 2), round(accuracy_max, 2), round(seed_accuracy_max, 2), round(sensibility_mean, 2), round(sensibility_std, 2), round(sensibility_min, 2), round(sensibility_max, 2), round(specificity_mean, 2), round(specificity_std, 2), round(specificity_min, 2), round(specificity_max, 2),
        start_index, end_index])
        resultados_df = pd.DataFrame(resultados, columns=["file", "cortar_extremidades", "pre_processamento", "contaminacao", "accuracy_mean", "accuracy_std", "accuracy_min", "accuracy_max", "seed_accuracy_max", "sensibility_mean", "sensibility_std", "sensibility_min", "sensibility_max",
        "specificity_mean", "specificity_std", "specificity_min", "specificity_max", "start_index", "end_index"])
        resultados_df.sort_values(['accuracy_mean'], ascending=False).to_csv(f"{insert_results_subpath(file_name_no_ext)}_{coluna_predicao}_RF_Best_.csv", index=False)  # Save to CSV without index
        perf_return = [best_n_components, combinacao, round(contamination, 2), round(accuracy, 2)]

    if "OneClassPLS" in modelo:
        if verbose:
            print(f"Executing OneClassPLS Classifier...")          
        ocpls_modelo = OneClassPLS(n_components=2, inlier_class=inlier_class, n_splits=3, plotar=plot, file_name_no_ext=file_name_no_ext, coluna_y_nome=coluna_predicao)
        best_accuracy, best_n_components, best_sensitivity, best_specificity = ocpls_modelo.fit_and_evaluate_full_pipeline(sub_data, sub_Ys, coluna_predicao, plot)
        resultados.append([ file, cortar_extremidades, combinacao, round(contamination, 2), round(best_accuracy, 2), round(best_sensitivity, 2), round(best_specificity, 2), start_index, end_index, round (best_n_components)])
        resultados_df = pd.DataFrame(resultados, columns=[ "file", "cortar_extremidades", "pre_processamento", "contaminacao", "accuracy", "sensitivity", "specificity", "start_index", "end_index", "best_n_components"                          ])
        resultados_df.sort_values(['accuracy'], ascending=False).to_csv(f"{insert_results_subpath(file_name_no_ext)}_{coluna_predicao}_best_OneClassPLS_.csv", index=False) # Save to CSV without index
        perf_return = [best_n_components, combinacao, round(contamination, 2), round(accuracy, 2)]

    if "DDSIMCA" in modelo:
        if verbose:
            print(f"Executing DDSIMCA Classifier...")          
        ddsimca_modelo = DDSIMCA(n_components=2, inlier_class=inlier_class,plotar_DDSIMCA=plot)
        best_accuracy, best_n_components, best_sensitivity, best_specificity = ddsimca_modelo.fit_and_evaluate_full_pipeline(sub_data, sub_Ys, coluna_predicao)
        resultados.append([ file, cortar_extremidades, combinacao, round(contamination, 2), round(best_accuracy, 2), round(best_sensitivity, 2), round(best_specificity, 2), start_index, end_index, round (best_n_components)])
        resultados_df = pd.DataFrame(resultados, columns=[ "file", "cortar_extremidades", "pre_processamento", "contaminacao", "accuracy", "sensitivity", "specificity", "start_index", "end_index", "best_n_components"                          ])
        resultados_df.sort_values(['accuracy'], ascending=False).to_csv(f"{insert_results_subpath(file_name_no_ext)}_{coluna_predicao}_best_DDSIMCA_.csv", index=False)
        perf_return = [best_n_components, combinacao, round(contamination, 2), round(accuracy, 2)]
    
    return perf_return
