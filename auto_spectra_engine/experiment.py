import os
import pandas as pd

from util import load_data
from preprocessing import mean_centering, autoscaling, smoothing, first_derivative, second_derivative, msc, snv
from machinelearning import iso_forest_outlier_removal, get_RF_performance
from analysis import plot_PCA, get_plsda_performance, OneClassPLS

def run_experiment(file, start_index=0, end_index=None, contamination=0.0, combinacao="mc+smo+d1", cortar_extremidades=False, inicio_espectro_col=3, plotar=True, modelo="PLSDA", inlier_class="Pure", coluna_predicao=0):
    """
    Função para executar um experimento completo de pré-processamento, remoção de outliers e avaliação de modelos de regressão ou classificação.

    Parâmetros:
    file (str): Caminho do arquivo CSV contendo os dados espectrais.
    start_index (int): Índice da coluna onde inicia o espectro.
    end_index (int): Índice da coluna onde termina o espectro.
    contamination (float): Percentual de outliers na base de dados.
    combinacao (str): Combinação de pré-processamentos a serem aplicados.
    cortar_extremidades (bool): Indica se as extremidades do espectro devem ser
    """

    start_index = 394 # start_index inicio do espectro e end_index fim do espectro (valor numerico da coluna)
    end_index = 788
    contamination = 0.0 # Qual o percentual de outliers
    combinacao = "mc+smo+d1" # Qual o preprocessamento
    cortar_extremidades = False # Quer cortar a extremidade do espectro
    resultados = []
    file_name_no_ext = os.path.splitext(file)[0] # Obtendo o nome das colunas
    data = load_data(file) # Carregando a matriz de dados
    data = data.infer_objects()

    X = data.iloc[:, inicio_espectro_col:] # Filtrando o que é a matriz preditora e o que é resposta ou classe
    Ys = data.iloc[:, :inicio_espectro_col]

    data = X

    if cortar_extremidades: # Caso corte as extremidades
        perc_corte = int(data.shape[1] * 0.02)
        data = data.iloc[:, perc_corte:]
        data = data.iloc[:, :-perc_corte]

    data = data.iloc[:, start_index:end_index]  # Armazenando start_index e end_index
    label_espectro = data.columns.str.split('.').str[0]


    # Pré-processamento
    df_pp = data
    wavelength_columns = df_pp.columns
    #print(df_pp.shape)
    #print(df_pp)
    if "mc" in combinacao:
        df_pp = pd.DataFrame(mean_centering(df_pp.values), columns=wavelength_columns)

    if "scal"in combinacao:
        df_pp = pd.DataFrame(autoscaling(df_pp.values), columns=df_pp.columns)

    if "smo" in combinacao:
        df_pp = pd.DataFrame(smoothing(df_pp.values), columns=wavelength_columns)

    if "d2" in combinacao:
        df_pp = pd.DataFrame(second_derivative(df_pp.values), columns=wavelength_columns)

    if "d1" in combinacao:
        df_pp = pd.DataFrame(first_derivative(df_pp.values), columns=wavelength_columns)

    if "msc" in combinacao:
        df_pp = pd.DataFrame(msc(df_pp.values), columns=wavelength_columns)

    if "snv" in combinacao:
        df_pp = pd.DataFrame(snv(df_pp.values), columns=wavelength_columns)

    sub_data, sub_Ys = iso_forest_outlier_removal(df_pp.values, Ys, contamination) # Aplicação do Isolation Forest
    
    if plotar:
        plot_PCA(df_pp.values, sub_Ys.loc[:, coluna_predicao], combinacao, file_name_no_ext)

    if "PLSR" in modelo:
        # Chamando a função
        best_n_components, rmse_cal, rmsecv, rmse_test, rpd, rer, r2_cal, r2_cv, r2_test = get_plsr_performance(sub_data.values,sub_Ys.iloc[:,coluna_predicao],combinacao,test_size=0.33,max_components=10, n_splits=5, plotar_plsr = plotar_plsr)
        # Armazenamento dos resultados
        resultados.append([file, cortar_extremidades, combinacao, round(contamination, 2), round(best_n_components, 2), round(rmse_cal, 2), round(rmsecv, 2), round(rmse_test, 2), round(rpd, 2), round(rer, 2), round(r2_cal, 2), round(r2_cv, 2), round(r2_test, 2), start_index, end_index])
        resultados_df = pd.DataFrame(resultados, columns=[ "file", "cortar_extremidades", "pre_processamento", "contaminacao", "best_n_components", "rmse_cal", "rmsecv", "rmse_test", "RPD", "RER", "r2_cal", "r2_cv", "r2_test", "start_index", "end_index"])
        resultados_df.sort_values(['rmse_test']).to_csv(os.path.join(PATH, f"PLSR_Best_{file_name_no_ext}_{coluna_predicao}.csv"), index=False)  # Save to CSV without index

    if "PLSDA" in modelo:
        # Chamando a função
        best_n_components, accuracy = get_plsda_performance(sub_data.values, sub_Ys.iloc[:, coluna_predicao], combinacao, test_size=0.33, max_components=20, n_splits=10, plotar_plsda=plotar_plsda, label_espectro=label_espectro)
        # Armazenamento dos resultados
        resultados.append([file, cortar_extremidades, combinacao, round(contamination, 2), round(best_n_components, 2), round(accuracy, 2), start_index, end_index])
        resultados_df = pd.DataFrame(resultados, columns=[ "file", "cortar_extremidades", "pre_processamento", "contaminacao", "best_n_components", "accuracy", "start_index", "end_index"])
        resultados_df.sort_values(['accuracy'], ascending=False).to_csv(f"{file_name_no_ext}_{coluna_predicao}_PLSDA_Best_.csv", index=False)  # Save to CSV without index

    if "RF" in modelo:
        # Chamando a função
        accuracy_mean, accuracy_std, accuracy_min, accuracy_max, seed_accuracy_max, sensibility_mean, sensibility_std, sensibility_min, sensibility_max, specificity_mean, specificity_std, specificity_min, specificity_max = get_RF_performance(sub_data.values, sub_Ys.iloc[:, coluna_predicao], test_size=0.33, n_splits=5, plotar_RF=plotar, feature_names = label_espectro, file_name_no_ext=file_name_no_ext)
        resultados.append([ file, cortar_extremidades, combinacao, round(contamination, 2), round(accuracy_mean, 2), round(accuracy_std, 2), round(accuracy_min, 2), round(accuracy_max, 2), round(seed_accuracy_max, 2), round(sensibility_mean, 2), round(sensibility_std, 2), round(sensibility_min, 2), round(sensibility_max, 2), round(specificity_mean, 2), round(specificity_std, 2), round(specificity_min, 2), round(specificity_max, 2),
        start_index, end_index])
        # Armazenamento dos resultados
        resultados_df = pd.DataFrame(resultados, columns=["file", "cortar_extremidades", "pre_processamento", "contaminacao", "accuracy_mean", "accuracy_std", "accuracy_min", "accuracy_max", "seed_accuracy_max", "sensibility_mean", "sensibility_std", "sensibility_min", "sensibility_max",
        "specificity_mean", "specificity_std", "specificity_min", "specificity_max", "start_index", "end_index"])
        resultados_df.sort_values(['accuracy_mean'], ascending=False).to_csv(f"{file_name_no_ext}_{coluna_predicao}_RF_Best_.csv", index=False)  # Save to CSV without index

    if "OneClassPLS" in modelo:
        # Chamando a classe e função
        ocpls_modelo = OneClassPLS(n_components=2, inlier_class=inlier_class, n_splits=3, plotar=plotar, file_name_no_ext=file_name_no_ext, coluna_y_nome=coluna_predicao)
        best_accuracy, best_n_components, best_sensitivity, best_specificity = ocpls_modelo.fit_and_evaluate_full_pipeline(sub_data, sub_Ys, coluna_predicao, plotar)
        # Armazenamento dos resultados
        resultados.append([ file, cortar_extremidades, combinacao, round(contamination, 2), round(best_accuracy, 2), round(best_sensitivity, 2), round(best_specificity, 2), start_index, end_index, round (best_n_components)])
        resultados_df = pd.DataFrame(resultados, columns=[ "file", "cortar_extremidades", "pre_processamento", "contaminacao", "accuracy", "sensitivity", "specificity", "start_index", "end_index", "best_n_components"                          ])
        resultados_df.sort_values(['accuracy'], ascending=False).to_csv(f"{file_name_no_ext}_{coluna_predicao}_best_OneClassPLS_.csv", index=False) # Save to CSV without index

    if "DDSIMCA" in modelo:
        # Chamando a função
        ddsimca_modelo = DDSIMCA(n_components=2, inlier_class=inlier_class,plotar_DDSIMCA=plotar)
        best_accuracy, best_n_components, best_sensitivity, best_specificity = ddsimca_modelo.fit_and_evaluate_full_pipeline(sub_data, sub_Ys, coluna_predicao)
        #print(best_accuracy)
            # Armazenamento dos resultados
        resultados.append([ file, cortar_extremidades, combinacao, round(contamination, 2), round(best_accuracy, 2), round(best_sensitivity, 2), round(best_specificity, 2), start_index, end_index, round (best_n_components)])
        resultados_df = pd.DataFrame(resultados, columns=[ "file", "cortar_extremidades", "pre_processamento", "contaminacao", "accuracy", "sensitivity", "specificity", "start_index", "end_index", "best_n_components"                          ])
        resultados_df.sort_values(['accuracy'], ascending=False).to_csv(f"{file_name_no_ext}_{coluna_predicao}_best_DDSIMCA_.csv", index=False)


#TODO: Importar BatchExperiment



#file = "/home/barbon/PycharmProjects/AutoSpectraEngine/auto_spectra_engine/datasets/raman.csv"
file = "/home/barbon/Python/AutoSpectraEngine/auto_spectra_engine/datasets/raman.csv"
#run_experiment(file, modelo="OneClassPLS", coluna_predicao="Class", plotar=True) # Executar 
run_experiment(file, modelo="OneClassPLS", coluna_predicao="Class", plotar=True) # Executar 
