import auto_spectra_engine.experiment as ase

file = "/home/barbon/Python/AutoSpectraEngine/auto_spectra_engine/datasets/NIR1.csv"
best_experiment = ase.run_ga_experiments(file, modelo="PLSDA", coluna_predicao="Adulterant", budget=500, pipeline_family="NIR", use_parallelization=True)

#best_experiment = {'start_index': 26, 'end_index': 1084, 'contamination': 0.02, 'combinacao': 'mc + d1', 'performance': 0.96}
print(best_experiment)
ase.run_experiment(file, start_index=best_experiment["start_index"], end_index=best_experiment["end_index"], contamination=best_experiment["contamination"], combinacao=best_experiment["combinacao"], 
                   plot=True, modelo="PLSDA", verbose=True, coluna_predicao="Adulterant")