import auto_spectra_engine.experiment as ase

file = "/home/barbon/Python/AutoSpectraEngine/auto_spectra_engine/datasets/raman.csv"
best_experiment = ase.run_ga_experiments(file, modelo="PLSDA", budget=50)

print(best_experiment)