import auto_spectra_engine.experiment as ase

#file = "/home/barbon/PycharmProjects/AutoSpectraEngine/auto_spectra_engine/datasets/raman.csv"
file = "/home/barbon/Python/AutoSpectraEngine/auto_spectra_engine/datasets/raman.csv"
#file = "/home/barbon/Python/AutoSpectraEngine/auto_spectra_engine/datasets/NIR1.csv"
#print(run_experiment(file, modelo="PLSDA", coluna_predicao="Class", plot=False, verbose=True))
ase.run_all_experiments(file, modelo="PLSDA", coluna_predicao="Adulterant")
