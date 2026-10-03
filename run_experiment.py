import numpy as np
import pandas as pd
from train_and_test import train_and_test
from utils.load_data import load_ghana_dataset, load_breast_ultrasound

# Set experiment names (modify for each experiment)
experimentId = 'Experiment1'
datasetA = 'Ghana'
datasetB = 'BrEaST'

# Load datasets (modify for each experiment)
print("Loading data...")
spectra_X, spectra_y = load_ghana_dataset('spectra', 'ultrasound')
dcl_X, dcl_y = load_ghana_dataset('dcl', 'ultrasound')
setA_X = np.concatenate((spectra_X, dcl_X), axis=0)
setA_y = np.concatenate((spectra_y, dcl_y), axis=0)
setB_X, setB_y = load_breast_ultrasound()
print("Data loaded!")

# Initialise dataframes to save results
row_names = pd.DataFrame({'Metrics': ['Accuracy', 'Precision', 'Recall', 'AUC']})
metrics_report_A, metrics_report_B, metrics_report_prebalA = row_names, row_names, row_names

# Train and test over i iterations
for i in range(0, 50):
    print(f"Starting iteration {i}")

    # Train and test
    setA_metrics, setB_metrics, prebalA_metrics = train_and_test(i, setA_X, setA_y, setB_X, setB_y)

    # Store metrics
    metrics_report_A = pd.concat([metrics_report_A, setA_metrics], axis=1)
    metrics_report_B = pd.concat([metrics_report_B, setB_metrics], axis=1)
    metrics_report_prebalA = pd.concat([metrics_report_prebalA, prebalA_metrics], axis=1)

# Save results in CSV files
print("Saving results...")
filenameA = experimentId + '_' + datasetA + '_metrics_report.csv'
metrics_report_A.to_csv(filenameA, index=False)

filenameB = experimentId + '_' + datasetB + '_metrics_report.csv'
metrics_report_B.to_csv(filenameB, index=False)

filenameAprebal = experimentId + '_' + datasetA + 'prebal_metrics_report.csv'
metrics_report_prebalA.to_csv(filenameAprebal, index=False)
print("Results saved!")
