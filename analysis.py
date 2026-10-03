import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.stat_tests import get_shapiro, get_ttest

# Set experiment names and fetch AUCs
experimentId = 'DM-1A'
datasetA = 'Ghana'
datasetB = 'MIAS'
setA_aucs = []
setB_aucs = []

# Perform Shapiro test for normality
shapiro_df = pd.DataFrame({
    'ID': ['Dataset',
           'Mean',
           'Stat',
           'p_value',
           'Outcome']
})
setA_shapiro = pd.DataFrame({experimentId: get_shapiro(datasetA, setA_aucs)})
setB_shapiro = pd.DataFrame({experimentId: get_shapiro(datasetB, setB_aucs)})
shapiro_df = pd.concat([shapiro_df, setA_shapiro, setB_shapiro], axis=1)

# Perform t-test for dataset bias
bias_ttest_df = pd.DataFrame({
    'ID': ['Primary dataset',
           'Secondary dataset',
           'Primary Mean',
           'Secondary Mean',
           'Stat',
           'p_value',
           'Outcome']
})
bias_ttest = pd.DataFrame({experimentId: get_ttest(datasetA, datasetB, setA_aucs, setB_aucs)})
bias_ttest_df = pd.concat([bias_ttest_df, bias_ttest], axis=1)

# Save results to CSV
shapiro_df.to_csv('shapiro.csv')
bias_ttest_df.to_csv('bias_ttest.csv')

# Plot KDE and save image
sns.kdeplot(setA_aucs, label=datasetA, fill=True, color='red', alpha=0.5)
sns.kdeplot(setB_aucs, label=datasetB, fill=True, color='blue', alpha=0.5)
plt.title(f'Experiment {experimentId}: Distribution of AUCs over 50 iterations')
plt.xlabel('AUC')
plt.ylabel('Frequency Density')
plt.legend()
plt.savefig(f'Experiment{experimentId}_distribution.png')
plt.clf()
