# Importing the required libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, median_absolute_error, explained_variance_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the dataset
folder_path = '../../../Downloads/new_sigs/'
sig_data = pd.DataFrame()

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    # Filter to match the pattern 'SBS96_catalogue.<sample_name>.hg19.tally.csv' -  only doing sbs first
    if filename.startswith('SBS96_catalogue.') and filename.endswith('.hg19.tally.csv'):
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path, index_col=None)

        # Extract the sample name from the filename (e.g., 'TCGA-CA-6717-01')
        sample_name = filename.split('.')[1]
        select_data = data.drop(columns=['type', 'count'])

        # Pivot the DataFrame so that 'channel' becomes the columns
        pivot_data = select_data.pivot_table(index=None, columns="channel", values="fraction").reset_index(drop=True)
        pivot_data.insert(0, 'sample', sample_name)
        sig_data = pd.concat([sig_data, pivot_data], ignore_index=True)

# Add final column to data (empty) - maybe dont do it here??
sig_data["p53 status"] = ""

# Display the final DataFrame or save it to a file
print(sig_data)
print("\n--------------------------\n")

############ downloaded maf

# maf_file_path = '../../../Downloads/October_2016_whitelist_2583.snv_mnv_indel.maf.xena.nonUS'
# maf_data = pd.read_csv(maf_file_path, sep='\t', comment='#')
# # print(maf_data)
# try_filter = maf_data.query('gene == "TP53"')
# print(try_filter)

############# EU  File
new_file_tester =  '../../../Downloads/UMD_variants_EU.xlsx'
tester_data =  pd.read_excel(new_file_tester)
temp = tester_data[tester_data['Mutational_event'].str.contains('^[A, C, G, T]>[A, C, G, T]$') & 
                    tester_data['WT_Codon'].str.contains('^[ACGT]{3}$') & 
                    tester_data['Mutant_Codon'].str.contains('^[ACGT]{3}$')]

temp['Signature'] = temp.apply(lambda row: row['Mutant_Codon'][0] + '[' + row['Mutational_event'] + ']' + row['Mutant_Codon'][2], axis=1)
filtered = temp[['Variant_Classification', 'Pathogenicity', 'Final comment',  'Signature']]

grouped_signature_data = filtered.groupby(['Signature', 'Pathogenicity']).size().reset_index(name='Count')

pathogenicity_mapping = {
    'Pathogenic': 1,
    'Likely Pathogenic': 0.75,
    'Possibly pathogenic': 0.5,
    'VUS': 0.25,
    'Benign': 0
}

grouped_signature_data['Pathogenicity_Score'] = grouped_signature_data['Pathogenicity'].map(pathogenicity_mapping)

weighted_likelihood = grouped_signature_data.groupby('Signature').apply(
    lambda x: (x['Count'] * x['Pathogenicity_Score']).sum() / x['Count'].sum()
).reset_index(name='Pathogenicity_Likelihood')

# Print the DataFrame
print(weighted_likelihood)

print("\n--------------------------\n")

final_data_temp = sig_data.melt(id_vars=['sample'], var_name='Signature', value_name='Contribution')

# Merge the melted data with likelihoods
merged_data = pd.merge(final_data_temp, weighted_likelihood, on='Signature', how='left')
merged_data['Weighted_Pathogenicity'] = merged_data['Contribution'] * merged_data['Pathogenicity_Likelihood']

# Step 4: Sum the weighted likelihood for each sample
sample_pathogenicity = merged_data.groupby('sample')['Weighted_Pathogenicity'].sum().reset_index()

def map_pathogenicity(likelihood):
    if likelihood >= pathogenicity_mapping['Pathogenic']:
        return 'Pathogenic'
    elif likelihood >= pathogenicity_mapping['Likely Pathogenic']:
        return 'Likely Pathogenic'
    elif likelihood >= pathogenicity_mapping['Possibly pathogenic']:
        return 'Possibly pathogenic'
    elif likelihood >= pathogenicity_mapping['VUS']:
        return 'VUS'
    else:
        return 'Benign'

sample_pathogenicity['Pathogenicity_Category'] = sample_pathogenicity['Weighted_Pathogenicity'].apply(map_pathogenicity)

print(sample_pathogenicity)
print("\n--------------------------\n")

X = sample_pathogenicity.drop(columns=['sample', 'Pathogenicity_Category'])
y = sample_pathogenicity['Pathogenicity_Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))