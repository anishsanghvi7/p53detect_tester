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
final_data = pd.DataFrame()

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
        final_data = pd.concat([final_data, pivot_data], ignore_index=True)

# Add final column to data (empty) - maybe dont do it here??
final_data["p53 status"] = ""

# Display the final DataFrame or save it to a file
print(final_data)
print("--------------------------\n")

# Save the final DataFrame to a CSV file
# final_data.to_csv(os.path.join(folder_path, 'combined_samples.csv'), index=False)

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

print(tester_data)
print("--------------------------\n")
print(filtered)
