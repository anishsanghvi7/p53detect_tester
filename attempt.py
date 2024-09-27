from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
import csv
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load Data
data = pd.read_csv('../../../Downloads/new_sigs/SBS96_catalogue.TCGA-CA-6717-01.hg19.tally.csv')
print(data.head()) 
print(data.info())

X = data.drop(columns=['channel', 'type'])
y = data['channel']

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train rfc
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#  Make predictions then evaluate model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print evaluation
print(f"Accuracy: {accuracy * 100:.2f}%")
# print(classification_report(y_test, y_pred))