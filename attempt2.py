# Importing the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, median_absolute_error, explained_variance_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('../../../Downloads/new_sigs/SBS96_catalogue.TCGA-CA-6717-01.hg19.tally.csv')
data_encoded = pd.get_dummies(data, columns=['channel', 'type'])

# Define the features (X) and the target (y)
X = data_encoded.drop(columns=['count'])
y = data_encoded['count']

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nX_train shape: ", X_train.shape)   # Train features (without label)
print("y_train shape: ", y_train.shape)     # Train label of samples
print("X_test shape: ", X_test.shape)       # Test features (without label)
print("y_test shape: ", y_test.shape)       # Train label of samples

print("\n--------------------------\n")

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the classifier
rf_classifier.fit(X_train, y_train)
# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Median Absolute Error: {medae:.3f}")
print(f"R-squared (R²): {r2:.3f}")
print(f"Explained Variance Score: {explained_var:.3f}\n")
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("--------------------------\n")

#### HYPERPARAMETER SELECTION ####
print("Parameters available : ", rf_classifier.get_params())

## The following values are only applicable for random forests
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
'max_features': max_features,
'max_depth': max_depth,
'min_samples_split': min_samples_split,
'min_samples_leaf': min_samples_leaf,
'bootstrap': bootstrap}
print("\nParameter values for testing : ", random_grid)

print("\n--------------------------\n")

rf_random = RandomizedSearchCV(estimator = rf_classifier, param_distributions = random_grid,
n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
rf_grid = rf_random.fit(X_train, y_train)

print("\nBest parameters : ", rf_grid.best_params_)
best_random_rf = rf_grid.best_estimator_                # Save best hyperparameters model
y_pred_test_random = best_random_rf.predict(X_test)     # Predict labels for the test set features

print("\n--------------------------\n")

## Accuracy
print("Accuracy score original: ", accuracy_score(y_test, y_pred))
print("Balanced accuracy score original: " , balanced_accuracy_score(y_test, y_pred), '\n')

print("Accuracy score best hyperparameters: ", accuracy_score(y_test, y_pred_test_random))
print("Balanced accuracy score best hyperparameters: " , balanced_accuracy_score(y_test, y_pred_test_random))

## Classification report
print("\nClassification report :")
print(classification_report(y_test, y_pred_test_random, zero_division=0))

print("--------------------------\n")

print(f"Model Performance: {rf_classifier.score(X_test, y_test):.5f} \n")