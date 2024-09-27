from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, median_absolute_error, explained_variance_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
import csv
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load Data
data = pd.read_csv('../../../Downloads/new_sigs/SBS96_catalogue.TCGA-CA-6717-01.hg19.tally.csv')
print(data.info())

X = data.drop(columns=['channel', 'type', 'count'])
y = data['count']

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train rfc
rf_regressor = RandomForestClassifier(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)

# Print evaluation results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Median Absolute Error: {medae:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"Explained Variance Score: {explained_var:.2f}")

# Plotting predicted vs actual counts
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)
plt.xlabel('Actual Counts')
plt.ylabel('Predicted Counts')
plt.title('Actual vs Predicted Counts')
plt.show()

# According to chat gpt - for TCGA-CA-6717-01.tally (SBS96)
# The high R² (0.99) means that your model is doing an excellent job at explaining the variation in the data.
# 
# The MSE (41.38) suggests that while the predictions are close, there’s still an average deviation from the 
# true values, which you may want to reduce by further tuning the model or using more advanced techniques 
# (if the MSE is deemed high based on your specific domain knowledge).
#
# MAE indicates that, on average, your model's predictions are off by 3.24 units. Since MAE doesn't square the errors, 
# it is less sensitive to outliers compared to MSE or RMSE. This value suggests your model is generally quite accurate.
#
# This shows that the median error in your predictions is just 1 unit. This suggests that half of your predictions are 
# within 1 unit of the actual value, which is a very strong result and indicates that most predictions are very accurate.
#
# RMSE is the square root of MSE, giving an error measure in the same units as the count values. On average, the model's 
# predictions are about 6.43 units off from the actual counts.
#
# Explained Variance Score: 0.99 - This is another strong indicator that your model captures nearly all of the variability 
# in the data, much like R². 
# 
# Your model is performing exceptionally well based on these metrics, especially given the high R² and Explained Variance Score.
# The relatively low RMSE and MAE suggest that your predictions are close to the actual counts, and the Median Absolute Error of 1 
# unit implies that most predictions are very accurate.


