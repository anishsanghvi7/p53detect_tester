from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # import other methods here
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV

#### LOAD DATA ####
# Replace X and y with your own data or another toy set
X,y=load_breast_cancer(return_X_y=True,as_frame=True)

# Change test size to reflect how big you want you test set to be (0.2-0.4 range)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=0, stratify=y)

print("X_train shape : ", X_train.shape) # Train features (without label)
print("y_train shape : ", y_train.shape) # Train label of samples
print("X_test shape : ", X_test.shape) # Test features (without label)
print("y_test shape : ", y_test.shape) # Train label of samples


#### TRAIN MODELS ####
# Initialise your classifier (you can change this line to another method - change hyperpa)
clf = RandomForestClassifier(max_depth=8, random_state=0)
clf = clf.fit(X_train, y_train) # Train model with training data and labels
y_pred_test = clf.predict(X_test) # Predict labels for the test set features


#### HYPERPARAMETER SELECTION ####
print("Parameters available : ", clf.get_params())

## The following values are only applicable for random forests
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
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
print("Parameter values for testing : ", random_grid)

# Test the grid of hyperparameters for the best combination
clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid,
n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
clf_grid = clf_random.fit(X_train, y_train)

print("Best parameters : ", clf_grid.best_params_)
best_random_clf = clf_grid.best_estimator_ # Save best hyperparameters model
y_pred_test_random = best_random_clf.predict(X_test) # Predict labels for the test set features


#### ASSESS MODEL PERFORMANCE ####
## Accuracy
print("Accuracy score original: ", accuracy_score(y_test, y_pred_test))
print("Balanced accuracy score original :" , balanced_accuracy_score(y_test, y_pred_test))

print("Accuracy score best hyperparameters: ", accuracy_score(y_test, y_pred_test_random))
print("Balanced accuracy score best hyperparameters:" , balanced_accuracy_score(y_test, y_pred_test_random))

## Confusion matrix
cm = confusion_matrix(y_test, y_pred_test_random, labels=best_random_clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp.plot()
plt.show()

## Classification report
print("Classification report :")
print(classification_report(y_test, y_pred_test_random))

## Precision recall curve
disp = PrecisionRecallDisplay.from_estimator(best_random_clf, X_test, y_test)
disp.plot()
plt.ylim([0, 1])
plt.show()


#### RECURSIVE FEATURE ELIMINATION ####
# Note: Restarting with a blank model
rfc = RandomForestClassifier(max_depth=8, random_state=0)
# Remove one feature each step from the model (cross-validated 10 times)
rfc = RFECV(rfc, step=1, cv=10)
rfc = rfc.fit(X_train, y_train)
y_pred_test = rfc.predict(X_test)

print("Optimal number of features : ", rfc.n_features_)
print("Ranking of features : ",rfc.ranking_)
print("Importance of features ranked #1 : ",rfc.estimator_.feature_importances_)