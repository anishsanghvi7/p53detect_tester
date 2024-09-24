from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import train_test_split

X,y = load_breast_cancer(returnX_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0, stratify = y)

clf = RandomForestClassifier(max_depth = 8, random_state = 0)
clf = clf.fit(X_train, y_train)