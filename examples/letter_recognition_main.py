""" 
This will be a multi-class classification problem, as the target variable has 26 unique classes (letters of the alphabet). Use SVM with polynomial kernel and internal cross validation to select the best degree for the polynomial kernel and cost of the SVM. 

Since SVM is inherently a binary classifier, we will use the one-vs-all strategy to extend it to multi-class classification. one vs one is too computationally expensive since  n(n-1)/2 classifiers are trained for n classes. But for one vs all, only n classifiers are trained.

"""
from datasets import load_dataset

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

ds = load_dataset("wwydmanski/tabular-letter-recognition")

train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()


X_train = train_df.drop(columns=["target"])
Y_train = train_df["target"]

X_test = test_df.drop(columns=["target"])
Y_test = test_df["target"]

model = SVC(kernel="poly", decision_function_shape="ovr")
param_grid = {
    "degree": [2, 3, 4 ],
    "C": [0.1, 10, 100, 1000],
}

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-2)

grid_search.fit(X_train, Y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

Y_pred = best_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)

print(f"Accuracy: {accuracy}")
print(f"Best Params: {grid_search.best_params_}")
print("_____________________________________")
print("Confusion Matrix:")
print(conf_matrix)


