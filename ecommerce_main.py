"""

A SVM classifier with a polynomial kernel was used. 5-fold internal cross validation was used to determine the best cost,degree pair for the SVM classifier. The metric during interval cross-validation was the accuracy of the classifier. The degrees used belonged in :  [2, 3, 4, 7 ], the costs used belonged in : [0.1, 10, 100, 1000]. 


Features:
Administrative: Number of pages visited related to account management.
Administrative_Duration: Total time (in seconds) spent on account management pages.
Informational: Number of pages visited containing website, communication, and address information.
Informational_Duration: Total time (in seconds) spent on informational pages.
ProductRelated: Number of pages visited related to product information.
ProductRelated_Duration: Total time (in seconds) spent on product-related pages.
BounceRates: Average bounce rate of the pages visited.
ExitRates: Average exit rate of the pages visited.
PageValues: Average value of the pages visited.
SpecialDay: Closeness of the visit to a special day (e.g., Mother's Day, Valentine's Day).
Month: Month of the visit.
OperatingSystems: Operating system used by the visitor.
Browser: Browser used by the visitor.
Region: Geographic region from which the session originated.
TrafficType: Traffic source (e.g., banner, SMS, direct).
VisitorType: Type of visitor (e.g., new, returning).
Weekend: Boolean indicating whether the visit occurred on a weekend.


Response Variable:
Revenue: Binary variable indicating whether the session resulted in a purchase ('True' or 'False').

"""

from datasets import load_dataset

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


ds = load_dataset("jlh/uci-shopper")

df = ds["train"].to_pandas()

X = df.drop(columns=["Revenue"])
y = df["Revenue"]

# # limit data for local testing
X = X.sample(n=100)
y = y.loc[X.index]



# train test split 20% of data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# preprocess numerical features
numerical_features = X.select_dtypes(include=["float64", "int64"]).columns

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())      
])

#preprocess categorical features
categorical_features = X.select_dtypes(include=["object", "bool"]).columns

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# combine the two preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# model construction
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="poly"))
])  

#hyperparameter grid for grid search
param_grid = {  
    "classifier__degree": [2, 3, 4, 7 ],
    "classifier__C": [0.1, 10, 100, 1000],
}


# Set up GridSearchCV with the pipeline, parameter grid, and cross-validation strategy, 5-fold internal cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5 , scoring='accuracy', n_jobs=-2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Retrieve the best parameters and the corresponding score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best parameters: {best_params}')
print(f'Training Accuracy: {best_score:.4f}')

# Evaluate the model on the test set
y_pred = grid_search.predict(X_test)
test_score = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {test_score:.4f}')



from sklearn.inspection import permutation_importance
import pandas as pd

# Compute feature importance
result = permutation_importance(grid_search, X_test, y_test, scoring="accuracy", n_repeats=10, random_state=42)

# Sort and display the feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": result.importances_mean
}).sort_values(by="Importance", ascending=False)

print(feature_importance)
