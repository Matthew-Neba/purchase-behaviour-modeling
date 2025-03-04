"""


3)
Task: Binary Classification


Data Background:

This dataset from hugging face is based on the "Diabetes 130-US hospitals for years 1999-2008 Data Set," . It encompasses over a decade of clinical care data from 130 U.S. hospitals pertaining to diabetes patients. Each sample in the dataset represents a hospital admission involving a patient diagnosed with diabetes, capturing various aspects of their hospital encounter.
Response Variable: Readmitted variable, which indicates whether a patient was readmitted to the hospital after discharge. This variable can have a value of 0 or 1.

Features:
Demographics:
race: Patient's race (e.g., Caucasian, African American, Asian, Hispanic, Other)
gender: Patient's gender (Male, Female)
age: Age group of the patient, binned in the following intervals (e.g., [0-10), [10-20),[20-50], [50-70])
Admission Details:
admission_type: Type of admission (e.g., Emergency, Trauma Center, Elective, …)
admission_source: Source of admission (e.g., Referral, Emergency Room, Transfer from another hospital, Other)
discharge_disposition: Disposition of the patient at discharge (e.g., Discharged to home, Other)
medical_specialty: Integer identifier of a specialty of the admitting physician, corresponding to 84 distinct values, for example, cardiology, internal medicine, family/general practice, and surgeon
Medical History and Diagnoses:
time_in_hospital: Length of stay in days
num_lab_procedures: Number of lab tests performed during the encounter
num_procedures: Number of procedures (excluding lab tests) performed
num_medications: Number of distinct medications administered
number_diagnoses: Number of diagnoses recorded for the patient
Number_outpatient: Number of outpatient visits of the patient in the year preceding the encounter
Number_inpatient: Number of inpatient visits of the patient in the year preceding the encounter
Number_emergency: Number of emergency visits of the patient in the year preceding the encounter
diag_1, diag_2, diag_3: Primary, secondary, and tertiary diagnoses (coded using ICD-9 codes)
Laboratory Results:
max_glu_serum: Maximum glucose serum test result (e.g., >200, >300, Normal, None)
A1Cresult: Hemoglobin A1c test result (e.g., >7, >8, Normal, None)

Drugs:
The following drugs were included in the dataset with four categorical variables corresponding to whether or not the dosage of the drug was increased for the patient during their visit, remained steady, was decreased by the hospital or the patient is not being administered the drug

The Drugs are: metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, glipizide, glyburide, pioglitazone, rosiglitazone, acarbose, miglitol, tolazamide, insulin, glyburide-metformin, 

A critical pattern to explore within this dataset is the relationship between various patient and treatment factors and the likelihood of early readmission (within 30 days). Understanding this relationship can aid in identifying high-risk diabetes patients and developing targeted interventions to reduce readmission rates.


Data Preprocess:
https://github.com/csinva/imodels-data/blob/master/notebooks_fetch_data/00_get_datasets_custom.ipynb

Model Construction: 
SVM binary classification with polynomial kernel using internal cross validation for degree selection

"""


from datasets import load_dataset

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ds = load_dataset("imodels/diabetes-readmission")

train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()

X_train = train_df.drop(columns=["readmitted"])
y_train = train_df["readmitted"]

# limit dataset to random 1000 samples, remove later
X_train = X_train.sample(n=2000)
y_train = y_train.loc[X_train.index]

X_test = test_df.drop(columns=["readmitted"])
y_test = test_df["readmitted"]


model = SVC(kernel="poly")
param_grid = {
    "degree": [2, 3, 4 ],
    "C": [0.1, 10, 100, 1000],
}   

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-2)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Best Params: {grid_search.best_params_}")
print("Confusion Matrix:")
print(conf_matrix)


# check nature of dataset, This dataset is quite janky, work on it later
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))


