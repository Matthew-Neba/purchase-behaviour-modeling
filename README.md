# SVM Classifier with Polynomial Kernel and GridSearchCV for Hyperparameter Tuning

## Project Overview
This project implements a Support Vector Machine (SVM) classifier with a polynomial kernel to classify website visitor data. The model aims to predict whether a session resulted in a purchase based on various features. Hyperparameter tuning is performed using GridSearchCV to optimize the model.

## Key Features
- **SVM Classifier with Polynomial Kernel**: Used for classification of visitor sessions.
- **Hyperparameter Tuning with GridSearchCV**: Optimizes model parameters for better accuracy.
- **Data Preprocessing Pipelines**: Handles numerical and categorical feature transformations.
- **Permutation Importance Analysis**: Assesses feature impact on predictions.
- **Model Evaluation & Visualization**: Includes confusion matrix, learning curves, and feature distributions.

## Files in the Project
- `ecommerce_notebook.ipynb`: Jupyter notebook containing the complete implementation.
- `requirements.txt`: List of dependencies required to run the project.

## How to Run
1. Ensure you have Python installed.
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Open and run `ecommerce_notebook.ipynb` in Jupyter Notebook or execute the script in a Python environment.

## Steps in the Project
1. **Import Necessary Libraries**
2. **Load and Split the Data**: Uses `train_test_split` for splitting.
3. **Preprocess Data**: Uses pipelines for numerical and categorical features.
4. **Define Model and Hyperparameter Grid**: Sets up SVM classifier with polynomial kernel.
5. **Train Model with GridSearchCV**: Optimizes parameters through cross-validation.
6. **Evaluate Model on Test Set**: Computes accuracy and feature importance.
7. **Visualizations**:
   - Feature distributions (numerical & categorical)
   - Correlation heatmap
   - Confusion matrix
   - Learning curve for training sample sizes

## Results
- Best hyperparameters selected using GridSearchCV.
- Training and test accuracy scores.
- Feature importance ranking via permutation importance.
- Learning curve visualization showing model performance over different sample sizes.

## Conclusion
This project provides a structured approach to classifying website visitor sessions using an SVM classifier with a polynomial kernel. Through hyperparameter tuning and feature importance analysis, it offers insights into the most relevant factors influencing purchase decisions.

## Author
Matthew Neba