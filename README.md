Wine Quality Prediction Project

Overview

This project focused on predicting wine quality using both classification and regression models on red and white wine datasets. The goal was to explore different machine learning approaches and understand their effectiveness in predicting wine quality.

Datasets

Red Wine Dataset: Used for classification tasks.

White Wine Dataset: Used for regression tasks.

Key Learnings

1. Data Preprocessing

Standardization: Applied StandardScaler to normalize feature distributions.

Log Transformation: Reduced skewness in features with right-skewed distributions.

Outlier Removal: Used the Interquartile Range (IQR) method to eliminate extreme values.

2. Model Development

Classification (Red Wine Dataset)

Models used: Logistic Regression, Random Forest Classifier, Support Vector Machine (SVM).

Achieved 98% accuracy with Random Forest, highlighting that the dataset is well-suited for classification due to its discrete target variable.

Regression (White Wine Dataset)

Models used: Linear Regression, Random Forest Regressor, Support Vector Regressor (SVR).

The best R² score was 0.48, indicating limited predictive power for regression.

Grid Search was used for hyperparameter tuning, improving model performance but still constrained by the data.

3. Model Evaluation

Classification: Evaluated using Accuracy, Precision, Recall, and F1-score.

Regression: Evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.

Conclusions

Classification Models Outperformed Regression Models: The red wine dataset's discrete quality ratings are more suitable for classification tasks.

Regression Models Struggled: The white wine dataset did not yield high R² scores, likely due to the ordinal nature of the target variable and potential missing features.

Data Quality Matters: More comprehensive data (e.g., sensory scores, production details) could improve regression performance.

Future Work

Feature Engineering: Create new features or interaction terms to capture hidden patterns.

Advanced Models: Experiment with models like Gradient Boosting or XGBoost.

Classification for White Wine: Reframe the white wine problem as a classification task.

External Data Integration: Include more relevant data, such as vineyard conditions or grape types.

Model Storage

The best models were saved for future reference in the SavedModels/ folder using joblib.

Acknowledgments

This project provided hands-on experience with preprocessing, model development, evaluation, and improvement strategies in machine learning.