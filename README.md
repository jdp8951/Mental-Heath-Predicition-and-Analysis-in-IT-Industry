# Mental-Heath-Predicition-and-Analysis-in-IT-Industry


## Project Overview
This project explores the mental health challenges faced by individuals in the tech industry. Using datasets, we aim to analyze various factors contributing to mental health issues and identify potential solutions to improve overall well-being in this sector.

## Table of Contents
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)


## Motivation
The tech industry is known for its fast-paced environment, which can lead to increased stress and mental health issues among employees. This project aims to shed light on these challenges and promote awareness and solutions.

## Dataset
We utilized datasets from Open Sourcing Mental Illness (OSMI) surveys conducted between 2017 and 2021. These datasets include responses from IT professionals, focusing on demographics, work factors, and mental health indicators. Key preprocessing steps included renaming columns, handling missing data, and feature scaling.
- **`mental_health.csv`**: Contains 8,500 rows of data.

## Methodology
- Data cleaning and preprocessing.
- Exploratory data analysis to identify trends and patterns.
- Implementation of machine learning models to predict mental health outcomes.
- Models :
    -Logistic Regression: Simple, interpretable, used for binary outcomes.
    -Random Forest: Ensemble method using multiple decision trees for high accuracy.
    -K-Nearest Neighbors (KNN): Non-parametric, groups data points by proximity.
    -Gradient Boosting Classifier: Sequential model boosting to reduce errors.
    -XGBoost: Optimized gradient boosting technique for superior performance.

## Results
### Model Performance

| Model               | Precision | Recall  | F1 Score | Accuracy |
|---------------------|-----------|---------|----------|----------|
| K-Nearest Neighbor  | 0.95      | 0.98    | 0.97     | 97.67%   |
| Logistic Regression | 0.68      | 0.52    | 0.59     | 74.49%   |
| XGBoost             | 0.84      | 0.69    | 0.76     | 84.52%   |
| Random Forest       | 1.00      | 0.98    | 0.99     | 99.35%   |
| Gradient Boosting   | 0.86      | 0.70    | 0.77     | 85.39%   |



### Conclusion
The Project uses  machine learning models, particularly ensemble methods, are effective in predicting mental health issues in the tech industry. The Random Forest model emerged as the most reliable, achieving the highest accuracy, precision, recall, and F1-score. KNN also performed well, while models like XGBoost and Gradient Boosting, though competitive, fell slightly behind Random Forest in robustness.

Logistic Regression struggled with the dataset's complexity, indicating that more advanced methods are needed to capture non-linear relationships in mental health prediction. This study highlights the importance of selecting appropriate machine learning models for mental health data, with Random Forest proving to be the most successful. Future work could explore additional ensemble methods or deep learning techniques to further improve predictive accuracy and support mental health interventions in the tech sector.




Let me know if you need help with any specific sections or additional information!

