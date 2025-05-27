## Classification of Laterite Type using Random Forest

### Overview
This project focuses on predicting the type of laterite using a Random Forest Classifier based on several geotechnical properties. Accurate classification plays a vital role in mining, geotechnical engineering, and construction industries by helping experts understand rock behavior, site suitability, and excavation risks.

The model has been trained and tested to achieve over 95% accuracy, with strong generalization on unseen data.

### Dataset
The dataset contains 500 real-world samples with the following geotechnical features:

Feature	Description

Ds	Dry Density, 
UCS	Uniaxial Compressive Strength, 
IS50	Point Load Index at 50 mm diameter,
TS	Tensile Strength,
Pw	Porosity,
Di	Degree of Saturation,
Mc	Moisture Content,
RQD	Rock Quality Designation,

Target Variable: Laterite Type (Categorical — represents the class/type of laterite)

### Objective
Build a high-performing Random Forest model to classify Laterite Type.

Understand the influence of each geotechnical property.

Enable user-friendly prediction interface for practical use in mining or field assessments.

### Methodology

1.Data Preprocessing
Handled missing values, Performed feature scaling and normalization, Categorical encoding of target variable.

2.Exploratory Data Analysis (EDA)
Distribution plots for each feature, Correlation heatmap to identify relationships, Outlier detection and treatment using IQR/Z-score, Feature importance visualization

3.Model Building
Implemented a Random Forest Classifier, Applied Grid Search Cross-Validation for hyperparameter tuning,Used K-Fold Cross Validation to ensure robust model performance

4.Evaluation Metrics
Accuracy: ~95%
Confusion Matrix
Precision / Recall / F1 Score
Feature Importance Plot

5.User Input & Prediction
The project includes a simple interactive Python script / Jupyter Notebook that takes user inputs for geotechnical parameters and predicts the Laterite Type

![image](https://github.com/user-attachments/assets/06a4df89-c9bb-4052-91eb-7624ec94bbe6)

![image](https://github.com/user-attachments/assets/a11a9f33-03ac-4e0b-a971-5276909be4fc)

