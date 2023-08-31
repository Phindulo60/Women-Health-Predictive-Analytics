# Women's Health Predictive Analytics

**Repository owner**: Phindulo60

## Overview

This repository contains a Python script that performs data analysis on a dataset related to women's health. The dataset, cerv_ca.csv, contains various health metrics and the script performs exploratory data analysis (EDA) to understand the distribution and relationships between different variables.
The project further aims to analyze the risk factors associated with cervical cancer. By leveraging machine learning techniques, we delve into feature selection and data preprocessing to identify the most significant predictors of cervical cancer and HPV diagnosis.

## Libraries Used

The dataset consists of the following columns:

- pandas
- numpy
- glob
- os
- pathlib
- datetime
- matplotlib
- seaborn
- cvxpy
- scipy
- ...and many more


## Dataset Description

The dataset cerv_ca.csv contains the following columns:

- Age
- Number of sexual partners
- First sexual intercourse
- Num of pregnancies
- Smokes
- Smokes (years)
- Smokes (packs/year)
- Hormonal Contraceptives
- Hormonal Contraceptives (years)
- IUD
- ... (and many more)

Analysis Steps
Data Loading: The dataset is loaded into a pandas DataFrame.
Data Description: Basic statistics like count, mean, standard deviation, min, 25th percentile, median, 75th percentile, and max are computed for each column.
Data Cleaning: Missing values are identified and handled. Data types of columns are verified and corrected if necessary.
Exploratory Data Analysis (EDA):
A heatmap is generated to visualize the correlation between different variables.
Strong correlations between variables are identified.
Histograms are plotted for variables like Age, Number of sexual partners, and Num of pregnancies to understand their distribution.

## Steps Covered in the Notebook

1. **Data Loading**: The dataset is loaded into pandas dataframes.
2. **Data Description:**: Basic statistics like count, mean, standard deviation, min, 25th percentile, median, 75th percentile, and max are computed for each column.
3. **Data Cleaning:**:  Missing values are identified and handled. Data types of columns are verified and corrected if necessary.
4. **Exploratory Data Analysis (EDA):**: 
   - A heatmap is generated to visualize the correlation between different variables.
   - Strong correlations between variables are identified.
   - Histograms are plotted for variables like Age, Number of sexual partners, and Num of pregnancies to understand their distribution.


## Data Exploration
The dataset contains various features related to the patient's medical history, lifestyle, and previous diagnoses. Initial exploration provides insights into data distribution, missing values, and basic statistics.

## Checking for Class Imbalance
Class imbalance can lead to biased model predictions. We checked the distribution of the target variables Dx:Cancer and Dx:HPV. Both showed a significant imbalance, with a majority of the samples being negative.

##  Fixing the Imbalance
To address the imbalance, we employed the RandomOverSampler from the imblearn library. This method oversamples the minority class, ensuring an equal distribution of both classes.

## Feature Selection
Two methods were used for feature selection:

- Recursive Feature Elimination (RFE): Using Logistic Regression as the underlying model, RFE identified a subset of 5 features that were most relevant for predicting cancer diagnosis.

- Random Forest Feature Importance: A Random Forest classifier was used to rank features based on their importance. A bar plot visualized the relative importance of each feature.

Additionally, a correlation matrix heatmap was generated to understand the relationships between different features.

## Feature Subset Creation
Based on the results from the feature selection methods, a new dataset was created containing only the most important features. This subset will be used for subsequent model training and evaluation.

## Data Splitting
The dataset was split into training, validation, and test sets. This was done separately for both Dx:Cancer and Dx:HPV targets. The splits were:

- Training set: 60%
- Validation set: 20%
- Test set: 20%

## Logistic Regression
Logistic Regression was trained for both targets. The maximum number of iterations was set to 10,000 to ensure convergence. The performance of the models was evaluated using classification reports, which provide precision, recall, f1-score, and support for both classes.

Results:

- For Dx:Cancer: The model achieved an accuracy of 94%.
- For Dx:HPV: The model achieved an accuracy of 92%.

## Evaluating Multiple Models
Several classifiers were trained and evaluated:

- Support Vector Classifier (SVC)
- Decision Tree Classifier
- K-Nearest Neighbors Classifier
- Random Forest Classifier
- Multi-layer Perceptron Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier
The performance of each classifier was evaluated on the training, test, and validation datasets for both targets. The results were stored in a dataframe for easy comparison.

## Model Tuning
### Decision Tree Regressor
The Decision Tree Regressor was tuned using GridSearchCV for both targets. The best hyperparameters were identified and the models were retrained using these parameters.

Results:

- For Dx:Cancer: The model achieved an R-Squared value of 0.8332.
- For Dx:HPV: The model achieved an R-Squared value of 0.7529.

### Random Forests
The Random Forest Regressor was also tuned using GridSearchCV for both targets. The best hyperparameters were identified and the models were retrained using these parameters.

Results:

- For Dx:Cancer: The model achieved an R-Squared value of 0.8387.
- For Dx:HPV: The model achieved an R-Squared value of 0.7585.

### Gradient Boosting Regression
The Gradient Boosting Regressor was tuned using GridSearchCV for both targets. The best hyperparameters were identified and the models were retrained using these parameters.

Results:

- For Dx:Cancer: The model achieved an R-Squared value of -0.0427.
- For Dx:HPV: The model achieved an R-Squared value of -0.0769.

## How to Use

1. Clone the repository.
2. Ensure you have the required libraries installed (pandas, numpy, scikit-learn, gensim, etc.).
3. Open the Jupyter notebook and run the cells sequentially.
