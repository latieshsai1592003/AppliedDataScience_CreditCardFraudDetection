# Credit Card Fraudlent Detection using Machine Learning README

This Jupyter Notebook provides a machine learning model to predict whether the transaction is Normal or Fraudlent. The code includes data preprocessing, exploratory data analysis, data cleaning, and model building. Below are the steps to run this code:

## Prerequisites
1. You need to have Python installed on your system.
2. Install Jupyter Notebook and the required libraries using pip:
   ```bash
   pip install jupyter numpy pandas seaborn matplotlib scikit-learn scipy
   ```

## Steps to Run the Code
1. Clone or download the Jupyter Notebook file and the dataset ('creditcard.csv') to your local machine.
2. Open a terminal and navigate to the directory containing the Jupyter Notebook and the dataset.
3. Start a Jupyter Notebook session:
   ```bash
   jupyter notebook
   ```
4. In the Jupyter Notebook dashboard, open the 'ADS_Phase5.ipynb' file.
5. Run the code cells in the notebook sequentially by clicking on each cell and pressing Shift + Enter.
6. You can interact with the code, view visualizations, and see model performance metrics as the code executes.
7. The final model, a Random Forest Classifier, is saved as 'ccfdmodel.pkl' and can be used to classify Normal or Fraudlent transactions.

## About the datasets

The dataset contains transactions made by credit cards in September 2013 by European cardholders.  This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.  It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable, and it takes value 1 in case of fraud and 0 otherwise.

We were utilized these dataset to classify the transactions types for instances Normal or Fraudlent transactions.

Feel free to explore and modify the code to gain a deeper understanding of credit card fraud detection using this rich dataset.


## Understanding the Code
The code consists of the following sections:
- Importing necessary libraries and reading the dataset.
- Data preprocessing, cleaning.
- Exploratory data analysis with visualizations.
- Building and evaluating classification models:
  - Logistic Regression
  - Decision Tree Classifier
  - Light GBM Classifier
  - Random Forest Classifier
- Building and evaluating a classification model for fraudlent transactions.
- Saving the trained Random Forest Classifier model using joblib.

## Data Sources
The dataset used in this code ('creditcard.csv') is assumed to be available in the same directory as the Jupyter Notebook. This dataset contains various features like time, including amounts are used to train and evaluate the models.
