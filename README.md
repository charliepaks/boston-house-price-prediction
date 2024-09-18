
# Packaging the ML Model of Classification

#### Predicting House Prices in Boston: A Machine Learning Exploration
This project outlines a guided approach to predicting house prices in Boston, Massachusetts, using machine learning techniques. This project leverages the well-established Boston Housing Dataset, a valuable resource for introductory machine learning exercises.

#### Project Methodology:
1. Data Acquisition and Exploration:
Import the Boston Housing Dataset. Conduct a thorough exploration of the data, including:

Identifying and understanding the features (independent variables) that might influence house prices (dependent variable).
Analyzing data distribution, identifying potential outliers or missing values.
Visualizing relationships between features and the target variable using techniques like scatter plots and heatmaps.
Context The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.

#### Attribute Information Input features in order:

CRIM: per capita crime rate by town
ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS: proportion of non-retail business acres per town
CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX: nitric oxides concentration (parts per 10 million) [parts/10M]
RM: average number of rooms per dwelling
AGE: proportion of owner-occupied units built prior to 1940
DIS: weighted distances to five Boston employment centres
RAD: index of accessibility to radial highways
TAX: full-value property-tax rate per  10,000[ /10k]
PTRATIO: pupil-teacher ratio by town
B: The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT: % lower status of the population
Output variable:

MEDV: Median value of owner-occupied homes in  1000′𝑠[𝑘 ]
Source StatLib - Carnegie Mellon University

Relevant Papers Harrison, David & Rubinfeld, Daniel. (1978). Hedonic housing prices and the demand for clean air. Journal of Environmental Economics and Management. 5. 81-102. 10.1016/0095-0696(78)90006-2. LINK

Belsley, David A. & Kuh, Edwin. & Welsch, Roy E. (1980). Regression diagnostics: identifying influential data and sources of collinearity. New York: Wiley LINK

#### 2. Data Preprocessing:
Address missing values through techniques like imputation or removal (based on data specifics).
Employ feature scaling or normalization to ensure all features are on a similar scale and contribute equally to model training.
Model Selection and Training:
Choose an appropriate machine learning model for predicting house prices. This could involve:

Considering linear regression for its interpretability when dealing with a continuous target variable like price.
Exploring non-linear models like decision trees or random forests for potentially complex relationships between features and prices.
Split the data into training and testing sets. The training set will be used to train the model, and the testing set will be used to evaluate its performance on unseen data.
Train the chosen model on the training data, allowing it to learn the underlying patterns and relationships within the data.
Model Evaluation:
Evaluate the model's performance on the testing set using metrics like mean squared error or R-squared. These metrics provide insights into how accurately the model predicts house prices for unseen data. Consider visualization techniques like error histograms to gain a deeper understanding of the prediction errors and potential areas for improvement.

#### Hyperparameter Tuning (Optional):
If necessary, we will explore hyperparameter tuning to optimize the chosen model's performance. This can involve adjusting model parameters to potentially improve its generalizability and accuracy.

#### Model Interpretation (if using interpretable models):
If a model with interpretable features like linear regression is chosen, analyze the learned coefficients to understand the relative impact of each feature on house price predictions.

Source: Kaggle



## Virtual Environment
Install virtualenv

```python
python3 -m pip install virtualenv
```

Check version
```python
virtualenv --version
```

Create virtual environment

```python
virtualenv ml_package
```

Activate virtual environment

For Linux/Mac
```python
source ml_package/bin/activate
```
For Windows
```python
ml_package\Scripts\activate
```

Deactivate virtual environment

```python
deactivate
```


## Directory structure

```bash
prediction_model


├── MANIFEST.in
├── prediction_model
│   ├── config
│   │   ├── config.py
│   │   └── __init__.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── test.csv
│   │   └── train.csv
│   ├── __init__.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── processing
│   │   ├── data_handling.py
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── trained_models
│   │   ├── classification.pkl
│   │   └── __init__.py
│   ├── training_pipeline.py
│   └── VERSION
├── README.md
├── requirements.txt
├── setup.py
└── tests
    ├── pytest.ini
    └── test_prediction.py
```


# Build the Package

1. Goto Project directory and install dependencies
`pip install -r requirements.txt`

2. Create Pickle file after training:
`python prediction_model/training_pipeline.py`

3. Create source distribution and wheel
`python setup.py sdist bdist_wheel`

# Installation of Package

Go to project directory where `setup.py` file is located

1. To install it in editable or developer mode
```python
pip install -e .
```
```.``` refers to current directory

```-e``` refers to --editable mode

2. Normal installation
```python
pip install .
```
```.``` refers to current directory

3. Also can be installed from git as well after pushing to github

```
pip install git+https://github.com/charliepaks/boston-house-price-prediction.git
```

# Testing the Package Working

1. Remove the PYTHONPATH from environment variables 
2. Goto a separate location which is outside of package directory
3. Create a new virual environment using the commands mentioned above & activate it
4. Before installing, test whether you are able to import the package of `prediction_model` - (you should not be able to do it)
5. Now in the new environment install the package using the generated file
`pip install git+https://github.com/charliepaks/boston-house-price-prediction.git`
6. Now try importing the prediction_model, you should be able to do it successfully
7. Extras : Run training pipeline using the package, and also conduct the test


