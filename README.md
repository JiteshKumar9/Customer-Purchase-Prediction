# Customer-Purchase-Prediction

# Customer Purchase Prediction

## Project Overview

This project is aimed at predicting whether a customer will make a purchase based on their demographic and review data. The dataset contains customer information such as age, gender, education level, and review ratings. Using this data, a machine learning model is built to predict if a customer will make a purchase (`Yes` or `No`).

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dataset Description

The dataset contains the following columns:

- **Customer ID**: Unique identifier for each customer.
- **Age**: Age of the customer.
- **Gender**: Gender of the customer (Male/Female).
- **Education**: Education level (School/UG/PG).
- **Review**: Customer review rating (Poor/Average/Good).
- **Purchased**: Whether the customer made a purchase (Yes/No).

### Sample Data

| Customer ID | Age | Gender | Education | Review  | Purchased |
|-------------|-----|--------|-----------|---------|-----------|
| 1021        | 30  | Female | School    | Average | No        |
| 1022        | 68  | Female | UG        | Poor    | No        |
| 1023        | 70  | Female | PG        | Good    | No        |
| 1024        | 72  | Female | PG        | Good    | No        |
| 1025        | 16  | Female | UG        | Average | No        |

## Data Preprocessing

- **Missing Data**: No missing values were found in the dataset.
- **Categorical Encoding**: 
  - `Review`: Encoded as `Poor` = 0, `Average` = 1, `Good` = 2.
  - `Education`: Encoded as `School` = 0, `UG` = 1, `PG` = 2.
  - `Gender`: Encoded as `Male` = 0, `Female` = 1.
  
The target variable `Purchased` is binary (Yes/No), where `Yes` = 1 and `No` = 0.

## Model Training

1. **Train-Test Split**: The data was split into training (70%) and testing (30%) sets using `train_test_split`.
2. **Model**: A **Random Forest Classifier** was used to build the model.
3. **Training**: The model was trained on the training data and then evaluated on the test data.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = customer.drop(['Purchased', 'Customer ID'], axis=1)
y = customer['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)

model = RandomForestClassifier()
model.fit(X_train, y_train)

**Evaluation**

The model's performance was evaluated using the following metrics:
Accuracy: 73.33%
Confusion Matrix:

```lua

[[7, 0],
 [5, 3]]

**Classification Report:**
  ```markdown

            precision    recall  f1-score   support

        No       0.64      1.00      0.78         7
       Yes       1.00      0.50      0.67         8

  accuracy                           0.73        15
 macro avg       0.82      0.75      0.72        15
weighted avg 0.83 0.73 0.72 15




## Installation

To run this project locally, follow these steps:

1. Clone the repository:
 ```bash
 git clone https://github.com/your-username/customer-purchase-prediction.git
Navigate to the project directory:

```bash
cd customer-purchase-prediction

Install the required dependencies:
```bash
pip install -r requirements.txt


**Usage**
Once the dependencies are installed, you can run the model training and evaluation script with the following command:

```bash
python customer_purchase_prediction.py
This will load the dataset, preprocess it, train the Random Forest model, and display the evaluation metrics.

**Contributing**
If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. Please make sure to follow the coding standards and write tests for any new functionality you add.
