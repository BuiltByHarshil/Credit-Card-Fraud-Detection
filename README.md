# Credit Card Fraud Detection (Machine Learning)

This project builds a Credit Card Fraud Detection system using Machine Learning in Python. It classifies transactions as legitimate or fraudulent using historical transaction data.

## Tech Stack
- Python
- NumPy
- Pandas
- Scikit-learn

## Approach
- Load and explore the dataset
- Handle class imbalance using undersampling
- Split data into training and testing sets
- Train a Logistic Regression model
- Evaluate the model using accuracy

## Model
- Logistic Regression (binary classification)

## How to Run
1. Install dependencies:
   pip install numpy pandas scikit-learn

2. Update dataset path in the code:
   pd.read_csv('path/to/creditcard.csv')

3. Run the script:
   python your_script_name.py

## Dataset Info
- 0 = Legitimate transaction
- 1 = Fraudulent transaction

## Note
The dataset is highly imbalanced, so undersampling is used to balance the classes.

