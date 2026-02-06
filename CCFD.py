import numpy as py 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Importing dataset 
credit_card_data = pd.read_csv('C:/Users/harsh/Desktop/-/Projects/Credit Card Fraud Detection/data/creditcard.csv')

# To see the amount of fradulent transactions and real legit transactions
print(credit_card_data['Class'].value_counts())
 # 0 = legit transaction ; 1 = fradulent transaction

# Separating dataset with two variables that is legit and fraud
legit = credit_card_data[credit_card_data.Class == 0 ]
fraud = credit_card_data[credit_card_data.Class == 1 ]

# Statistical values for the dataset
print (legit.Amount.describe())
print (fraud.Amount.describe())

# compare the stats between legit and fraud transactions 
print (credit_card_data.groupby('Class').mean())

## To solve the problem of class imbalance(one class having way more data than the other) we will use the method of UNDERSAMPLING(throw away extra majority samples to balance classes)

# Building a sample dataset having similar distributions of legit and fradulent transactions 

legit_sample = legit.sample(n=492)

# Concatinating both datasets (legit_sample and fraud)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# compare the stats between legit and fraud transactions in new dataset
print (new_dataset.groupby('Class').mean()) 

# Splitting data into features and targets 
X = new_dataset.drop(columns='Class' , axis = 1)
Y = new_dataset['Class']

# Splitting Data in training and testing data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, stratify=Y, random_state=3)

# Model training (Logistic Regression for binary values) 
model = LogisticRegression(max_iter=1000)

# Training Logistic Regression Model 
model.fit(X_train, Y_train) 

# Model Evaluation 
## ACCURACY SCORE

# on training data 
X_train_predict = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predict, Y_train)
print('Training Data Accuracy = ' , training_data_accuracy ) 

# on training data 
X_test_predict = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predict, Y_test)
print('Test Data Accuracy = ', test_data_accuracy)