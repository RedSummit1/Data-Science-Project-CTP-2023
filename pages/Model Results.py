import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    """
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    This function plots the ROC curve
    """
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

st.title('Logistic Regression Model Performance Dashboard')

# Load the model
with open('/home/jonathan/Code/CTP/streamlit/data/evan_logistic_regression_model.pkl', 'rb') as file:
    lgr = pickle.load(file)

# Load and preprocess the data
data_path = '/home/jonathan/Documents/Obsidian_Vaults/Cuny Tech Prep/Authentic Classes/Offical_Class/Code/CTP-Team-2-Final-Project/Fraud.csv'
data = pd.read_csv(data_path)

# Apply the preprocessing steps as before
# ...

data["per_change"] = ((data["newbalanceOrig"] - data["oldbalanceOrg"])/(data["oldbalanceOrg"]))
data["per_change"] = pd.Series(np.where(data["oldbalanceOrg"] == 0.0,0,data["per_change"]))

data = data.drop(columns=['isFlaggedFraud', 'oldbalanceOrg', 'oldbalanceDest','newbalanceOrig', 'newbalanceDest', 'amount'])
data = pd.get_dummies(data, columns=['type'], prefix=['type'])
data = data.drop(columns=['nameOrig', 'nameDest', 'step', 'type_PAYMENT', 'type_CASH_IN'])

data = data.sample(n=600000, random_state=0)

data.head(5)

# Ensure all columns are of type float32
X_test = data.drop('isFraud', axis=1).astype('float32')
y_test = data['isFraud']

st.write("Data Snapshot:")
st.write(data.head())

# Make predictions
predictions = lgr.predict(X_test)
probabilities = lgr.predict_proba(X_test)[:, 1]

# Generate the confusion matrix
lgr_cm = confusion_matrix(y_test, predictions)

# Calculate ROC curve and AUC
lgr_fpr, lgr_tpr, _ = roc_curve(y_test, probabilities)
lgr_roc_auc = auc(lgr_fpr, lgr_tpr)

# Plotting Confusion Matrix
st.write("Confusion Matrix")
plot_confusion_matrix(lgr_cm, classes=np.array([0, 1]))
st.pyplot()

# Plotting ROC Curve
st.write("ROC Curve")
plot_roc_curve(lgr_fpr, lgr_tpr, lgr_roc_auc)
st.pyplot()