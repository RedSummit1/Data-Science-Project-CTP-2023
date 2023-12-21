#from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from Objective import get_data
#import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

st.title("Model Comparison and Neural Network Construction")



st.write('''Our initial approach to selecting a model for detecting fraudulent credit card transactions
was to train a logistic regression model. However, after training and testing the model, we
noticed that our scores didn’t perform as well as we hoped, even after making adjustments to our
dataset.''')

tab1, tab2  = st.tabs(["Logistic Regression Confusion Matrix","Logistic Regression ROC Curve"])

with tab1:
    st.image("data/LogisticRegressionConfu.png")    
with tab2:
    st.image("data/LogRegROC.png")    

st.write(""" 
From here, we decided to run a model comparison between various classification models. We
were able to do this using a library called Pycaret.
""")

st.image("/home/jonathan/Documents/Obsidian_Vaults/Cuny Tech Prep/Authentic Classes/Offical_Class/Code/streamlit/data/Pycaretimage.png")


st.write("""
After viewing the results, we identified random forest or decision tree classifiers as
potentially better options, in line with findings from other Kaggle notebooks.
""")

tab1, tab2  = st.tabs(["Random Forest Classifier Confusion Matrix","Random Forest Classifier ROC Curve"])

with tab1:
    st.image("data/Random Forest Classifier Confusion Matrix.png")

with tab2: 
    st.image("data/Random Forest Classifier ROC Curve.png")
    

st.write("""We wanted to explore another approach to the problem and since we had a very large
dataset with 6 million+ samples, we decided to build a neural network to see if we could match
the Random Forest Classifier or even beat it in accuracy by some amount.
For our neural network, we tested out various architectures ranging from small to large
sizes. Given the size of the dataset and the fact that we included ten features for our input, we
decided that a somewhat complex architecture would be needed. The final version consists of an
input layer with 10 nodes, 4 hidden layers (nodes: 20 150 20 3), one dropout layer after the
biggest hidden layer, and an output layer with one node.""")

st.image("data/Neural_Network_Arcitecture.png")
tab1, tab2  = st.tabs(["Neural Network Confusion Matrix","Neural Network ROC Curve"])

st.write("""
         We experimented with architectures of varying sizes and settled on a complex design
with 10 input nodes, 4 hidden layers (20-150-20-3 nodes), a dropout layer, and a sigmoid-output
layer. To accommodate a new "per change" column with negative and positive values, we
implemented the Leaky ReLU activation function for the hidden layer nodes.
To manage the computational challenges posed by the large dataset, we down-sampled to
600 thousand rows for more efficient experimentation. During training, we used a learning rate
of 0.01, a batch size of 64, and set epochs to 200. Learning rate reduction and early stopping
were implemented to enhance adaptability and mitigate overfitting.
Despite not training on the entire dataset, our neural network exhibited promising results
on the testing set with 97.622% accuracy and 96.731% precision. Although it trailed behind the
Random Forest Classifier, we anticipate that training on the full dataset could further improve
accuracy. Additionally, future hyperparameter tuning may optimize the neural network's
performance.
""")

with tab1:
    st.image("data/Neural_Network_Confusion_Matrix.png")

with tab2: 
    st.image("data/Neural_Network_ROC_Curve.png")
    

    
# Load the model
#st.markdown("---")
#st.write("Hello World")
#df = get_data()
#st.write(df.head())
#st.write(tf.__version__)



#loaded_model = load_model('/home/jonathan/Documents/Obsidian_Vaults/Cuny Tech Prep/Authentic Classes/Offical_Class/Code/streamlit/data/final.keras')
#loaded_scaler = pickle.load(open('/home/jonathan/Documents/Obsidian_Vaults/Cuny Tech Prep/Authentic Classes/Offical_Class/Code/streamlit/data/finalScaler.pkl','rb'))



#df = pd.read_csv("/home/jonathan/Documents/Obsidian_Vaults/Cuny Tech Prep/Authentic Classes/Offical_Class/Code/streamlit/data/baby_fraud.csv")
#df["per_change"] = ((df["newbalanceOrig"] - df["oldbalanceOrg"])/(df["oldbalanceOrg"]))
#df["per_change"] = pd.Series(np.where(df["oldbalanceOrg"] == 0.0,0,df["per_change"]))
#df.drop(columns=['isFlaggedFraud','oldbalanceOrg', 'oldbalanceDest','nameOrig','nameDest'], axis=1, inplace=True)
#df = pd.get_dummies(df, columns=['type'], prefix=['type'])
#df = df.sample(n=600000, random_state=42)
#X = df.drop('isFraud', axis=1)
#y = df['isFraud']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)




#st.write(loaded_model.summary())
#X_test = loaded_scaler.transform(X_test)
#y_pred = loaded_model.predict(X_test)
#y_pred = (y_pred > 0.5).astype("int32")
#y_test = y_test.astype("int32")
#st.write(y_test.shape)
#st.write(y_pred.shape)
#
#st.write(y_test.shape)
#st.write(y_pred.reshape[-1,:])
#
#st.write(y_test.values.dtype)
#st.write(y_pred.dtype)
#st.write(accuracy_score(y_test.values,y_pred))
#st.write("Done")
#st.write(type(y_test.values))
#st.write(type(y_pred))
#
#cm_normalized = confusion_matrix(np.array(y_test[:,0]), y_pred, normalize='true')
#
## Plot the confusion matrix
#plt.figure(figsize=(10, 7))
#sns.heatmap(cm_normalized, annot=True, fmt='g', cmap='coolwarm')  # 'g' format makes sure integers are displayed properly
#plt.xlabel('Predicted')
#plt.ylabel('True')
#plt.show()
#
#
#
#
#        
## Use original_df as df
#X = df.drop('isFraud', axis=1)
#y = df['isFraud']
#
#
#smote = SMOTE(sampling_strategy='auto', random_state=42)
#X,y = smote.fit_resample(X,y)
#
#X = loaded_scaler.transform(X)
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#
#
#final = loaded_model.predict(X_test)
#
#
#
#accuracy = accuracy_score(y_test, y_pred)
#precision = precision_score(y_test, y_pred)
#recall = recall_score(y_test, y_pred)
#f1 = f1_score(y_test, y_pred)
#
#
#print(f"Accuracy: {accuracy}")
#print(f"Precision: {precision}")
#print(f"Recall: {recall}")
#print(f"F1 Score: {f1}")
#
#
#cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
#
## Plot the confusion matrix
#plt.figure(figsize=(10, 7))
#sns.heatmap(cm_normalized, annot=True, fmt='g', cmap='coolwarm')  # 'g' format makes sure integers are displayed properly
#plt.xlabel('Predicted')
#plt.ylabel('True')
#plt.show()
#
#
#
#import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve, auc
#
## Assuming ytest and ypred_test_lgr are already defined and are the true labels and model's probabilities
## Normally, ypred_test_lgr should be the probability estimates of the positive class, not the predicted labels.
## If ypred_test_lgr is indeed the predicted labels, you need to modify the model output to be probabilities.
#
## Calculate the ROC curve and AUC
#fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#roc_auc = auc(fpr, tpr)
#
## Plot the ROC curve
#plt.figure(figsize=(10, 7))
#plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) Curve')
#plt.legend(loc="lower right")
#plt.show()
#
