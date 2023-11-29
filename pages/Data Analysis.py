import streamlit as st
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from altair import datum
from vega_datasets import data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import vegafusion as vf

#-------------------------------------------------------------------------------------------------------------------------------------
#Setting configurations for the page
st.set_page_config(
    page_title = "Data Analysis",
    page_icon = "📊",
)
#-------------------------------------------------------------------------------------------------------------------------------------

#Data dicitonary
st.markdown("""
    <h1>Context</h1>
    <p> In developing a model for predicting fraudulent transactions for a financial company and use insights from the model to develop an actionable plan. The data that we are working in a large CSV format and has 6362620 rows and 10 columns.</p>
    <h1>Content</h1>
    <p>Data Dictionary</p>
    <p><strong>step</strong> - maps a unit of time in the real world. In this case 1 step is 1 hour of time.Total steps 744 (30 days simulation).</p>
    <p><strong>type</strong> - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
    <p><strong>amount</strong> - amount of the transaction in local currency.</p>
    <p><strong>nameOrig</strong> - customer who started the transaction</p>
    <p><strong>oldbalanceOrg</strong> - initial balance before the transaction</p>
    <p><strong>newbalanceOrig</strong> - new balance after the transaction</p>
    <p><strong>nameDest</strong> - customer who is the recipient of the transaction</p>
    <p><strong>oldbalanceDest</strong> - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).</p>
    <p><strong>newbalanceDest</strong> - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).</p>
    <p><strong>isFraud</strong> - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.</p>
    <p><strong>isFlaggedFraud</strong> - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.</p>
\n""",True)
#-------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------
# Reading the .csv to the script and performing needed computations on the code
fraud_base = pd.read_csv("/home/jonathan/Documents/Obsidian_Vaults/Cuny Tech Prep/Authentic Classes/Offical_Class/Code/CTP-Team-2-Final-Project/Fraud.csv")
fraud_base.rename(columns={'step':'hours'},inplace=True)
Customers = fraud_base[pd.Series(not bool(re.search('M+',"".join(row))) for row in fraud_base[["nameOrig","nameDest"]].values)].reset_index(drop=True)
Customers["per_change"] = ((Customers["newbalanceOrig"] - Customers["oldbalanceOrg"])/(Customers["oldbalanceOrg"])) 
Customers["per_change"] = pd.Series(np.where(Customers["oldbalanceOrg"] == 0.0,0,Customers["per_change"]))
Customers = Customers[Customers["per_change"] <= 1.00]
st.markdown("---")

#-------------------------------------------------------------------------------------------------------------------------------------
#<----------Code for chart, will comment out if code is in the way ---------->
#Creating the chart to display the distribution of fraud transactions 
Fraud_Distribution = alt.Chart(Customers[["isFraud"]]).encode(
    x = "isFraud:N",
    y = "count(isFraud):O",
    color="isFraud:N",
    text = "count(isFraud)"
).properties(
    title = "Distribution of Fradulent transactions to Non Fraudulent transactions",
    width = 750,
    height = 600
)
# Displaying the graph in streamlit
Fraud_Distribution = Fraud_Distribution.mark_bar() + Fraud_Distribution.mark_text(dx=2, dy=-10, fontSize=21, fontWeight="bold") 
st.markdown("# Distribution of fraudulent and nonfraudulent transactions")
st.altair_chart(Fraud_Distribution,use_container_width=True)
#-------------------------------------------------------------------------------------------------------------------------------------
#Add numbers to the top of the bar graphs

#Showing the type of transactions that are fraudulent
x_domain = Customers.type.unique()


Types_of_Fraud = alt.Chart(Customers[['type','isFraud']]).properties(
    width=750,height=600).encode(
    alt.X("type",scale=alt.Scale(domain=list(x_domain))),
    alt.Y("count(isFraud)",type="temporal"),
    color = alt.value("#FF5733"),
    text = 'sum(isFraud)'
    ).transform_filter(
        datum.isFraud == 1
    )
Types_of_Fraud = Types_of_Fraud.mark_bar() + Types_of_Fraud.mark_text(dx=2, dy=-10, fontSize=21, fontWeight="bold") 
st.markdown("# Distribution of fraudulent transactions between 4 different transactions")
st.altair_chart(Types_of_Fraud,use_container_width=True)

#Histogram that represents the time in the day
twentyfourhours = pd.concat([Customers["per_change"],Customers['isFraud'],Customers["hours"] % 24],axis=1)
twentyfourhours = twentyfourhours[twentyfourhours["isFraud"] == 1].reset_index()
dailyFraud = alt.Chart(twentyfourhours).mark_bar().encode(
        alt.X("hours:Q", bin=alt.Bin(extent=[0,23], step=1)),
        alt.Y('count()',title="Instances of fraud"),
    )


#<-------------------------------- Expeimental Code --------------------------------------------------------------------->
# Want to add a density curve to show the distribution of percent change transactions
#
#test = alt.Chart(twentyfourhours).transform_density(
#    'per_change',
#    as_ = ["per_change","density"],
#    extent=[-1.10,0.1],
#    x = alt.X()
#)
# Want to highlight and mark on the histogram 
#<-- ? Work in progress https://altair-viz.github.io/gallery/bar_chart_with_single_threshold.html
#rule = alt.Chart().mark_rule().encode(
#        y = alt.Y(datum = 400)
#)
#<-------------------------------- Experimental Code --------------------------------------------------------------------->



#-------------------------------------------------------------------------------------------------------------------------------------
# Logistic Regression
st.markdown("#")
st.altair_chart(dailyFraud,use_container_width=True)
st.markdown("---")
st.write("Logistic Regression")
Customers = pd.concat([Customers,pd.get_dummies(Customers['type'],drop_first=True)],axis=1)
Customers.drop("type",axis=1,inplace=True)
selected_features = ["per_change","CASH_OUT","DEBIT","TRANSFER"]
X = Customers[selected_features]
y = Customers["isFraud"]
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
L = LogisticRegression()
L.fit(X=X_train,y=y_train)
y_pred = L.predict(X_test)
y_pred_proba = L.predict_proba(X_test)[:,1].round(2)
pred_df = pd.DataFrame.from_dict(
    {'y_true': y_test,
     'y_pred': y_pred,
     'probability':y_pred_proba
    }
)
sm = SMOTE(sampling_strategy = .5,k_neighbors = 5, random_state = 100)
X_resampled, y_resampled = sm.fit_resample(X_train,y_train)
L_smoted = LogisticRegression()
L_smoted.fit(X_resampled,y_resampled)
y_pred = L_smoted.predict(X_test)
st.write("Accuracy:", accuracy_score(y_test,y_pred))
st.write("Precision:", precision_score(y_test,y_pred))
st.write("Recall:", recall_score(y_test,y_pred))

