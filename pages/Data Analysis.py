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
from sklearn.metrics import accuracy_score, precision_score ,recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import vegafusion as vf
from Objective import get_data
from streamlit_echarts import st_echarts
from sklearn.utils import resample

#-------------------------------------------------------------------------------------------------------------------------------------
#Setting configurations for the page

st.set_page_config(
    page_title = "Data Analysis",
    page_icon = "📊",
)

Fraud = get_data()
st.write(Fraud.isFraud.sum())



#-------------------------------------------------------------------------------------------------------------------------------------
#Data dicitonary

st.markdown("""
###### The data was hosted on kaggle for analysis 
###### For more information about the source, please visit the [link](https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data) here. 
---
""")


st.subheader("Context",divider="red")
st.markdown("##### The dataframe contains **8** features")
col = Fraud.columns.tolist()
definitions = [
    "Maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).",
    "CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.",
    "Amount of the transaction in local currency.",
    "Customer who started the transaction.",
    "Initial balance before the transaction.",
    "New balance after the transaction.",
    "Customer who is the recipient of the transaction.",
    "Initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).",
    "New balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).",
    "This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.",
    "The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction."
    ]
dict_definitions = {k:v for k,v in zip(col,definitions)}
options = st.multiselect("Select the feature(s) of interest",col,None)
st.dataframe(Fraud.head(),column_order=options,use_container_width=True,hide_index=True)

#(f"{k}:{v}" for k,v in dict_definitions if k in options)   
text = "<br><br>".join(f'<span style="color:#8d99ae">{i[0].title()}</span> - {i[1]}' for i in dict_definitions.items() if i[0] in options)
st.markdown("###### " + text,True)

st.subheader("Data Manipulation",divider="red")
tab1, tab2 = st.tabs(['`step` to `hours`','Add `per_change`'])

with tab1:
    col1,col2 = st.columns(2)
    with col1:
        st.dataframe(Fraud.step.head(),hide_index=True)
        Fraud.rename(columns={"step":"hours"},inplace=True)

    with col2:
        st.dataframe(Fraud.hours.head(),hide_index=True)
    
    with st.expander("Explanation"):
        st.markdown("""
        ###### `step` was changed to `hours` to improve readablility of features.
        """)

with tab2:
    Fraud["per_change"] = ((Fraud["newbalanceOrig"] - Fraud["oldbalanceOrg"])/(Fraud["oldbalanceOrg"])) 
    Fraud["per_change"] = pd.Series(np.where(Fraud["oldbalanceOrg"] == 0.0,0,Fraud["per_change"]))
    st.dataframe(Fraud[["oldbalanceOrg","newbalanceOrig","per_change"]].head(),hide_index=True)
    with st.expander("Explanation"):
        st.markdown("""
        ###### Added `per_change` to have a numeric value that better highlightes the significance of value for each transaction regardless of `oldbalanceOrg`. 
        """)

@st.cache_data
def get_fig():
    plt.figure(figsize=(10, 6))
    plot = sns.scatterplot(data=Fraud, x='amount', y='hours', hue='isFraud', alpha=0.5)
    plt.xlabel('Hours')
    plt.ylabel('Amount')
    plt.title('Scatter Plot of Hours vs. Amount with Fraudulent Status')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title='Is Fraud')
    return plot

@st.cache_data
def get_model(df):
    df = pd.concat([df,pd.get_dummies(df['type'],drop_first=True)],axis=1).copy()
    df.drop("type",axis=1,inplace=True)
    minority_class = df[df["isFraud"] == 1]
    majority_class = df[df["isFraud"] != 1]
    undersampled_majority = resample(majority_class, replace=False,n_samples=len(minority_class))
    undersampled_data = pd.concat([undersampled_majority,minority_class])
    undersampled_data = undersampled_data.sample(frac=1)
    selected_features = ["per_change","CASH_OUT","DEBIT","TRANSFER"]
    X = undersampled_data[selected_features]
    y = undersampled_data["isFraud"]
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2)
    L = LogisticRegression()
    L.fit(X=X_train,y=y_train)
    return (L, X_test, y_test)

L_tuple = get_model(Fraud)        
st.subheader("Data Exploration",divider="red")
tab1, tab2, tab3 ,tab4 ,tab5 = st.tabs(['Fraud vs NonFraud Transactions','`Customer` to `Merchant`, `Debit`, and `Cash_In` are not Fraud','Relationship between `hours` and `amount` for the first 10 hours',"Daily Fraudulent Transactions","Distribution of Transactions"])

with tab1:
    Fraud_Distribution = alt.Chart(Fraud[["isFraud"]]).encode(
        x = "isFraud:N",
        y = alt.Y("count(isFraud):O",axis=alt.Axis(title="Number of instances")),
        color=alt.Color('isFraud:N',scale=alt.Scale(range=["#8d99ae","#ef233c"])),
        text = "count(isFraud)",
    ).properties(
        width = 750,
        height = 600
    )
    # Displaying the graph in streamlit
    Fraud_Distribution = Fraud_Distribution.mark_bar() + Fraud_Distribution.mark_text(dx=2, dy=-10, fontSize=21, fontWeight="bold") 
    st.altair_chart(Fraud_Distribution,use_container_width=True)
    with st.expander("Explanation"):
        st.markdown("""
            ###### The drastic difference between the amount of NonFraudulent to Fraudulent transactions.0 means not fraudulent and 1 means it is fraudulent.
            """)
    
    with tab2:
        col1,col2 = st.columns(2)
        with col1:
            nameDests = Fraud[["nameDest","isFraud"]]
            nameDests["nameDest"] = (nameDests["nameDest"].apply(func=lambda x:"M" if "M" in x else "C"))
            st.write(nameDests.pivot_table(columns='isFraud',values='isFraud',index="nameDest",aggfunc='size'))
        with col2:
            nameDests["type"] = Fraud["type"]
            st.write(nameDests.pivot_table(columns='isFraud',values='isFraud',index="type",aggfunc='size'))
        with st.expander("Explanation"):
            st.markdown("""
                ###### Both of the contingency tables identify values of features that are susceptible to fraud.
                ###### In the left column, it shows that in the dataset transactions between Fraud to Merchants are not susceptible to fraud.
                ###### In the right, it shows that in the dataset that only features that are `CASH_OUT` and `TRANSFER` are susceptible to fraud.
                """)
    with tab3:
        #nameFraud = pd.DataFrame(data=None)
        #nameFraud['Frequency'] = Fraud.nameDest.value_counts()
        #groupbyobj = Fraud.groupby(['nameDest'])
        #badActors= pd.DataFrame(groupbyobj["isFraud"].agg(["count","sum"]).sort_values(by=["sum","count"],ascending=False))
        #badActors = badActors[badActors["count"] == badActors["sum"]]
        #st.text("Under construction")
        #st.write(Fraud["nameDest"].values in set(badActors.index.to_list()))
        #plt.figure(figsize=(10, 6))
        #plot = sns.scatterplot(data=Fraud, x='amount', y='hours', hue='isFraud', alpha=0.5)
        #plt.xlabel('Hours')
        #plt.ylabel('Amount')
        #plt.title('Scatter Plot of Hours vs. Amount with Fraudulent Status')
        #plt.grid(True)
        #plt.tight_layout()
        #plt.legend(title='Is Fraud')
        #st.pyplot(plot.get_figure())
       f = get_fig()
       st.pyplot(f.get_figure())

#    with tab4: 
#        #hours = [F % 24 for F in Fraud.hours.to_list()].sort()
#        typeofTansaction= ["notFraud","isFraud"]
#        timeofFraud = Fraud[["isFraud","hours"]]
#        timeofFraud["hours"] = timeofFraud.hours.apply(func=lambda x: (x%24))
#        fraudTrans = timeofFraud[timeofFraud ["isFraud"] == 1]
#        nfraudTrans = timeofFraud[timeofFraud["isFraud"] != 1]
#        fraudTrans = fraudTrans.hours.value_counts().sort_index()
#        nfraudTrans = nfraudTrans.hours.value_counts().sort_index()
#        
#        nfraudTranspercent = fraudTrans.apply(func=lambda x: x/8213).rename("notFraud")
#        fraudTranspercent = nfraudTrans.apply(func=lambda x: x/6354407).rename("Fraud")
#        test = pd.DataFrame([nfraudTranspercent,fraudTranspercent]).T
#        #st.write(test)
#        hours = test.index.to_list()
#        #st.write(pd.concat([nfraudTranspercent,fraudTranspercent]))
#        tests = pd.concat([nfraudTranspercent,fraudTranspercent]).T
#        #st.write(test.values)
#        #[y,x,z[y]]
#        data = [[x,y,round(z[y] * 10000)/10000] for x,z in zip(range(24),test.values) for y in range(2)] 
#        #st.write(data)
#        #data = [[d[1], d[0], d[2] if d[2] != 0 else "-"] for d in data]
#
#        option = {
#            "tooltip": {"position": "top"},
#            "grid": {"height": "50%", "top": "10%"},
#            "xAxis": {"type": "category", "data": hours, "splitArea": {"show": True}},
#            "yAxis": {"type": "category", "data": typeofTansaction, "splitArea": {"show": True}},
#            "visualMap": {
#                "min": 0,
#                "max": 1,
#                "calculable": True,
#                "orient": "horizontal",
#                "left": "center",
#                "bottom": "15%",
#            },
#            "series": [
#                {
#                    "name": "Fraudulent Transaction",
#                    "type": "heatmap",
#                    "data": data,
#                    "label": {"show": False},
#                    "emphasis": {
#                        "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}
#                    },
#                }
#            ],
#        }
#        st_echarts(option,height="400px")
#        with st.expander("Explanation"):
#            st.markdown("""
#                ###### This heatmap uses the provided hours feature to visualize the distribution of transactions within a 24 hour period
#                ###### It can be seen that a majority of the fraudulent transactions occur between the 10th to 20th hour 
#                """)
#    
#    with tab5:
#        option2 = {
#            "title": [
#                {"text": "Distribution of Fraud and nonFraud transactions", "left": "center"},
#                {
#                    "borderColor": "#999",
#                    "borderWidth": 1,
#                    "textStyle": {"fontWeight": "normal", "fontSize": 14, "lineHeight": 20},
#                    "left": "10%",
#                    "top": "90%",
#                },
#            ],
#            "dataset": [
#                {
#                    "source": [
#                         Fraud[(Fraud["isFraud"] != 1) & (Fraud["per_change"] < 0.5)]["per_change"].to_list() , Fraud[(Fraud["isFraud"] == 1) & (Fraud["per_change"] < 0.5) & (Fraud["per_change"] > -0.99)]["per_change"].to_list()
#                    ],
#                },
#                {
#                    "transform": {
#                        "type": "boxplot",
#                        "config": {"itemNameFormatter": "Fraud {value}"},
#                    }
#                },
#                {"fromDatasetIndex": 1, "fromTransformResult": 1},
#            ],
#            "tooltip": {"trigger": "item", "axisPointer": {"type": "shadow"}},
#            "grid": {"left": "10%", "right": "10%", "bottom": "15%"},
#            "xAxis": {
#                "type": "category",
#                "boundaryGap": True,
#                "nameGap": 30,
#                "splitArea": {"show": False},
#                "splitLine": {"show": False},
#            },
#            "yAxis": {
#                "type": "value",
#                "splitArea": {"show": True},
#            },
#            "series": [
#                {"name": "boxplot", "type": "boxplot", "datasetIndex": 1},
#                {"name": "outlier", "type": "scatter", "datasetIndex": 2},
#            ],
#        }
#        st_echarts(option2, height="500px")
#        with st.expander("Explanation"):
#            st.markdown("""
#                ###### The box plot shows the distribution of values for Fraudulent and NonFraudulent values
#                ###### On the left labeled \"Fraud 0\", shows the distribution of nonFraudulent transactions
#                ###### One the right labled \"Fraud 1\",shows the distribution of Fraudulent transactions*
#                ###### * Note that values of -1 were removed to highlight other fraudulent transactions
#                """)
#
st.subheader("Data Modeling",divider="red")
st.write("Logistic Regression")



L = L_tuple[0]
X_test = L_tuple[1]
y_test = L_tuple[2]
y_pred = L.predict(X_test)
y_pred_proba = L.predict_proba(X_test)[:,1].round(2)
pred_df = pd.DataFrame.from_dict(
    {'y_true': y_test,
     'y_pred': y_pred,
     'probability':y_pred_proba
    }
   )

y_pred = L.predict(X_test)
st.write("Accuracy:", accuracy_score(y_test,y_pred))
st.write("Precision:", precision_score(y_test,y_pred))
st.write("Recall:", recall_score(y_test,y_pred))
st.write("F1 score:", f1_score(y_test,y_pred))

equation = str(round(L.intercept_[0] * 10)/10)

for i in range(L.coef_.size):
    equation += " + " + f"{round(L.coef_[0][i] * 10)/10}"+ "x_{}".format("{"+f"{i + 1}"+"}")

st.latex("\hat{g} = " + equation)









#-------------------------------------------------------------------------------------------------------------------------------------



#
##-------------------------------------------------------------------------------------------------------------------------------------
## Reading the .csv to the script and performing needed computations on the code
#fraud_base = pd.read_csv("/home/jonathan/Documents/Obsidian_Vaults/Cuny Tech Prep/Authentic Classes/Offical_Class/Code/CTP-Team-2-Final-Project/Fraud.csv")
#fraud_base.rename(columns={'step':'hours'},inplace=True)
#Fraud = fraud_base[pd.Series(not bool(re.search('M+',"".join(row))) for row in fraud_base[["nameOrig","nameDest"]].values)].reset_index(drop=True)
#Fraud["per_change"] = ((Fraud["newbalanceOrig"] - Fraud["oldbalanceOrg"])/(Fraud["oldbalanceOrg"])) 
#Fraud["per_change"] = pd.Series(np.where(Fraud["oldbalanceOrg"] == 0.0,0,Fraud["per_change"]))
#Fraud = Fraud[Fraud["per_change"] <= 1.00]
#st.markdown("---")
#
##-------------------------------------------------------------------------------------------------------------------------------------
##<----------Code for chart, will comment out if code is in the way ---------->
##Creating the chart to display the distribution of fraud transactions 
#Fraud_Distribution = alt.Chart(Fraud[["isFraud"]]).encode(
#    x = "isFraud:N",
#    y = "count(isFraud):O",
#    color="isFraud:N",
#    text = "count(isFraud)"
#).properties(
#    title = "Distribution of Fradulent transactions to Non Fraudulent transactions",
#    width = 750,
#    height = 600
#)
## Displaying the graph in streamlit
#Fraud_Distribution = Fraud_Distribution.mark_bar() + Fraud_Distribution.mark_text(dx=2, dy=-10, fontSize=21, fontWeight="bold") 
#st.markdown("# Distribution of fraudulent and nonfraudulent transactions")
#st.altair_chart(Fraud_Distribution,use_container_width=True)
##-------------------------------------------------------------------------------------------------------------------------------------
##Add numbers to the top of the bar graphs
#
##Showing the type of transactions that are fraudulent
#x_domain = Fraud.type.unique()
#
#
#Types_of_Fraud = alt.Chart(Fraud[['type','isFraud']]).properties(
#    width=750,height=600).encode(
#    alt.X("type",scale=alt.Scale(domain=list(x_domain))),
#    alt.Y("count(isFraud)",type="temporal"),
#    color = alt.value("#FF5733"),
#    text = 'sum(isFraud)'
#    ).transform_filter(
#        datum.isFraud == 1
#    )
#Types_of_Fraud = Types_of_Fraud.mark_bar() + Types_of_Fraud.mark_text(dx=2, dy=-10, fontSize=21, fontWeight="bold") 
#st.markdown("# Distribution of fraudulent transactions between 4 different transactions")
#st.altair_chart(Types_of_Fraud,use_container_width=True)
#
##Histogram that represents the time in the day
#twentyfourhours = pd.concat([Fraud["per_change"],Fraud['isFraud'],Fraud["hours"] % 24],axis=1)
#twentyfourhours = twentyfourhours[twentyfourhours["isFraud"] == 1].reset_index()
#dailyFraud = alt.Chart(twentyfourhours).mark_bar().encode(
#        alt.X("hours:Q", bin=alt.Bin(extent=[0,23], step=1)),
#        alt.Y('count()',title="Instances of fraud"),
#    )
#
##<-------------------------------- Expeimental Code --------------------------------------------------------------------->
## Want to add a density curve to show the distribution of percent change transactions
##
##test = alt.Chart(twentyfourhours).transform_density(
##    'per_change',
##    as_ = ["per_change","density"],
##    extent=[-1.10,0.1],
##    x = alt.X()
##)
## Want to highlight and mark on the histogram 
##<-- ? Work in progress https://altair-viz.github.io/gallery/bar_chart_with_single_threshold.html
##rule = alt.Chart().mark_rule().encode(
##        y = alt.Y(datum = 400)
##)
##<-------------------------------- Experimental Code --------------------------------------------------------------------->
#
#
#
##-------------------------------------------------------------------------------------------------------------------------------------
## Logistic Regression
#st.markdown("#")
#st.altair_chart(dailyFraud,use_container_width=True)
#st.markdown("---")
#st.write("Logistic Regression")
#Fraud = pd.concat([Fraud,pd.get_dummies(Fraud['type'],drop_first=True)],axis=1)
#Fraud.drop("type",axis=1,inplace=True)
#selected_features = ["per_change","CASH_OUT","DEBIT","TRANSFER"]
#X = Fraud[selected_features]
#y = Fraud["isFraud"]
#X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#L = LogisticRegression()
#L.fit(X=X_train,y=y_train)
#y_pred = L.predict(X_test)
#y_pred_proba = L.predict_proba(X_test)[:,1].round(2)
#pred_df = pd.DataFrame.from_dict(
#    {'y_true': y_test,
#     'y_pred': y_pred,
#     'probability':y_pred_proba
#    }
#)
#sm = SMOTE(sampling_strategy = .5,k_neighbors = 5, random_state = 100)
#X_resampled, y_resampled = sm.fit_resample(X_train,y_train)
#L_smoted = LogisticRegression()
#L_smoted.fit(X_resampled,y_resampled)
#y_pred = L_smoted.predict(X_test)
#st.write("Accuracy:", accuracy_score(y_test,y_pred))
#st.write("Precision:", precision_score(y_test,y_pred))
#st.write("Recall:", recall_score(y_test,y_pred))
#
#
