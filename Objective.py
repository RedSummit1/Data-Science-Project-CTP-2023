import streamlit as st 
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts


# Extras -> https://arnaudmiribel.github.io/streamlit-extras/extras/


if __name__ == "__main__":
    st.set_page_config(
        page_title = "Project Motivation",
        page_icon = "🗨️",
    )

    st.markdown("""
            # Credit Card Fraud prevalence increasing in the United States
       <br>
        <h5> According to the Nilson Report, global losses due to credit card fraud has exceeded <span style="color:red"> $35 billion </span>in 2020.

        ##### Currently, <span style="background-color:yellow">Card-Not-Present </span> (CNP) fraud has become more common due to growing number of online transactions. 

        """,True)

    col1,col2 = st.columns(2)
    with col1:
        st.header("Total Fraud Loss")
        st.image("data/Total_Fraud_Loss.png")

    with col2:
        st.header("Change in Fraud ")
        st.image("data/Total_Fraud_Loss_Change.png")

    st.markdown("""
        <br>
        <h5>It is projected that CNP fraud will make up to <span style="color:red">74%</span> of the total amout of fraud by 2024, 17% more that 2019.</h5> 
        """,True)

    st.markdown("""
        ## Building Smarter Fraud Detection Software with Machine Learning 
        ##### Banks and other financial instititions are leveraging Machine Learning to combat CNP fraud. 
        ##### Machine Learning models can adapt to dynamic transaction conditions and detect complex patterns in real-time. 
        <h5>We have used Machine Learning to develop a model that is around <span style="color:green">90% </span> more reliable that currently deployed models.</h5>
        <br>
        """,True)

    comparison_chart = {
        "legend": {},
        "tooltip": {},
        "dataset": {
            "source": [
                ['Models', 'Accuracy', 'Precision','Recall','F1'],
                ['Our Model',0.9267, 0.877, 0.9927, 0.9313],
                ['Their Model', 0.9987, 1.0, 0.0019, 0.00389],
            ]
        },
      "xAxis": { "type": 'category' },
      "yAxis": {},
      "series": [{ "type": 'bar',"color":"#2b2d42" }, { "type":'bar',"color":"#8d99ae"}, { "type": 'bar',"color":"#edf2f4"},{"type":'bar',"color":"#ef233c"}]
    }

    st_echarts(options=comparison_chart,height="600px")


#------------------------- Cache data here for the rest of the presentation -------------------------------------#

@st.cache_data
def get_data():
    df = pd.read_csv("data/Fraud.csv")
    return df

get_data()



    
