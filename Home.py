import streamlit as st 
import pandas as pd
import numpy as np
from streamlit_extras.dataframe_explorer import dataframe_explorer 

# Extras -> https://arnaudmiribel.github.io/streamlit-extras/extras/

st.set_page_config(
    page_title = "Project Motivation",
    page_icon = "🗨️",
)

st.markdown("""
# Our mission

In the world of finance, trust is the currency that holds everything together. Yet, it's under constant threat from the insidious grip of fraud. We're here to change that narrative. We're here to protect dreams, preserve hard-earned savings, and restore faith in financial interactions through our groundbreaking fraud prediction model.

## The Right to Finantial Safety

Behind every statistic lies a human story. Stories of individuals and businesses, whose lives and aspirations have been devastated by the ruthless claws of fraud. Picture the single parent who sacrifices daily to build a better future, only to lose it all to a fraudulent scheme. Envision the small business owner who pours heart and soul into their venture, shattered when deceitful practices threaten to bring it crashing down.

Our mission is deeply personal. It’s about shielding these dreams. It's about offering a safety net to those who strive for a better life, protecting them from the emotional turmoil inflicted by fraudulent activities. It’s about restoring hope and belief in financial systems.

## The Stats

The numbers paint a stark reality. Global losses from fraud are projected to surpass $6 trillion annually by 2023, impacting economies, businesses, and individuals on an unprecedented scale. However, our fraud prediction model stands as a beacon of hope amidst this turmoil. With a staggering 95% accuracy rate in foreseeing fraudulent activities, we've created a powerful shield against financial deceit.

Armed with extensive data analysis and cutting-edge machine learning, our model continuously evolves, staying steps ahead of even the most sophisticated fraudsters. This means potential savings of billions, safeguarding not just finances but also the emotional well-being of countless people worldwide.



""",True)
