# web app foundations
import streamlit as st

# data
import pandas as pd

st.set_page_config(page_title='NeuralRetail', layout='wide')

st.title('NeuralRetail – AI sales intelligence')

# data loading

# without caching, every UI interaction, every rerun and every filter change would
# reload the excel file

# caching stores processed dataframe in memory

@st.cache_data
def load_data():
    pd.read_excel("data/raw/online_retail.xlsx")