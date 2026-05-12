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
    df = pd.read_excel("data/raw/online_retail.xlsx")
    # loads data from excel into Pandas DF; an in-built spreadsheet inside of pandas with rows/columns
    # you can manipulate with python

    # previews
    first2 = df.head(5)

    last2 = df.tail(5)

    print(f"Preview:\n{first2}\n...\n{last2}")

    columns_list = df.columns.to_list()

    print(f"Columns: {columns_list}")


    