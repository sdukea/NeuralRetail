# web app foundations
import streamlit as st

# data
import pandas as pd

st.set_page_config(page_title='NeuralRetail', layout='wide')

st.title('NeuralRetail – AI sales intelligence')

# data loading

# without caching, every UI interaction, every rerun and every filter change would
# reload the excel file; app becomes slow
"""
First run:
Load Excel → store result

Next runs:
Reuse stored result
"""
# caching stores processed dataframe in memory

@st.cache_data
def load_data():
    df = pd.read_excel("data/raw/online_retail.xlsx")
    # loads data from excel into Pandas DF; an in-built spreadsheet inside of pandas with rows/columns
    # you can manipulate with python

    # previews
    first2 = df.head(5)
    last2 = df.tail(5)
    columns_list = df.columns.to_list()
    dtypes = df.dtypes.tolist()

    print(df.info())

    # data cleaning

    # remove missing customers
    df.dropna(subset=['CustomerID'])

    # mask; quantity
    true_rows_q = df['Quantity'] > 0    
    df = df.loc[true_rows_q]

    # mask; unit price 
    true_rows_p = df['UnitPrice'] > 0
    df = df.loc[true_rows_p]

    # convert dtype
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # engineer feature
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    return df, first2, last2, columns_list, dtypes

df, f2, l2, cols, dtypes = load_data()

# sidebar; filters

st.sidebar.header('Filters')

country = st.sidebar.multiselect(
    "Select country", df['Country'].unique(),
    default=df['Country'].unique()
)

# df['Country'].unique() -> chooses all countries without repetition; unique
# default=... -> sets the default multi-selected options as all of these unique countries

# KPIs