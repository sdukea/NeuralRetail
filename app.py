# web app foundations
import streamlit as st

# data
import pandas as pd

st.set_page_config(page_title='NeuralRetail – Welcome', layout='wide')

st.title('NeuralRetail – AI Sales Intelligence')

# data loading

# without caching, every UI interaction, every rerun and every filter change would
# reload the excel file; app becomes slow

# First run:
# Load Excel → store result

# Next runs:
# Reuse stored result

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

# Pandas notes:
# 1. df['columnname'] -> get that specific column
# 2. df[mask], where mask is some boolean Series (conventionally a 1D array) like df['Quantity'] > 0
# -> filters rows (eg: df[df['Quantity'] > 0])
# 3. df[0:5] -> gets rows 1 to 5
# 4. df[0] -> NOTE: does not get rows; instead, finds column named 0. If not, KeyError: 0 is the
# error you'll get
# So, yes, a single index will index the column BY NAME; an index of ':' (slicing) will get you rows
# by INDEX/POSITION (0th row (1st row normally) to 5th row (6th row normally))

# df.loc[row_selector]
# df.loc[row_selector, column_selector]
# Use .loc if you want to get specific rows and row/column combo-ed values by label
# i.e. if row is given label a, b, c, d, or 1, 2, 3, 4, then you will have to do:
# df.loc['a'] or df.loc[1] to get the first row
# and if a column has a specific name, you will have to specify its name/label
# df.loc['a', 'Quantity'], etc.

# df.iloc[row_selector]
# df.iloc[row_selector, column_selector]
# Use .iloc if you want to get specific rows and row/column combo-ed values by position
# So any name given to rows or columns – does not matter
# You will have to index them by position and only then get them

# KPIs

# we need three performance indicators/3 KPIs

col1, col2, col3 = st.columns(3)

# Total revenue
col1.metric("Total revenue", f"{df['TotalPrice'].sum():,.0f}")

# metric -> streamlit UI component; a KPI card

# Total orders
col2.metric('Total orders', df['InvoiceNo'].nunique())

# Total customers
col3.metric('Total customers', df['CustomerID'].nunique())

