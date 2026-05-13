import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

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

# sales trends

st.subheader('Sales trends')

# what groupy() does:
# - splits data into groups
# - apply operations into each group
# - combine results
# So, split → apply → combine

# | Country | Revenue |
# | ------- | ------- |
# | India   | 100     |
# | India   | 200     |
# | USA     | 300     |
# | USA     | 150     |     (eg. data)

# grouped = df.groupby('Country')

# You'll just get a DataFrameGroupBy object like
# India → [0, 1]
# USA   → [1, 2]
# you can see groups by doing grouped.groups
# No values yet; just the row indexes/row positions that stay true to each group
# actual: <pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001F4A92B7FD0>
# two groups; 2 unique 'Country' column/group
# NOTE: df is a DataFrame; grouped is not a DataFrame but a GroupByDataFrame
# its not actual copied tables; just mapping as you see above
# if you print grouped.groups, you'll see something like:
# {
#     'India': [0, 1],
#     'USA': [1, 2]
# }
# so you store group name (uniqe from 'Country') and row indices/rows
# |
# if you did:
# for name, group in grouped:
#   print(name)
#   print(grouped)

# this is the output:
# India
#   Country  Revenue
# 0   India      100
# 1   India      200

# USA
#   Country  Revenue
# 2     USA      300
# 3     USA      150

# So a GroupByDataFrame (grouped) = multiple smaller dataframes grouped together
# these two DataFrames entirely are what 'grouped', the GroupByDataFrame stores
# So,
# India (-> rows) -> dataframe (consisting of instances of group (rows) and all columns/values for each) 
# USA   (-> rows) -> dataframe (consisting of instances of group (rows) and all columns/values for each)

# this is the first step of mental model; split

# Now, you have to apply an operation
# rev_grouped = df.groupby('Country')['Revenue']

# Now, this creates a GroupBySeries now.
# why: because 'grouped' = df.groupby('Country')' is a GroupByDataFrame consisting of multiple data
# frames of groups (each dataframe making up GBDF has multiple rows/instances as you know it)
# and when you index by 'Revenue' column alone, this is what happens:
# India -> rows -> only 'Revenue' column values
# USA -> rows -> only 'Reveneue' column values

# So, from

# India
#   Country  Revenue
# 0   India      100
# 1   India      200

# USA
#   Country  Revenue
# 2     USA      300
# 3     USA      150

# it turns into:
# for name, group in revenue_grouped:
#   print(name)
#   print(group)
# |
# India
# 0    100
# 1    200

# USA
# 2    300
# 3    150

# And so a GroupBySeries consists of multiple Series inside it
# so, this is what happens when you go on to do:
# df.groupby('Country')['Revenue']

# NOTE: df.groupby('Country') -> each group in 'grouped' had DataFrame (grouped = GroupByDataFrame)
# Now, df.groupby('Country')['Revenue'] -> each group in 'rev_grouped' has Series
# (rev_grouped = GroupBySeries)

# now, you can SUM/APPLY OPERATION on a GroupBySeries just like a Series (applies op to every Series
# inside GBS)
# So, .sum() sums values for each Series in GroupBySeries

# rev_grouped_sum = df.groupby('Country')['Revenue'].sum()

# This turns it into a normal Pandas Series now
# why: because a GroupBySeries, which consisted of Series per each group in it (1D array with index and 
# multiple values 'TotalPrice' in it)
# now only consists of ONE VALUE per each group
# so, it isn't a Series anymore; its been converted into a single value by summing all Series 'TotalPrice'
# values.
# |
# So, a GroupBySeries, which should consists of multiple Series (for each group) now only consists
# of values; so a GroupBySeries does not tick anymore
# hence, the only Pandas datatype that holds values (0D/summed values for each group/Series) in it is a 
# normal Series itself, with index 
# |
# So, rev_grouped_sum is now a Series where groups are indices and summed values of each group Series
# (from rev_grouped GroupBySeries) are associated to each of this group index in this normal Series now

# So, DataFrame (df) -> GroupByDataFrame (df.grouped('Country')) -> GroupBySeries 
# (df.grouped('Country')['Revenue']) -> Series (df.grouped('Country')['Revenue'].sum())

daily_sales = df.groupby('InvoiceDate')['TotalPrice'].sum()

# NOTE: daily_sales here is a Series; an index column and values alongside it
# like a 1D array but with indices, that's it.

fig1 = plt.figure()
daily_sales.plot()
# plot the Series daily_sales this way, using the inner Matplotlib functionality in Pandas
# the index in Series – x-axis
# the values in Series – y-axis

plt.xlabel('Date')
plt.ylabel('Revenue')
st.pyplot(fig1)

# top products

st.subheader('Top Products')

top_products = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
fig2 = plt.figure()
top_products.plot(kind='bar')
st.pyplot(fig2)

# customer segmentation

# specifically: RFM customer segmentation
# R - recency, F - frequency, M - monetary
# we want to segment customers based on these
# 1. the most recent customers
# 2. the most frequent customers
# 3. the most high paying customers

# latest date
snap_date = df['InvoiceDate'].max()

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snap_date - x.max()).days,
    'InvoiceNo':'count',
    'TotalPrice':'sum'
})

# from the GroupByDF, consisting of dataframes for each group and their instances/rows and all columns/
# column values associated
# for each group now, only have three columns (leave the rest)
# and for each column, transform data accordingly (count, mean, etc)
# that's what .agg() does
# NOTE: the columns you will need will SHOULD EXACTLY MIRROR the names of the columns in main data

# all other columns will NOT be included for each row/instance of group; only 'InvoiceDate' column,
# 'InvoiceNo' column and 'TotalPrice' column will be associated for each group and instances

# NOTE: rfm turns into a DataFrame after you do .agg() and will look like
# | CustomerID | InvoiceDate | InvoiceNo | TotalPrice |
# | ---------- | ----------: | --------: | ---------: |
# | 101        |          12 |         3 |        500 |
# | 102        |          90 |         1 |         50 |
# | 103        |           5 |         8 |       1200 |

# why aggregate?
# because we need data (as a 2D matrix/dataframe) where for each customer (customer segmentation, of 
# course we need customers), we need their
# recency measure; captured by InvoiceDate
# frequency measure; captured by InvoiceNo
# monetary measure; captured by TotalPrice

# grouped by customer –> becomes GBDF
# aggregate it -> becomes DF again like the one you see above; a normal Pandas DataFrame

rfm.columns = ['Recency', 'Frequency', 'Monetary']

scaler = StandardScaler()

# scale data to a similar range to make all these measures comparable
rfm_scaled = scaler.fit_transform(rfm)

# 4 clusters
kmeans = KMeans(n_clusters=4)

rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
# assign, for each customer alongside their recency, frequency and monetary, their cluster (out of the 4)

fig3 = plt.figure()
sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm)

# see
st.pyplot(fig3)

# demand forecasting

st.subheader('Demand forecast (next 30 days)')

daily_sales_df = df.groupby('InvoiceDate')['TotalPrice'].sum().reset_index()
daily_sales_df.columns = ['ds', 'y']

model = Prophet()
# Prophet, by Meta, is a forecasting engine
# designed to detect: trends (sales going up/down), seasonality (weekly/monthly/patterns),
# noise (random fluctuations)

model.fit(daily_sales_df)
# studies/prepares data
# - are sales increasing over time?
# - do weekends have higher sales?
# - is there a repeating pattern every month

# it decomposes the the time series into trend + seasonanality + noise

future = model.make_future_dataframe(periods=30)

# 30 future dates where we predict/forecast demand

forecast = model.predict(future)

# output is something like:
# | ds    |            yhat | yhat_lower | yhat_upper |
# | ----- | --------------: | ---------: | ---------: |
# | Jan 1 |             300 |        ... |        ... |
# | Jan 2 |             150 |        ... |        ... |
# | Jan 3 |             180 |        ... |        ... |

# confidence intervals
# - yhat_lower; worst case estimate
# - yhat_upper;- best-case estimate

fig4 = model.plot(forecast)

# creates: black dots/lines - actual past sales | blue line - prediction sales trend
# shaded region - uncertainty region

st.pyplot(fig4)

# churn prediction

st.subheader('Churn Prediction')

last_purchase = df.groupby('CustomerID')['InvoiceDate'].max()

print(last_purchase)

# NOTE: here, df.groupby('CustomerID')['InvoiceDate'] is just a Series, not a GroupBySeries which
# consists of many Pandas Series
# because for each group/customer ID and multiple instances for it, there exists only 1 associated value 
# of 'InvoiceDate', not multiple
# and so index becomes customer ID and then values are invoice dates; hence a Series format and turned
# into a Series itself

churn = (snap_date - last_purchase).dt.days > 90

# snap_date - last_purchase –> computes time difference from snapshot date (date we're considering from
# for recency)
# .dt.days -> only extracts number of days
# > 90 -> checks if customer has been inactive for more than 90 days

churn_df = churn.reset_index()

churn_df.columns = ['CustomerID', 'Churn']

st.write(churn_df.head())
# gets top 10

