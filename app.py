import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

st.set_page_config(page_title="NeuralRetail Dashboard", layout="wide")

st.title("NeuralRetail - AI Retail Intelligence")


# LOAD DATA

@st.cache_data
def load_data():
    df = pd.read_excel("data/raw/online_retail.xlsx")
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

df = load_data()


# SIDEBAR FILTERS

st.sidebar.header("Filters")

country = st.sidebar.multiselect(
    "Select Country",
    df['Country'].unique(),
    default=df['Country'].unique()
)

df = df[df['Country'].isin(country)]


# KPIs

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"{df['TotalPrice'].sum():,.0f}")
col2.metric("Total Orders", df['InvoiceNo'].nunique())
col3.metric("Total Customers", df['CustomerID'].nunique())


# SALES TREND

st.subheader("📈 Sales Trend")

daily_sales = df.groupby('InvoiceDate')['TotalPrice'].sum()

fig1 = plt.figure()
daily_sales.plot()
plt.xlabel("Date")
plt.ylabel("Revenue")
st.pyplot(fig1)


# TOP PRODUCTS

st.subheader("🏆 Top Products")

top_products = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)

fig2 = plt.figure()
top_products.plot(kind='bar')
st.pyplot(fig2)


# CUSTOMER SEGMENTATION

st.subheader("👥 Customer Segmentation")

snapshot_date = df['InvoiceDate'].max()

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=4)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

fig3 = plt.figure()
sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm)
st.pyplot(fig3)


# DEMAND FORECASTING

st.subheader("🔮 Demand Forecast (Next 30 Days)")

daily_sales_df = df.groupby('InvoiceDate')['TotalPrice'].sum().reset_index()
daily_sales_df.columns = ['ds', 'y']

model = Prophet()
model.fit(daily_sales_df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

fig4 = model.plot(forecast)
st.pyplot(fig4)


# CHURN PREDICTION

st.subheader("⚠️ Churn Prediction")

last_purchase = df.groupby('CustomerID')['InvoiceDate'].max()
churn = (snapshot_date - last_purchase).dt.days > 90

churn_df = churn.reset_index()
churn_df.columns = ['CustomerID', 'Churn']

st.write(churn_df.head())


# INVENTORY RECOMMENDATION

st.subheader("📦 Inventory Recommendation")

forecast_30 = forecast[['ds','yhat']].tail(30)
stock = forecast_30['yhat'].sum()

st.metric("Recommended Stock (Next 30 Days)", f"{stock:,.0f}")


# DOWNLOAD OPTION

st.download_button(
    "Download Processed Data",
    df.to_csv(index=False),
    file_name="processed_data.csv"
)