#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

@st.cache_data
def load_and_process_data():
    df = pd.read_csv("customer_shopping_data.csv")
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True)
    df['total_amount'] = df['quantity'] * df['price']
    agg_df = df.groupby("customer_id").agg({
        "gender": "first",
        "age": "first",
        "total_amount": "sum",
        "invoice_no": "nunique",
        "quantity": "sum",
        "category": pd.Series.nunique,
        "shopping_mall": pd.Series.nunique,
        "payment_method": pd.Series.nunique,
        "invoice_date": ["max", "min"]
    }).reset_index()

    agg_df.columns = [
        "customer_id", "gender", "age", "total_spent", "num_transactions",
        "total_quantity", "unique_categories", "unique_malls", "unique_payment_methods",
        "last_purchase", "first_purchase"
    ]

    latest_date = df["invoice_date"].max()
    agg_df["recency_days"] = (latest_date - agg_df["last_purchase"]).dt.days
    agg_df["customer_lifetime_days"] = (agg_df["last_purchase"] - agg_df["first_purchase"]).dt.days

    agg_df["avg_order_value"] = agg_df["total_spent"] / agg_df["num_transactions"]
    agg_df["avg_items_per_order"] = agg_df["total_quantity"] / agg_df["num_transactions"]

    agg_df["avg_spent_per_category"] = agg_df["total_spent"] / agg_df["unique_categories"]
    agg_df["avg_spent_per_mall"] = agg_df["total_spent"] / agg_df["unique_malls"]

    le = LabelEncoder()
    agg_df["gender"] = le.fit_transform(agg_df["gender"])

    return agg_df

def cluster_customers(agg_df, n_clusters):
    features = [
        "gender", "age", "total_spent", "num_transactions",
        "total_quantity", "unique_categories", "unique_malls",
        "unique_payment_methods", "recency_days", "customer_lifetime_days",
        "avg_order_value", "avg_items_per_order",
        "avg_spent_per_category", "avg_spent_per_mall"
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg_df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    agg_df["Cluster"] = kmeans.fit_predict(X_scaled)

    return agg_df

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Dashboard")

agg_df = load_and_process_data()
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=4)
agg_df = cluster_customers(agg_df, n_clusters)

st.subheader("Cluster Summary")
cluster_summary = agg_df.groupby('Cluster').agg({
    'age': 'mean',
    'total_spent': 'mean',
    'num_transactions': 'mean',
    'total_quantity': 'mean',
    'recency_days': 'mean',
    'avg_order_value': 'mean'
}).round(2).reset_index()
st.dataframe(cluster_summary)

st.subheader("Customer Details")
selected_cluster = st.selectbox("Select Cluster", agg_df["Cluster"].unique())
st.dataframe(agg_df[agg_df["Cluster"] == selected_cluster])


# In[ ]:




