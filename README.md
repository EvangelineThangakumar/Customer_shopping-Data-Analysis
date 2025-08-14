# Customer_shopping-Data-Analysis

## Overview
This project focuses on analyzing customer purchase behavior from an e-commerce dataset to:
- **Segment customers** into groups using clustering algorithms (KMeans)
- **Recommend products** based on collaborative filtering
- **Visualize insights** through an interactive Streamlit dashboard

It combines **data analytics**, **machine learning**, and **interactive web app development** for actionable business intelligence.

## Project Structure
- Data set (customer_shopping_data.csv)
- Jupyter notebooks for EDA & modeling (Customer Shopping-Unsupervised.ipynb)
- Streamlit application (app.py)
- Streamlit output (output.pdf)

## Methodology
**Step 1: Data Preprocessing**
- Convert invoice_date to datetime
- Handle missing values
- Create total_amount = quantity * price
- Aggregate customer metrics

**Step 2: Feature Engineering**

**RFM Analysis:**
- Recency → Days since last purchase
- Frequency → Number of transactions
- Monetary → Total spend
- Standardize features using StandardScaler

**Step 3: Clustering**
- Apply KMeans with optimal cluster selection using the elbow method
- Assign each customer a Cluster label

**Step 5: Dashboard (Streamlit)**
- Cluster summary table (avg. age, quantity, spend)
- Interactive cluster filter
- Product recommendation module
