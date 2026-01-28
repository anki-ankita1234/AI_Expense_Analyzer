
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression


df = pd.read_csv('expenses.csv')

st.title("AI Expense Analyzer & Predictor")


st.header("Add New Expense")
date = st.date_input("Date")
category = st.selectbox("Category", ['Food', 'Transport', 'Shopping', 'Entertainment', 'Bills'])
amount = st.number_input("Amount", min_value=0)

if st.button("Add Expense"):
    new_data = pd.DataFrame({'Date':[date], 'Category':[category], 'Amount':[amount]})
    df = pd.concat([df, new_data], ignore_index=True)
    st.success(f"Added {category} expense of ₹{amount} on {date}")


st.header("Category-wise Total Expenses")
cat_total = df.groupby('Category')['Amount'].sum()
st.bar_chart(cat_total)


st.header("Daily Expense Trend")
daily_total = df.groupby('Date')['Amount'].sum()
st.line_chart(daily_total)


st.header("Predicted Expenses for Next 7 Days")
daily_total = df.groupby('Date')['Amount'].sum().reset_index()
daily_total['Day_Number'] = np.arange(len(daily_total))

X = daily_total[['Day_Number']]
y = daily_total['Amount']

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(daily_total), len(daily_total)+7).reshape(-1,1)
future_pred = model.predict(future_days)

pred_df = pd.DataFrame({'Day': np.arange(1,8), 'Predicted_Expense': future_pred})
st.table(pred_df)