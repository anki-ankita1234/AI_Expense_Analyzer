import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
df = pd.read_csv('expenses.csv')
print("first 5 rows of dataset:\n",df.head())
print("\nMissing values:\n",df.isnull().sum())
category_total = df.groupby('Category')['Amount'].sum()
print("\nCategory-wise total expenses:\n",category_total)

plt.figure(figsize=(8,5))
sns.barplot(x=category_total.index, y=category_total.values, palette="viridis")
plt.title('Total Expenses by Category')
plt.ylabel('Amount')
plt.xlabel('Category')
plt.show()
daily_total = df.groupby('Date')['Amount'].sum()
plt.figure(figsize=(12,5))
sns.lineplot(x=daily_total.index, y=daily_total.values, marker='o', color='orange')
plt.title('Daily Expense Trend')
plt.ylabel('Amount')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.show()


df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
month_total = df.groupby('Month')['Amount'].sum()
print("\nMonthly total expenses:\n", month_total)

plt.figure(figsize=(6,4))
sns.barplot(x=month_total.index, y=month_total.values, palette="magma")
plt.title('Monthly Total Expenses')
plt.ylabel('Amount')
plt.xlabel('Month')
plt.show()