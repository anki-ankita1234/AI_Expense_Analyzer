
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('expenses.csv')


daily_total = df.groupby('Date')['Amount'].sum().reset_index()
daily_total['Day_Number'] = np.arange(len(daily_total))


X = daily_total[['Day_Number']]
y = daily_total['Amount']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


next_day = np.array([[len(daily_total)]])
prediction = model.predict(next_day)
print(f"Predicted expense for next day: ₹{prediction[0]:.2f}")


future_days = np.arange(len(daily_total), len(daily_total)+7).reshape(-1,1)
future_pred = model.predict(future_days)
print("\nPredicted expenses for next 7 days:")
for i, val in enumerate(future_pred):
    print(f"Day {i+1}: ₹{val:.2f}")